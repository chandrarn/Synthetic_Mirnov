"""
Main orchestration script for sawtooth precursor analysis.

Coordinates the full pipeline:
1. Chi-squared mode identification (fit_oneshot_noRecomputation)
2. Burst detection (burst_detection)
3. Diamagnetic drift calculation (calc_diamagnetic_drift)
4. Results visualization and export
"""

from __future__ import annotations

import sys
import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

# Add project paths
SYNTHETIC_MIRNOV_DIR = Path(__file__).parent.parent
TARS_DIR = SYNTHETIC_MIRNOV_DIR.parent / "TARS"
CMOD_DIR = SYNTHETIC_MIRNOV_DIR / "C-Mod"

sys.path.insert(0, str(TARS_DIR))
sys.path.insert(0, str(CMOD_DIR))

from config import get_config, SawtoothAnalysisConfig
from burst_detection import (
    detect_bursts_connected_components,
    filter_bursts_by_region,
    print_burst_summary,
    SawtoothBurst,
)

# Import external modules
from tars.workflow.fit_oneshot_noRecomputation import (
    calc_chisq_all_modes_single_eq,
    plot_chisq_results,
    SEARCHED_MODES,
)
from tars.plotting.chisq_plots import plot_mode_chisq_diagnostic_grid

from calc_diamagnetic_drift import (
    load_profiles_for_shot,
    load_equilibrium_for_shot,
    compute_diamagnetic_drift_frequencies,
)


class SawtoothAnalysisPipeline:
    """Main orchestrator for sawtooth precursor analysis."""
    
    def __init__(
        self,
        scratch_dir: str,
        output_dir: str | None = None,
        use_multiprocessing: bool = True,
        max_workers: int = 4,
        debug: bool = False,
    ):
        """
        Initialize the analysis pipeline.
        
        Parameters
        ----------
        scratch_dir : str
            Path to TARS scratch directory containing input data
        output_dir : str, optional
            Output directory for results. Defaults to sawtooth_analysis/outputs
        use_multiprocessing : bool
            Whether to use multiprocessing for chi-squared computation
        max_workers : int
            Maximum number of worker processes
        debug : bool
            Enable debug output
        """
        self.scratch_dir = Path(scratch_dir)
        self.output_dir = (
            Path(output_dir) 
            if output_dir 
            else Path(__file__).parent / "outputs"
        )
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers
        self.debug = debug
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dataframe = None
        self.bursts_by_shot = {}
        self._last_detect_context: dict | None = None
        self._last_drift_context: dict | None = None

    @staticmethod
    def _extract_q1_series(value_grid: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
        """Extract a time series sampled at q=1 from [nspace, nt] grids."""
        value_arr = np.asarray(value_grid, dtype=float)
        q_arr = np.asarray(q_grid, dtype=float)
        if value_arr.shape != q_arr.shape:
            raise ValueError(
                f"Expected value and q grids to have identical shapes, got "
                f"{value_arr.shape} and {q_arr.shape}."
            )

        nt = value_arr.shape[1]
        out = np.full(nt, np.nan, dtype=float)
        for it in range(nt):
            q_col = q_arr[:, it]
            val_col = value_arr[:, it]
            valid = np.isfinite(q_col) & np.isfinite(val_col)
            if not np.any(valid):
                continue
            q_valid = q_col[valid]
            v_valid = val_col[valid]
            idx = int(np.argmin(np.abs(q_valid - 1.0)))
            out[it] = v_valid[idx]
        return out

    @staticmethod
    def _extract_toroidal_q1_series(profiles, result) -> np.ndarray:
        """Interpolate HIREX-SR toroidal rotation onto q=1 for each selected time."""
        return SawtoothAnalysisPipeline._extract_hirexsr_q1_series(
            profiles=profiles,
            result=result,
            values_grid=getattr(profiles, "omega_tor_rad_s", None),
        )

    @staticmethod
    def _extract_hirexsr_q1_series(profiles, result, values_grid) -> np.ndarray:
        """Interpolate an HIREX-SR profile quantity onto q=1 for each selected time."""
        nt = np.asarray(result.time_s).size
        out = np.full(nt, np.nan, dtype=float)

        if (
            values_grid is None
            or getattr(profiles, "psi_hx", None) is None
            or getattr(profiles, "time_diag", None) is None
        ):
            return out

        time_diag = np.asarray(profiles.time_diag, dtype=float)
        psi_hx = np.asarray(profiles.psi_hx, dtype=float)
        values = np.asarray(values_grid, dtype=float)
        q_grid = np.asarray(result.q, dtype=float)
        psi_q = np.asarray(result.psi_n, dtype=float)
        t_selected = np.asarray(result.time_s, dtype=float)

        if psi_hx.ndim != 2 or values.ndim != 2:
            return out

        npsi = min(psi_hx.shape[0], values.shape[0])
        nt = min(psi_hx.shape[1], values.shape[1], time_diag.size)
        psi_hx = psi_hx[:npsi, :nt]
        values = values[:npsi, :nt]
        time_diag = time_diag[:nt]

        for it, t_sel in enumerate(t_selected):
            q_col = q_grid[:, it]
            psi_col = psi_q[:, it]
            valid_q = np.isfinite(q_col) & np.isfinite(psi_col)
            if not np.any(valid_q):
                continue
            q_valid = q_col[valid_q]
            psi_valid = psi_col[valid_q]
            q1_idx = int(np.argmin(np.abs(q_valid - 1.0)))
            psi_at_q1 = psi_valid[q1_idx]

            tidx = int(np.argmin(np.abs(time_diag - t_sel)))
            psi_hx_col = psi_hx[:, tidx]
            values_col = values[:, tidx]
            valid_hx = np.isfinite(psi_hx_col) & np.isfinite(values_col)
            if np.count_nonzero(valid_hx) < 2:
                continue

            psi_sorted = psi_hx_col[valid_hx]
            values_sorted = values_col[valid_hx]
            order = np.argsort(psi_sorted)
            psi_sorted = psi_sorted[order]
            values_sorted = values_sorted[order]
            keep = np.concatenate(([True], np.diff(psi_sorted) > 1e-12))
            psi_sorted = psi_sorted[keep]
            values_sorted = values_sorted[keep]
            if psi_sorted.size < 2:
                continue

            out[it] = np.interp(
                psi_at_q1,
                psi_sorted,
                values_sorted,
                left=np.nan,
                right=np.nan,
            )

        return out

    @staticmethod
    def _interp_series_at_times(
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Linearly interpolate a 1D series onto target times, preserving NaNs out of range."""
        t_src = np.asarray(source_times, dtype=float)
        y_src = np.asarray(source_values, dtype=float)
        t_tgt = np.asarray(target_times, dtype=float)

        out = np.full(t_tgt.shape, np.nan, dtype=float)
        valid = np.isfinite(t_src) & np.isfinite(y_src)
        if np.count_nonzero(valid) < 2:
            return out

        t = t_src[valid]
        y = y_src[valid]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-12))
        t = t[keep]
        y = y[keep]
        if t.size < 2:
            return out

        out = np.interp(t_tgt, t, y, left=np.nan, right=np.nan)
        return out

    @staticmethod
    def _toroidal_to_khz(values: np.ndarray) -> np.ndarray:
        """Convert toroidal series to kHz if needed.

        HIREX-SR omega is often already reported in kHz in this code path.
        If magnitudes look like rad/s, convert to kHz.
        """
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return arr
        mag = float(np.nanmedian(np.abs(arr[finite])))
        if mag > 200.0:
            return arr / (2.0 * np.pi * 1e3)
        return arr

    @staticmethod
    def _normalize_target_modes(target_mode) -> List[tuple[int, int]]:
        """Normalize config target_mode into a list of (m, n) tuples."""
        if isinstance(target_mode, tuple):
            if len(target_mode) == 2 and all(isinstance(v, (int, np.integer)) for v in target_mode):
                return [(int(target_mode[0]), int(target_mode[1]))]
            if all(isinstance(v, tuple) and len(v) == 2 for v in target_mode):
                out: List[tuple[int, int]] = []
                for mode in target_mode:
                    n, m = mode
                    if not isinstance(n, (int, np.integer)) or not isinstance(m, (int, np.integer)):
                        raise ValueError(f"Invalid mode tuple {mode}; expected integer (m, n).")
                    out.append((int(n), int(m)))
                if out:
                    return out
        if isinstance(target_mode, list):
            out: List[tuple[int, int]] = []
            for mode in target_mode:
                if not isinstance(mode, tuple) or len(mode) != 2:
                    raise ValueError(
                        f"Invalid target_mode entry {mode}; expected list of integer (m, n) tuples."
                    )
                n, m = mode
                if not isinstance(n, (int, np.integer)) or not isinstance(m, (int, np.integer)):
                    raise ValueError(f"Invalid mode tuple {mode}; expected integer (m, n).")
                out.append((int(n), int(m)))
            if out:
                return out

        raise ValueError(
            "target_mode must be (m, n) or a non-empty list/tuple of (m, n) tuples."
        )
        
    def run_analysis(self, shot: int, config: SawtoothAnalysisConfig | None = None) -> pd.DataFrame:
        """
        Run complete analysis for a single shot.
        
        Parameters
        ----------
        shot : int
            Shot number
        config : SawtoothAnalysisConfig, optional
            Analysis configuration. If None, loads from config module.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with burst analysis results
        """
        if config is None:
            config = get_config(shot)
        
        print(f"\n{'='*70}")
        print(f"Analyzing Shot {shot}")
        print(f"{'='*70}")
        print(f"Target mode: {config.target_mode}")
        print(f"Time range: {config.time_range} s")
        print(f"Freq range: {np.array(config.freq_range)/1e3} kHz")
        print(f"Eq time index: {config.eq_time_idx}")
        print(f"Chi-squared threshold: {config.chisq_threshold}")
        # Step 1: Get chi-squared mode identification
        print("\n[1/3] Computing chi-squared mode identification...")
        ds_chisq = self._compute_chisq(shot, config)
        
        # Step 2: Detect bursts
        print("\n[2/3] Detecting sawtooth precursor bursts...")
        bursts = self._detect_bursts(ds_chisq, config)
        
        # Step 3: Get diamagnetic drift and rotation frequencies
        print("\n[3/3] Computing diamagnetic drift corrections...")
        results_df = self._compute_drift_and_rotation(shot, config, bursts)

        # Tag rows with shot/mode metadata and accumulate across shots.
        # NOTE: mode tuples are treated as (m, n) in this workflow.
        if not results_df.empty:
            results_df = results_df.copy()
            target_modes = self._normalize_target_modes(config.target_mode)
            m_values = np.asarray([mode[0] for mode in target_modes], dtype=float)
            n_values = np.asarray([mode[1] for mode in target_modes], dtype=float)
            results_df["shot"] = shot

            if np.all(np.isclose(n_values, n_values[0])):
                results_df["target_n"] = float(n_values[0])
            else:
                results_df["target_n"] = np.nan

            if np.all(np.isclose(m_values, m_values[0])):
                results_df["target_m"] = float(m_values[0])
            else:
                results_df["target_m"] = np.nan

            results_df["target_mode_label"] = "|".join(f"({m},{n})" for m, n in target_modes)
        
        self.bursts_by_shot[shot] = bursts
        if self.results_dataframe is None or self.results_dataframe.empty:
            self.results_dataframe = results_df
        else:
            self.results_dataframe = pd.concat(
                [self.results_dataframe, results_df],
                ignore_index=True,
            )
        self._last_config = config
        
        # Save results
        self._save_results(shot, config, results_df, bursts)
        
        return results_df
    
    def _compute_chisq(self, shot: int, config: SawtoothAnalysisConfig) -> xr.Dataset:
        """Compute chi-squared values for all modes."""
        cache_dir = self.scratch_dir / "chisq_results" / f"shot_{shot}" / "single_eq"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_dir = self.output_dir / f"shot_{shot}" / "chisq"
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_path = None
        output_path = None
        if config.eq_time_idx is not None:
            filename = f"chisq_single_eq_time{config.eq_time_idx}.nc"
            cache_path = cache_dir / filename
            output_path = output_dir / filename

            # TARS only reads cached chi-squared files from scratch/chisq_results.
            # If an earlier run wrote the file only under sawtooth_analysis/outputs,
            # seed the real cache location before invoking TARS.
            if output_path.exists() and not cache_path.exists():
                shutil.copy2(output_path, cache_path)

        # This calls the TARS fit_oneshot_noRecomputation module
        ds_chisq = calc_chisq_all_modes_single_eq(
            scratch_dir=str(self.scratch_dir),
            shot=shot,
            eq_time_idx=config.eq_time_idx,
            output_dir=str(cache_dir),
            use_multiprocessing=self.use_multiprocessing,
            max_workers=self.max_workers,
            debug=self.debug,
            forceRecompute=False,
        )

        if cache_path is not None and output_path is not None and cache_path.exists():
            shutil.copy2(cache_path, output_path)

        return ds_chisq
    
    def _detect_bursts(
        self,
        ds_chisq: xr.Dataset,
        config: SawtoothAnalysisConfig,
        min_area_points: int | None = None,
    ) -> List[SawtoothBurst]:
        """Detect sawtooth bursts in chi-squared data."""
        time = ds_chisq.time.data
        frequency = ds_chisq.frequency.data
        chisq_values = ds_chisq.chisq.data
        
        # Convert NaN to inf for processing
        chisq_clean = np.where(np.isnan(chisq_values), np.inf, chisq_values)
        
        # Get best mode at each time-frequency point
        best_mode_per_freq = np.argmin(chisq_clean, axis=2)
        best_chisq = np.min(chisq_clean, axis=2)
        
        # Build mode map for visualization/clustering
        # Mode indices to include in combined burst mask.
        target_modes = self._normalize_target_modes(config.target_mode)
        target_mode_indices = [SEARCHED_MODES.index(mode) for mode in target_modes]

        # Enforce configured frequency window at label stage so out-of-band points
        # are never tagged as sawtooth candidates.
        in_freq_band = (
            (np.asarray(frequency, dtype=float) >= float(config.freq_range[0]))
            & (np.asarray(frequency, dtype=float) <= float(config.freq_range[1]))
        )
        in_freq_band_2d = in_freq_band[np.newaxis, :]
        
        # Create binary map: include any selected mode that passes threshold.
        selected_mode_mask = np.isin(best_mode_per_freq, target_mode_indices)
        combined_mask = selected_mode_mask & (best_chisq <= config.chisq_threshold) & in_freq_band_2d
        mode_map = np.where(combined_mask, 1, -1)
        
        # Detect connected components (bursts)
        min_points = config.min_area_points if min_area_points is None else min_area_points
        bursts_all = detect_bursts_connected_components(
            mode_map=mode_map,
            target_mode=1,
            time=time,
            frequency=frequency,
            min_area_points=min_points,
            connectivity="moore",
        )
        
        # Filter to requested time-frequency region
        bursts = filter_bursts_by_region(
            bursts_all,
            time_range=config.time_range,
            freq_range=config.freq_range,
        )

        # Additional morphology filters to suppress tiny fragmented detections.
        filtered_bursts: List[SawtoothBurst] = []
        for burst in bursts:
            t0, t1 = burst.time_range
            f0, f1 = burst.freq_range
            if burst.n_points < config.min_area_points:
                continue
            # Reject bursts that extend outside configured frequency range.
            if f0 < config.freq_range[0] or f1 > config.freq_range[1]:
                continue
            if (t1 - t0) < config.min_time_span_s:
                continue
            if (f1 - f0) < config.min_freq_span_hz:
                continue
            filtered_bursts.append(burst)

        bursts = filtered_bursts

        self._last_detect_context = {
            "time": np.asarray(time, dtype=float),
            "frequency": np.asarray(frequency, dtype=float),
            "best_chisq": np.asarray(best_chisq, dtype=float),
            "target_mode_indices": [int(idx) for idx in target_mode_indices],
            "target_modes": list(target_modes),
            "mode_map": np.asarray(mode_map, dtype=float),
            "bursts": bursts,
        }
        
        print_burst_summary(bursts)
        
        return bursts
    
    def _compute_drift_and_rotation(
        self,
        shot: int,
        config: SawtoothAnalysisConfig,
        bursts: List[SawtoothBurst],
    ) -> pd.DataFrame:
        """Compute diamagnetic drift and HIREX_SR rotation for burst times."""
        
        if not bursts:
            print("No bursts detected; skipping drift calculation.")
            return pd.DataFrame()
        
        # Load profiles and equilibrium
        try:
            profiles = load_profiles_for_shot(
                shot,
                line=config.line,
                tht=config.tht,
            )
            equilibrium = load_equilibrium_for_shot(shot)
        except Exception as e:
            print(f"Warning: Could not load profiles/equilibrium: {e}")
            print("Skipping drift calculations.")
            return self._create_burst_dataframe_no_drift(bursts)
        
        # Select burst times for drift computation
        burst_times = [b.dominant_time for b in bursts]
        
        # Compute diamagnetic drift
        try:
            result = compute_diamagnetic_drift_frequencies(
                profiles=profiles,
                equilibrium=equilibrium,
                selected_times_s=burst_times,
                shot=shot,
                do_diagnostic_plot=False,
            )

            omega_star_e_q1 = self._extract_q1_series(
                result.omega_star_e_rad_s,
                result.q,
            )
            omega_star_i_q1 = self._extract_q1_series(
                result.omega_star_i_rad_s,
                result.q,
            )
            omega_tor_q1 = self._extract_toroidal_q1_series(profiles, result)
            omega_tor_q1 = self._toroidal_to_khz(omega_tor_q1)
            omega_tor_err_q1 = self._extract_hirexsr_q1_series(
                profiles=profiles,
                result=result,
                values_grid=getattr(profiles, "omega_tor_err_rad_s", None),
            )
            omega_tor_err_q1 = self._toroidal_to_khz(omega_tor_err_q1)

            omega_star_e_q1_at_bursts = self._interp_series_at_times(
                source_times=result.time_s,
                source_values=omega_star_e_q1,
                target_times=np.asarray(burst_times, dtype=float),
            )
            omega_star_i_q1_at_bursts = self._interp_series_at_times(
                source_times=result.time_s,
                source_values=omega_star_i_q1,
                target_times=np.asarray(burst_times, dtype=float),
            )
            omega_tor_q1_at_bursts = self._interp_series_at_times(
                source_times=result.time_s,
                source_values=omega_tor_q1,
                target_times=np.asarray(burst_times, dtype=float),
            )
            omega_tor_err_q1_at_bursts = self._interp_series_at_times(
                source_times=result.time_s,
                source_values=omega_tor_err_q1,
                target_times=np.asarray(burst_times, dtype=float),
            )

            # Build dense traces in the requested window for diagnostics.
            trace_context = {
                "time_s": np.asarray(result.time_s, dtype=float),
                "omega_star_e_q1": np.asarray(omega_star_e_q1, dtype=float),
                "omega_star_i_q1": np.asarray(omega_star_i_q1, dtype=float),
                "omega_tor_q1": np.asarray(omega_tor_q1, dtype=float),
                "omega_tor_q1_err": np.asarray(omega_tor_err_q1, dtype=float),
                "burst_time_s": np.asarray(burst_times, dtype=float),
                "omega_star_e_q1_burst": np.asarray(omega_star_e_q1_at_bursts, dtype=float),
                "omega_star_i_q1_burst": np.asarray(omega_star_i_q1_at_bursts, dtype=float),
                "omega_tor_q1_burst": np.asarray(omega_tor_q1_at_bursts, dtype=float),
                "omega_tor_q1_err_burst": np.asarray(omega_tor_err_q1_at_bursts, dtype=float),
            }

            try:
                trace_times = np.linspace(config.time_range[0], config.time_range[1], 200)
                trace_result = compute_diamagnetic_drift_frequencies(
                    profiles=profiles,
                    equilibrium=equilibrium,
                    selected_times_s=trace_times.tolist(),
                    shot=shot,
                    do_diagnostic_plot=False,
                )
                trace_context["time_s"] = np.asarray(trace_result.time_s, dtype=float)
                trace_context["omega_star_e_q1"] = self._extract_q1_series(
                    trace_result.omega_star_e_rad_s,
                    trace_result.q,
                )
                trace_context["omega_star_i_q1"] = self._extract_q1_series(
                    trace_result.omega_star_i_rad_s,
                    trace_result.q,
                )
                trace_context["omega_tor_q1"] = self._extract_toroidal_q1_series(
                    profiles,
                    trace_result,
                )
                trace_context["omega_tor_q1"] = self._toroidal_to_khz(
                    trace_context["omega_tor_q1"]
                )
            except Exception as trace_exc:
                print(f"Warning: Could not build dense q=1 traces: {trace_exc}")

            self._last_drift_context = trace_context

            drift_data = {
                "burst_id": [b.burst_id for b in bursts],
                "dominant_time_s": burst_times,
                "dominant_freq_hz": [b.dominant_freq for b in bursts],
                "omega_star_e_q1": omega_star_e_q1_at_bursts,
                "omega_star_i_q1": omega_star_i_q1_at_bursts,
                "omega_tor_q1": omega_tor_q1_at_bursts,
                "omega_tor_q1_err": omega_tor_err_q1_at_bursts,
            }
            
        except Exception as e:
            print(f"Warning: Drift computation failed: {e}")
            self._last_drift_context = None
            return self._create_burst_dataframe_no_drift(bursts)
        
        df = pd.DataFrame(drift_data)

        # Optional HIREX-SR quality cuts at burst times.
        err_lim = getattr(config, "max_hirexsr_omega_err_khz", None)
        snr_min = getattr(config, "min_hirexsr_snr", None)
        if not df.empty and "omega_tor_q1" in df.columns:
            keep = np.ones(len(df), dtype=bool)

            if err_lim is not None and "omega_tor_q1_err" in df.columns:
                err_vals = pd.to_numeric(df["omega_tor_q1_err"], errors="coerce").to_numpy(dtype=float)
                keep &= np.isfinite(err_vals)
                keep &= err_vals <= float(err_lim)

            if snr_min is not None and "omega_tor_q1_err" in df.columns:
                tor_vals = pd.to_numeric(df["omega_tor_q1"], errors="coerce").to_numpy(dtype=float)
                err_vals = pd.to_numeric(df["omega_tor_q1_err"], errors="coerce").to_numpy(dtype=float)
                snr = np.full_like(tor_vals, np.nan, dtype=float)
                valid = np.isfinite(tor_vals) & np.isfinite(err_vals) & (err_vals > 0.0)
                snr[valid] = np.abs(tor_vals[valid]) / err_vals[valid]
                keep &= np.isfinite(snr)
                keep &= snr >= float(snr_min)
                df["omega_tor_q1_snr"] = snr

            n_before = len(df)
            df = df.loc[keep].reset_index(drop=True)
            n_rejected = n_before - len(df)
            if n_rejected > 0:
                print(
                    "Filtered "
                    f"{n_rejected}/{n_before} burst points by HIREX-SR quality cuts "
                    f"(max_err={err_lim}, min_snr={snr_min})."
                )

        return df
    
    def _create_burst_dataframe_no_drift(self, bursts: List[SawtoothBurst]) -> pd.DataFrame:
        """Create dataframe with burst data but without drift calculations."""
        return pd.DataFrame({
            "burst_id": [b.burst_id for b in bursts],
            "dominant_time_s": [b.dominant_time for b in bursts],
            "dominant_freq_hz": [b.dominant_freq for b in bursts],
            "n_points": [b.n_points for b in bursts],
            "area_seconds_hz": [b.area_seconds_hz for b in bursts],
        })
    
    def _save_results(
        self,
        shot: int,
        config: SawtoothAnalysisConfig,
        results_df: pd.DataFrame,
        bursts: List[SawtoothBurst],
    ) -> None:
        """Save results to CSV and create diagnostic plots."""
        shot_dir = self.output_dir / f"shot_{shot}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        csv_path = shot_dir / "burst_analysis.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nSaved burst analysis to: {csv_path}")

        self._plot_mode_blob_overlay(shot=shot, config=config, shot_dir=shot_dir)
        if getattr(config, "plot_diamagnetic_drifts", True):
            self._plot_q1_time_traces(shot=shot, config=config, shot_dir=shot_dir)
        else:
            print("Skipping q=1 diamagnetic drift trace plot (plot_diamagnetic_drifts=False).")

    def _plot_mode_blob_overlay(
        self,
        shot: int,
        config: SawtoothAnalysisConfig,
        shot_dir: Path,
    ) -> None:
        """Overlay burst outlines on a chi-squared map for visual verification."""
        ctx = self._last_detect_context
        if ctx is None:
            return

        time = ctx["time"]
        frequency = ctx["frequency"]
        best_chisq = ctx["best_chisq"]
        bursts: List[SawtoothBurst] = ctx["bursts"]

        if best_chisq.size == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        mesh = ax.pcolormesh(
            time,
            frequency * 1e-3,
            np.asarray(best_chisq, dtype=float).T,
            shading="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            zorder=-5,
        )
        fig.colorbar(mesh, ax=ax, label=r"$\chi^2$")

        # Draw each connected component as a contour outline.
        for burst in bursts:
            burst_mask = np.zeros_like(best_chisq, dtype=float)
            burst_mask[burst.time_indices, burst.freq_indices] = 1.0
            if np.count_nonzero(burst_mask) == 0:
                continue
            ax.contour(
                time,
                frequency * 1e-3,
                burst_mask.T,
                levels=[0.5],
                colors="k",
                linewidths=2.2,
                alpha=0.95,
            )
            ax.contour(
                time,
                frequency * 1e-3,
                burst_mask.T,
                levels=[0.5],
                colors="#00e5ff",
                linewidths=1.1,
                alpha=0.95,
                zorder=-5
            )

        ax.set_title(
            f"Shot {shot}: Target Mode {config.target_mode} Burst Blobs over Best-$\\chi^2$"
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [kHz]")
        ax.set_xlim(config.time_range)
        ax.set_ylim(config.freq_range[0] * 1e-3, config.freq_range[1] * 1e-3)
        ax.grid(alpha=0.25)
        ax.set_rasterization_zorder(-1)

        overlay_path = shot_dir / f"mode_blob_overlay_shot_{shot}.pdf"
        fig.savefig(str(overlay_path), transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved mode-blob overlay to: {overlay_path}")

    def _plot_q1_time_traces(
        self,
        shot: int,
        config: SawtoothAnalysisConfig,
        shot_dir: Path,
        y_lims: Tuple[float, float] | None = (-25, 25),
    ) -> None:
        """Plot q=1 time traces for toroidal rotation and diamagnetic drift terms."""
        ctx = self._last_drift_context
        if ctx is None:
            return

        t = np.asarray(ctx["time_s"], dtype=float)
        w_tor = np.asarray(ctx["omega_tor_q1"], dtype=float)
        w_se = np.asarray(ctx["omega_star_e_q1"], dtype=float)
        w_si = np.asarray(ctx["omega_star_i_q1"], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.plot(t, w_tor, lw=2.0, label=r"$f_{\phi, q=1}$ (HIREX-SR)")
        ax.plot(t, w_se / (2.0 * np.pi * 1e3), lw=1.8, label=r"$f_{*e, q=1}$")
        ax.plot(t, w_si / (2.0 * np.pi * 1e3), lw=1.8, label=r"$f_{*i, q=1}$")

        tb = np.asarray(ctx["burst_time_s"], dtype=float)
        wb = np.asarray(ctx["omega_tor_q1_burst"], dtype=float)
        finite_burst = np.isfinite(tb) & np.isfinite(wb)
        if np.any(finite_burst):
            ax.scatter(
                tb[finite_burst],
                wb[finite_burst],
                s=26,
                c="k",
                alpha=0.7,
                label="Sawtooth precursor datapoints",
            )

        ax.set_title(f"Shot {shot}: q=1 Frequency Traces in Analysis Window")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [kHz]")
        if y_lims is not None:
            ax.set_ylim(*y_lims)
        ax.set_xlim(config.time_range)
        ax.grid(alpha=0.25)
        ax.legend()

        trace_path = shot_dir / f"q1_frequency_traces_shot_{shot}.pdf"
        fig.savefig(str(trace_path), transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved q=1 trace plot to: {trace_path}")
    
    def create_scatter_plot(
        self,
        output_path: str | Path | None = None,
        x_limits: Tuple[float, float] | None = None,
        y_limits: Tuple[float, float] | None = None,
        n_mode: float | None = None,
        include_diamagnetic_drifts: bool = True,
    ) -> None:
        """
        Create scatter plot of n=1 frequency vs q=1 rotation frequency.
        
        Combines results from all analyzed shots.

        Parameters
        ----------
        output_path : str | Path | None
            File path to save the plot. Uses default output path when None.
        x_limits : tuple[float, float] | None
            Optional x-axis limits (xmin, xmax). When None, matplotlib autoscale is used.
        y_limits : tuple[float, float] | None
            Optional y-axis limits (ymin, ymax). When None, matplotlib autoscale is used.
        include_diamagnetic_drifts : bool
            Whether to overlay electron/ion diamagnetic drift points.
        """
        if self.results_dataframe is None or len(self.results_dataframe) == 0:
            print("No results to plot.")
            return
        
        df = self.results_dataframe

        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')

        if n_mode is None:
            if "target_n" in df.columns:
                n_series = pd.to_numeric(df["target_n"], errors="coerce").astype(float)
                n_label = "per-shot n"
            else:
                last_cfg = getattr(self, "_last_config", None)
                inferred_n = -1.0
                if last_cfg is not None and getattr(last_cfg, "target_mode", None) is not None:
                    inferred_modes = self._normalize_target_modes(last_cfg.target_mode)
                    # mode tuples are treated as (m, n), so n is element 1
                    n_candidates = np.asarray([mode[1] for mode in inferred_modes], dtype=float)
                    if n_candidates.size > 0 and np.all(np.isclose(n_candidates, n_candidates[0])):
                        inferred_n = float(n_candidates[0])
                n_series = pd.Series(inferred_n, index=df.index, dtype=float)
                n_label = f"n={inferred_n:g}"
        else:
            n_series = pd.Series(float(n_mode), index=df.index, dtype=float)
            n_label = f"n={float(n_mode):g}"

        valid_n = np.isfinite(n_series.to_numpy()) & (~np.isclose(n_series.to_numpy(), 0.0))
        if not np.any(valid_n):
            print("No valid non-zero n values available; cannot divide frequencies by n.")
            plt.close(fig)
            return

        if "dominant_freq_hz" not in df.columns:
            print("Missing dominant_freq_hz column; cannot create scatter plot.")
            plt.close(fig)
            return

        y_precursor = (df["dominant_freq_hz"] / 1e3) / n_series

        if "omega_tor_q1" in df.columns and np.isfinite(df["omega_tor_q1"]).any():
            x_tor = pd.to_numeric(df["omega_tor_q1"], errors="coerce")
            valid_prec = np.isfinite(x_tor.to_numpy()) & np.isfinite(y_precursor.to_numpy()) & valid_n

            # Requested mapping: y = f/n (signed), x = q=1 toroidal frequency (signed).
            ax.scatter(
                x_tor[valid_prec],
                y_precursor[valid_prec],
                alpha=0.75,
                s=90,
                label=rf"Precursor: $(f_{{\mathrm{{prec}}}}/n)$, {n_label}",
            )

            # Optional drift overlays on the same signed axes.
            if include_diamagnetic_drifts and "omega_star_e_q1" in df.columns:
                y_e = (df["omega_star_e_q1"] / (2.0 * np.pi * 1e3)) / n_series
                valid_e = np.isfinite(x_tor.to_numpy()) & np.isfinite(y_e.to_numpy()) & valid_n
                ax.scatter(
                    x_tor[valid_e],
                    y_e[valid_e],
                    alpha=0.6,
                    s=70,
                    label=rf"Electron drift: $(f_{{*e,q=1}}/n)$, {n_label}",
                )

            if include_diamagnetic_drifts and "omega_star_i_q1" in df.columns:
                y_i = (df["omega_star_i_q1"] / (2.0 * np.pi * 1e3)) / n_series
                valid_i = np.isfinite(x_tor.to_numpy()) & np.isfinite(y_i.to_numpy()) & valid_n
                ax.scatter(
                    x_tor[valid_i],
                    y_i[valid_i],
                    alpha=0.6,
                    s=70,
                    label=rf"Ion drift: $(f_{{*i,q=1}}/n)$, {n_label}",
                )
        else:
            y_vals = y_precursor.to_numpy()
            valid_y = np.isfinite(y_vals) & valid_n
            ax.scatter(np.arange(len(df))[valid_y], y_vals[valid_y], alpha=0.7, s=90, label="Precursor")
            ax.set_xlabel("Burst index")

        ax.set_xlabel(r"$f_{\phi,q=1}$ (kHz)")
        ax.set_ylabel(r"Frequency$/n$ (kHz)")

        if x_limits is not None:
            ax.set_xlim(*x_limits)
        if y_limits is not None:
            ax.set_ylim(*y_limits)

        x_lo, x_hi = ax.get_xlim()
        y_lo, y_hi = ax.get_ylim()
        lo = min(x_lo, y_lo)
        hi = max(x_hi, y_hi)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, lw=1.2, label="x = y")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.grid(alpha=0.3)
        
        if output_path is None:
            shot_label = "unknown"
            if "shot" in df.columns:
                shot_vals = pd.to_numeric(df["shot"], errors="coerce")
                finite_shots = shot_vals[np.isfinite(shot_vals)].astype(int).to_numpy()
                unique_shots = np.unique(finite_shots)
                if unique_shots.size == 1:
                    shot_label = f"shot_{unique_shots[0]}"
                elif unique_shots.size > 1:
                    if unique_shots.size <= 4:
                        shot_label = "shots_" + "_".join(str(s) for s in unique_shots)
                    else:
                        shot_label = (
                            f"shots_{unique_shots.min()}_to_{unique_shots.max()}"
                            f"_{unique_shots.size}total"
                        )
            output_path = self.output_dir / f"sawtooth_vs_rotation_scatter_{shot_label}.pdf"
        else:
            output_path = Path(output_path)
        
        fig.savefig(str(output_path), transparent=True, bbox_inches="tight")
        print(f"Saved scatter plot to: {output_path}")
        plt.close(fig)


def main():
    """Example: Run analysis for one shot."""
    # Configuration
    scratch_dir = "/home/rianc/Documents/TARS/tars/scratch"
    output_dir = "/home/rianc/Documents/Synthetic_Mirnov/sawtooth_analysis/outputs"
    shot = 1120927023
    
    # Initialize pipeline
    pipeline = SawtoothAnalysisPipeline(
        scratch_dir=scratch_dir,
        output_dir=output_dir,
        use_multiprocessing=True,
        max_workers=4,
        debug=True,
    )
    
    # Run analysis
    config = get_config(shot)
    results_df = pipeline.run_analysis(shot, config)
    
    print("\nAnalysis Results:")
    print(results_df)
    
    # Create visualization
    pipeline.create_scatter_plot()


if __name__ == "__main__":
    main()
