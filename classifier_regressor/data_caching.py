#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data extraction and caching utilities for synthetic spectrogram datasets.

This module handles:
- Loading xarray NetCDF datasets
- Extracting sensor names and ordering them consistently
- Defining frequency regions around mode frequencies
- Averaging spectrograms over frequency bands or selecting peak frequencies
- Caching preprocessed training data to NPZ for fast iteration
- Optional geometry (theta, phi) injection from C-Mod bp_k data
"""
from __future__ import annotations
import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import xarray as xr

# Allow importing C-Mod utilities for geometry (optional)
import sys
sys.path.append('../C-Mod/')
try:
    from get_Cmod_Data import __loadData  # for bp_k geometry
except Exception:
    __loadData = None


# -------------------------
# Sensor ordering utilities
# -------------------------
def gen_sensor_ordering(sensor_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Return (sorted_names, original_names_array, order_indices) for a consistent ordering:
      1) BPXX_ABK, BPXX_GHK (XX numeric)
      2) BPXT_ABK, BPXT_GHK (XX numeric)
      3) BP_AA_BOT (AA letters) [all BOT first]
      4) BP_AA_TOP (AA letters)
    Case-insensitive matching; unknowns go to the end, lexicographic.
    """
    def _key(u: str):
        u = u.upper()
        m = re.match(r'^BP(\d+)_([A-Z]+)$', u)  # BPXX_ABK/GHK
        if m:
            num = int(m.group(1)); suf = m.group(2)
            sub = 0 if suf == 'ABK' else 1 if suf == 'GHK' else 2
            return (0, 0, num, sub, u)
        m = re.match(r'^BP(\d+)T_([A-Z]+)$', u)  # BPXT_ABK/GHK
        if m:
            num = int(m.group(1)); suf = m.group(2)
            sub = 0 if suf == 'ABK' else 1 if suf == 'GHK' else 2
            return (1, 0, num, sub, u)
        m = re.match(r'^BP_([A-Z]{2})_((?:TOP|BOT))$', u)  # BP_AA_TOP/BOT
        if m:
            aa = m.group(1); pos = m.group(2)
            # BOT category before TOP category
            cat = 2 if pos == 'BOT' else 3
            return (cat, 0, aa, 0 if pos == 'BOT' else 1, u)
        return (9, 9, u, 9, u)

    names_arr = np.array(sensor_names).astype(str)
    order = [i for i, _ in sorted(enumerate(names_arr), key=lambda t: _key(t[1]))]
    return names_arr[order], names_arr, order


# -------------------------
# Data extraction utilities
# -------------------------
@dataclass
class CacheConfig:
    """Configuration for dataset caching."""
    data_dir: str
    out_path: str
    num_timepoints: int = -1   # -1 => use all timepoints per dataset
    freq_tolerance: float = 0.1  # fractional band around mode frequency
    include_geometry: bool = True
    geometry_shot: Optional[int] = None  # If None and include_geometry, will try bp_k from this shot
    use_mode: str = 'n'  # 'n' or 'm'
    n_datasets: int = -1  # -1 => load all datasets


def _open_datasets(data_dir: str, n_datasets: int = -1) -> List[xr.Dataset]:
    """Load NetCDF datasets from directory."""
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')])
    datasets = []
    for fp in files:
        if n_datasets > 0 and len(datasets) >= n_datasets:
            break
        try:
            ds = xr.open_dataset(fp)
            ds.load()  # bring into memory for speed
            datasets.append(ds)
        except Exception as e:
            print(f"Warning: failed to open {fp}: {e}")
    return datasets


def _extract_sensor_names(ds: xr.Dataset) -> List[str]:
    """Extract base sensor names from dataset variables (strip _real/_imag suffixes)."""
    sn = [v for v in ds.data_vars if 'Mode' not in v]
    # Keep base names by stripping suffix; preserve pairing order: take unique in order
    base = []
    seen = set()
    for name in sn:
        if name.endswith('_real') or name.endswith('_imag'):
            b = name[:-5]
            if b not in seen:
                base.append(b)
                seen.add(b)
    return base


def _define_mode_regions_time(ds: xr.Dataset, F_mode_vars: List[str], time_indices: np.ndarray,
                              freq_tolerance: float) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Return regions per mode per selected time index: regions[mode][time_idx] = (freq_inds, sensor_inds)
    If frequency at timepoint is zero, freq_inds will be empty. sensor_inds = all sensors by default.
    Expands tolerance until at least one frequency bin falls in the band.
    """
    n_sensors = len(_extract_sensor_names(ds))
    freq_vals = ds['frequency'].values
    df = float(freq_vals[1] - freq_vals[0]) if len(freq_vals) > 1 else 0.0

    regions = []
    for mode_var in F_mode_vars:
        mode_regions = []
        freq_series = ds[mode_var].values  # (time,)
        for ti in time_indices:
            f0 = float(freq_series[ti])
            if f0 == 0 or not np.isfinite(f0):
                mode_regions.append((np.array([], dtype=int), np.arange(n_sensors)))
                continue
            tol = float(freq_tolerance)
            # ensure at least one bin
            fmin = fmax = f0
            while (fmax - fmin) < max(df, 1e-12):
                fmin = f0 * (1 - tol)
                fmax = f0 * (1 + tol)
                tol *= 1.5
                if tol > 1.0:  # safety
                    break
            freq_inds = np.where((freq_vals >= fmin) & (freq_vals <= fmax))[0]
            if freq_inds.size == 0:
                mode_regions.append((np.array([], dtype=int), np.arange(n_sensors)))
            else:
                mode_regions.append((freq_inds, np.arange(n_sensors)))
        regions.append(mode_regions)
    return regions


def _avg_band_per_sensor(ds: xr.Dataset, time_index: int, freq_inds: np.ndarray,
                         sensor_bases: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Select a single representative frequency within ``freq_inds`` and average over time.

    New behavior (modified as requested):
        1. Within the provided ``freq_inds`` window, find the frequency index whose
           real component (spectrogram) has the maximum average value across ALL
           sensors AND ALL timepoints.
        2. For that chosen frequency index, compute (per sensor) the average real and
           imaginary components over time.
        3. Return (avg_real[S], avg_imag[S]) where S = number of sensors.

    Notes:
        * ``time_index`` is retained for signature compatibility but is ignored in the
          new logic (the selection uses all timepoints).
        * If ``freq_inds`` is empty, returns zeros.
    """
    S = len(sensor_bases)
    real_out = np.zeros(S, dtype=np.float32)
    imag_out = np.zeros(S, dtype=np.float32)
    if freq_inds.size == 0:
        return real_out, imag_out

    # Build a (S, F_sub) array of real parts for the candidate frequency window
    # This allows computing the global mean across sensors for each frequency at this timepoint.
    try:
        real_stack = np.stack([ds[f"{base}_real"].values[time_index, freq_inds] for base in sensor_bases], axis=0)
    except KeyError as e:
        # In case a sensor variable is missing; return zeros gracefully
        print(f"Warning: missing sensor variable while stacking real components: {e}")
        return real_out, imag_out

    # real_stack shape: (S, F_sub)
    # Mean across sensors (axis=0) -> frequency profile length F_sub
    mean_freq = real_stack.mean(axis=0)  # (F_sub,)
    # Select frequency (within freq_inds) with maximum mean real value
    local_best = int(np.argmax(mean_freq))
    chosen_freq_idx = int(freq_inds[local_best])

    # Average real & imag over time at chosen frequency index per sensor
    for si, base in enumerate(sensor_bases):
        try:
            real_vals = ds[f"{base}_real"].values[:, chosen_freq_idx]
            imag_vals = ds[f"{base}_imag"].values[:, chosen_freq_idx]
        except KeyError:
            real_vals = np.array([])
            imag_vals = np.array([])
        real_out[si] = float(real_vals.mean()) if real_vals.size else 0.0
        imag_out[si] = float(imag_vals.mean()) if imag_vals.size else 0.0

    return real_out, imag_out


def _get_mode_labels(ds: xr.Dataset, F_mode_vars: List[str], use_mode: str) -> List[int]:
    """Return per-mode integer labels from dataset attrs for 'm' or 'n'."""
    key = 'mode_n' if use_mode.lower() == 'n' else 'mode_m'
    if key not in ds.attrs:
        # Fallback: try variable or default zeros
        print(f"Warning: {key} not found in attrs; using zeros")
        return [0] * len(F_mode_vars)
    vals = ds.attrs[key]
    # Ensure list length equals number of modes
    try:
        vals_list = list(vals) if np.ndim(vals) > 0 else [int(vals)]
    except Exception:
        vals_list = [int(vals)]
    if len(vals_list) == 1 and len(F_mode_vars) > 1:
        vals_list = vals_list * len(F_mode_vars)
    if len(vals_list) != len(F_mode_vars):
        raise ValueError(f"Length of {key} in attrs ({len(vals_list)}) != number of modes ({len(F_mode_vars)})")
    return [int(v) for v in vals_list]


def _maybe_geometry_from_bp_k(geometry_shot: Optional[int], names_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attempt to compute theta, phi per sensor using bp_k geometry for a given shot.
    Theta here is arctan2(Z, R - r_maj) with r_maj ~ 0.68 (C-Mod); Phi comes from bp_k.Phi.
    Names are matched case-insensitively. Returns (theta[S], phi[S]); zeros if unavailable.
    """
    S = len(names_sorted)
    th = np.zeros(S, dtype=np.float32)
    ph = np.zeros(S, dtype=np.float32)
    if __loadData is None or geometry_shot is None:
        return th, ph
    try:
        bp_k = __loadData(int(geometry_shot), pullData='bp_k', forceReload=['bp_k'] * False)['bp_k']
        name_map = {str(n).upper(): i for i, n in enumerate(bp_k.names)}
        rmaj = 0.68
        for i, nm in enumerate(names_sorted):
            u = str(nm).upper()
            if u in name_map:
                j = name_map[u]
                R = float(bp_k.R[j]); Z = float(bp_k.Z[j]); Phi = float(bp_k.Phi[j])
                th[i] = math.atan2(Z, R - rmaj)
                ph[i] = Phi
    except Exception as e:
        print(f"Warning: could not load geometry from bp_k: {e}")
    return th, ph


# -------------------------
# Main caching function
# -------------------------
def cache_training_dataset(cfg: CacheConfig) -> Dict[str, np.ndarray]:
    """
    Build dataset from NetCDFs and cache to cfg.out_path (.npz).
    Saved content:
      - X_ri: (N, S, 2) real/imag averaged in the mode band for each sample
      - y:    (N,) integer labels (n or m)
      - sensor_names: (S,) strings, in sorted order
      - theta: (S,) optional if include_geometry
      - phi:   (S,) optional if include_geometry
      - meta:  dict with bookkeeping (json string)
    Returns loaded dict of arrays.
    """
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    if os.path.exists(cfg.out_path):
        data = np.load(cfg.out_path, allow_pickle=True)
        print(f"Loaded cached dataset from {cfg.out_path} :: X_ri={data['X_ri'].shape}, y={data['y'].shape}")
        return {k: data[k] for k in data.files}

    datasets = _open_datasets(cfg.data_dir, n_datasets=cfg.n_datasets)
    if not datasets:
        raise FileNotFoundError(f"No NetCDF datasets found in {cfg.data_dir}")

    # Use the first dataset to determine sensor ordering, then apply globally
    base_sensor_names = _extract_sensor_names(datasets[0])
    names_sorted, names_orig, order_idx = gen_sensor_ordering(base_sensor_names)

    X_ri_list: List[np.ndarray] = []
    y_list: List[int] = []
    used_files: List[str] = []
    used_counts: List[int] = []

    for ds in datasets:
        sensor_bases = _extract_sensor_names(ds)
        # Reorder indices to match global sorted names; if any name missing, fill zeros later
        # Build mapping from sorted names to indices in this dataset
        idx_map = []
        name_to_idx = {b: i for i, b in enumerate(sensor_bases)}
        for nm in names_sorted:
            idx_map.append(name_to_idx.get(nm, -1))

        times = np.arange(len(ds['time']))
        if cfg.num_timepoints is not None and cfg.num_timepoints > 0 and cfg.num_timepoints < len(times):
            # random subset for caching speed
            rng = np.random.default_rng(42)
            times = np.sort(rng.choice(times, size=cfg.num_timepoints, replace=False))

        F_mode_vars = [v for v in ds.data_vars if v.startswith('F_Mode_')]
        mode_labels = _get_mode_labels(ds, F_mode_vars, cfg.use_mode)
        regions = _define_mode_regions_time(ds, F_mode_vars, times, cfg.freq_tolerance)

        n_added = 0
        for ti_i, ti in enumerate(times):
            for m_i, mode_var in enumerate(F_mode_vars):
                freq_inds, _sensor_inds = regions[m_i][ti_i]
                if freq_inds.size == 0:
                    continue  # skip zero-frequency times
                real_band, imag_band = _avg_band_per_sensor(ds, ti, freq_inds, sensor_bases)
                # Reorder to global sorted names; missing names -> 0
                S = len(names_sorted)
                real_sorted = np.zeros(S, dtype=np.float32)
                imag_sorted = np.zeros(S, dtype=np.float32)
                for s_idx, src in enumerate(idx_map):
                    if src >= 0:
                        real_sorted[s_idx] = real_band[src]
                        imag_sorted[s_idx] = imag_band[src]
                X_ri_list.append(np.stack([real_sorted, imag_sorted], axis=-1))  # (S, 2)
                y_list.append(mode_labels[m_i])
                n_added += 1
        used_files.append(getattr(ds, 'encoding', {}).get('source', ''))
        used_counts.append(n_added)
        print(f"Accumulated {n_added} samples from dataset")

    if not X_ri_list:
        raise RuntimeError("No samples accumulated. Check frequency bands and tolerance.")

    X_ri = np.stack(X_ri_list, axis=0)  # (N, S, 2)
    y = np.array(y_list, dtype=np.int64)

    theta = phi = None
    if cfg.include_geometry:
        theta, phi = _maybe_geometry_from_bp_k(cfg.geometry_shot, names_sorted)

    meta = {
        'data_dir': cfg.data_dir,
        'num_samples': int(X_ri.shape[0]),
        'num_sensors': int(X_ri.shape[1]),
        'used_files': used_files,
        'used_counts': used_counts,
        'use_mode': cfg.use_mode,
    }

    save_dict = {
        'X_ri': X_ri,
        'y': y,
        'sensor_names': names_sorted,
        'meta': np.array(json.dumps(meta)),
    }
    if theta is not None and phi is not None:
        save_dict['theta'] = theta
        save_dict['phi'] = phi

    np.savez(cfg.out_path, **save_dict)
    print(f"Saved cached dataset to {cfg.out_path} :: X_ri={X_ri.shape}, y={y.shape}")

    return save_dict


# -------------------------
# High-level interface
# -------------------------
def build_or_load_cached_dataset(data_dir: str, out_path: str, use_mode: str = 'n',
                                 include_geometry: bool = True, geometry_shot: Optional[int] = None,
                                 num_timepoints: int = -1, freq_tolerance: float = 0.1,
                                 n_datasets: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                                  Optional[np.ndarray], Optional[np.ndarray]]:
    """
    High-level function to build or load a cached training dataset.
    
    Args:
        data_dir: Directory containing NetCDF files
        out_path: Path to save/load cached NPZ file
        use_mode: 'n' or 'm' for target labels
        include_geometry: Whether to include theta/phi geometry
        geometry_shot: C-Mod shot number for bp_k geometry
        num_timepoints: Number of timepoints per dataset (-1 for all)
        freq_tolerance: Fractional tolerance for frequency band
        n_datasets: Number of datasets to load (-1 for all)
        
    Returns:
        X_ri: (N, S, 2) array of [real, imag] features
        y: (N,) array of labels
        sensor_names: (S,) array of sensor names in sorted order
        theta: (S,) array of theta coordinates (or None)
        phi: (S,) array of phi coordinates (or None)
    """
    cfg = CacheConfig(data_dir=data_dir, out_path=out_path, num_timepoints=num_timepoints,
                      freq_tolerance=freq_tolerance, include_geometry=include_geometry,
                      geometry_shot=geometry_shot, use_mode=use_mode, n_datasets=n_datasets)
    dat = cache_training_dataset(cfg)
    X_ri = dat['X_ri'] if isinstance(dat, dict) else dat.get('X_ri')
    y = dat['y'] if isinstance(dat, dict) else dat.get('y')
    sensor_names = dat['sensor_names'] if isinstance(dat, dict) else dat.get('sensor_names')
    theta = dat.get('theta', None) if isinstance(dat, dict) else None
    phi = dat.get('phi', None) if isinstance(dat, dict) else None
    return X_ri, y, sensor_names, theta, phi


if __name__ == "__main__":
    # Quick test/example
    print("Data caching module for synthetic spectrogram datasets.")
    print("Import this module and use build_or_load_cached_dataset() to prepare training data.")
