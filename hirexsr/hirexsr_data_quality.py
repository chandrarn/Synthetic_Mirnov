"""Data quality scanning for HIREX-SR shots.

Loads line-integrated and/or inverted-profile data for a list of shots/lines
and applies quality filters beyond simple data existence.

Quality filters applied per (shot, line):
  1. At least ``min_timepoints`` time slices must have velocity data (finite,
     not the -1 sentinel, and passing optional error-bar thresholds).
  2. Of those valid time slices, at least ``min_channels`` spatial channels
     must have good velocity data for the time slice to count.

Error-bar thresholds (all optional):
  - ``max_verr_abs``  : |verr| <= threshold  [km/s for lint; kHz for profile omega]
  - ``max_verr_rel``  : |verr| / |v| <= threshold  (fraction)
  - ``max_tierr_abs`` : |tierr| <= threshold  [keV]
  - ``max_tierr_rel`` : |tierr| / |ti| <= threshold  (fraction)

Usage (CLI)
-----------
    python hirexsr_data_quality.py --csv cmod_logbook_ntm_shots.csv \\
        --mode both \\
        --lint-out cmod_ntm_lint_quality.json \\
        --profile-out cmod_ntm_profile_quality.json

The JSON output format mirrors the notebook format used previously:

    {
        "_line_name_comment": "...",
        "_line_name_map": {"0": "helike.w", ...},
        "results": {
            "1120906030": [0, 2, 7],
            ...
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np
import mdsthin as mds

from hirexsr_lint_profile_py import (
    LintProfileResult,
    _line_config,
    _line_display_name,
    hirexsr_get_lint_profile_py,
)
from hirexsr_get_profile_py import InversionProfileResult, hirexsr_get_profile_py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_line_name_map(tht: int = 0) -> dict[str, str]:
    """Return {str(line_idx): 'family.name', ...} for all 10 line indices."""
    m = {}
    for i in range(10):
        cfg = _line_config(i, tht)
        path = str(cfg["path"]).upper()
        family = "helike" if ".HELIKE." in path else "hlike"
        line_name = path.split(".MOMENTS.")[-1].rstrip(":").lower()
        m[str(i)] = f"{family}.{line_name}"
    return m


def _is_valid_value(arr: np.ndarray) -> np.ndarray:
    """Boolean mask: finite and not the -1 MDSplus sentinel."""
    return np.isfinite(arr) & (arr != -1.0)


def _apply_error_threshold(
    valid: np.ndarray,
    value: np.ndarray,
    err: np.ndarray,
    max_abs: float | None,
    max_rel: float | None,
) -> np.ndarray:
    """Narrow *valid* mask by absolute and/or relative error thresholds."""
    if max_abs is not None:
        valid = valid & (np.abs(err) <= max_abs)
    if max_rel is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(err) / np.maximum(np.abs(value), 1e-30)
        valid = valid & (rel <= max_rel)
    return valid

def openTree(shotno: int, treeName: str = "CMOD"):
    conn = mds.Connection("alcdata")
    conn.openTree(treeName, shotno)
    return conn

# -----------------------------------------------------------------------------
# Data existance check: line-integrted
# ---------------------------------------------------------------------------
def check_available_lines(
    shots: list[int],
    tht: int = 0,
    quiet: bool = False,
) -> dict[int, list[int]]:
    """Probe the MDSplus tree for every (shot, line) combination and report which have data.

    For each shot and each supported line index (0-9 as defined in `_line_config`),
    this function attempts to read the MOM node.  A "%TREE-W-NNF, Node Not Found"
    error (or any connection error) is treated as "no data".  Any node that returns
    at least one finite value is considered to have data.

    Parameters
    ----------
    shots : list of int
        Shot numbers to probe.
    tht : int, optional
        Analysis tree index (0 = ANALYSIS, >0 = ANALYSIS<tht>).
    quiet : bool, optional
        If False (default), print a summary table to stdout when done.

    Returns
    -------
    dict mapping shot -> list of line indices that have data
        Example: {1120906030: [0, 2, 7], 1120808020: [2]}
    """
    all_lines = list(range(10))
    results: dict[int, list[int]] = {}

    for shot in shots:
        available: list[int] = []
        for line in all_lines:
            try:
                cfg = _line_config(line, tht)
                path = cfg["path"]
                conn = openTree(shot, "spectroscopy")
                try:
                    raw = conn.get(f"{path}MOM")
                    arr = np.asarray(raw.data())
                    has_data = arr.size > 0 and np.any(np.isfinite(arr))
                except Exception:
                    has_data = False
                finally:
                    conn.closeAllTrees()
            except Exception:
                has_data = False

            if has_data:
                available.append(line)

        results[shot] = available

    if not quiet:
        print("=== check_available_lines results ===")
        print(f"{'Shot':>12}  {'Available lines (index: family.name)'}")
        print("-" * 60)
        for shot, lines in results.items():
            if lines:
                labels = [f"{l}:{_line_display_name(l, tht)}" for l in lines]
                print(f"{shot:>12}  {', '.join(labels)}")
            else:
                print(f"{shot:>12}  (none)")

    return results

# ---------------------------------------------------------------------------
# Data existance check: inverted profile
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Quality check: line-integrated (LintProfileResult)
# ---------------------------------------------------------------------------

def quality_check_lint(
    out: LintProfileResult,
    min_timepoints: int = 5,
    min_channels: int = 5,
    max_verr_abs: float | None = None,
    max_verr_rel: float | None = None,
    max_tierr_abs: float | None = None,
    max_tierr_rel: float | None = None,
) -> bool:
    """Return True if *out* passes all quality gates.

    Parameters
    ----------
    out : LintProfileResult
        Result from ``hirexsr_get_lint_profile_py``.
    min_timepoints : int
        Minimum number of time slices that must have enough good channels.
    min_channels : int
        Minimum number of spatial channels with good velocity data required
        per time slice for that time slice to count.
    max_verr_abs : float or None
        Maximum allowed absolute velocity error [km/s].
    max_verr_rel : float or None
        Maximum allowed relative velocity error |verr|/|v|.
    max_tierr_abs : float or None
        Maximum allowed absolute Ti error [keV].
    max_tierr_rel : float or None
        Maximum allowed relative Ti error |tierr|/|ti|.

    Notes
    -----
    ``LintProfileResult`` arrays are [nt_valid, nch] (time-major).
    """
    if out.tau.size == 0 or out.v.size == 0:
        return False

    v = np.asarray(out.v)       # [nt, nch]
    verr = np.asarray(out.verr)
    ti = np.asarray(out.ti)
    tierr = np.asarray(out.tierr)

    # Base mask: finite and not sentinel
    vel_valid = _is_valid_value(v)

    # Error-bar filters on velocity
    vel_valid = _apply_error_threshold(vel_valid, v, verr, max_verr_abs, max_verr_rel)

    # Optional Ti error filter reduces the set of "good" channels further
    if max_tierr_abs is not None or max_tierr_rel is not None:
        ti_valid = _is_valid_value(ti)
        ti_valid = _apply_error_threshold(ti_valid, ti, tierr, max_tierr_abs, max_tierr_rel)
        vel_valid = vel_valid & ti_valid

    # For each time slice count channels passing all filters
    channels_per_time = vel_valid.sum(axis=1)  # [nt]
    good_times = int(np.sum(channels_per_time >= min_channels))
    return good_times >= min_timepoints


# ---------------------------------------------------------------------------
# Quality check: inverted profile (InversionProfileResult)
# ---------------------------------------------------------------------------

def quality_check_profile(
    out: InversionProfileResult,
    min_timepoints: int = 5,
    min_channels: int = 5,
    max_verr_abs: float | None = None,
    max_verr_rel: float | None = None,
    max_tierr_abs: float | None = None,
    max_tierr_rel: float | None = None,
) -> bool:
    """Return True if *out* passes all quality gates.

    Parameters
    ----------
    out : InversionProfileResult
        Result from ``hirexsr_get_profile_py``.
    min_timepoints : int
        Minimum number of time slices with enough good spatial channels.
    min_channels : int
        Minimum number of spatial channels with good omega/rotation data per
        time slice.
    max_verr_abs : float or None
        Maximum allowed absolute rotation-frequency error [kHz] (omgerr).
    max_verr_rel : float or None
        Maximum allowed relative rotation-frequency error |omgerr|/|omg|.
    max_tierr_abs : float or None
        Maximum allowed absolute Ti error [keV].
    max_tierr_rel : float or None
        Maximum allowed relative Ti error.

    Notes
    -----
    ``InversionProfileResult`` arrays are [npsi, nt] (channel-major).
    The velocity-like quantity used here is ``omg`` (rotation frequency in
    kHz), consistent with the profile field ordering.
    """
    if out.time.size == 0 or out.omg.size == 0:
        return False

    omg = np.asarray(out.omg)       # [npsi, nt]
    omgerr = np.asarray(out.omgerr)
    ti = np.asarray(out.ti)
    tierr = np.asarray(out.tierr)

    # Base mask: finite and not sentinel
    vel_valid = _is_valid_value(omg)

    # Error-bar filters on omega
    vel_valid = _apply_error_threshold(vel_valid, omg, omgerr, max_verr_abs, max_verr_rel)

    # Optional Ti error filter
    if max_tierr_abs is not None or max_tierr_rel is not None:
        ti_valid = _is_valid_value(ti)
        ti_valid = _apply_error_threshold(ti_valid, ti, tierr, max_tierr_abs, max_tierr_rel)
        vel_valid = vel_valid & ti_valid

    # For each time slice (axis-1) count spatial channels passing all filters
    channels_per_time = vel_valid.sum(axis=0)  # [nt]
    good_times = int(np.sum(channels_per_time >= min_channels))
    return good_times >= min_timepoints


# ---------------------------------------------------------------------------
# Main scanning routines
# ---------------------------------------------------------------------------

def scan_lint_quality(
    shots: Sequence[int],
    tht: int = 0,
    quiet: bool = False,
    min_timepoints: int = 5,
    min_channels: int = 5,
    max_verr_abs: float | None = None,
    max_verr_rel: float | None = None,
    max_tierr_abs: float | None = None,
    max_tierr_rel: float | None = None,
) -> dict[int, list[int]]:
    """Scan shots for line-integrated data quality and return passing lines.

    Returns
    -------
    dict mapping int(shot) -> list of line indices that pass all quality gates.
    """
    results: dict[int, list[int]] = {}
    for shot in shots:
        passing: list[int] = []
        for line_idx in range(10):
            try:
                out = hirexsr_get_lint_profile_py(
                    shot=int(shot),
                    line=line_idx,
                    tht=tht,
                    use_idl_profile_moments=False,
                    debug_velocity_checks=False,
                )
                ok = quality_check_lint(
                    out,
                    min_timepoints=min_timepoints,
                    min_channels=min_channels,
                    max_verr_abs=max_verr_abs,
                    max_verr_rel=max_verr_rel,
                    max_tierr_abs=max_tierr_abs,
                    max_tierr_rel=max_tierr_rel,
                )
                if ok:
                    passing.append(line_idx)
            except Exception:
                pass
        results[int(shot)] = passing
        if not quiet:
            labels = [f"{l}:{_line_display_name(l, tht)}" for l in passing]
            status = ", ".join(labels) if labels else "(none)"
            print(f"  lint  {int(shot):>12}  {status}")
    return results


def scan_profile_quality(
    shots: Sequence[int],
    tht: int = 0,
    quiet: bool = False,
    min_timepoints: int = 5,
    min_channels: int = 5,
    max_verr_abs: float | None = None,
    max_verr_rel: float | None = None,
    max_tierr_abs: float | None = None,
    max_tierr_rel: float | None = None,
) -> dict[int, list[int]]:
    """Scan shots for inverted-profile data quality and return passing lines.

    Returns
    -------
    dict mapping int(shot) -> list of line indices that pass all quality gates.
    """
    results: dict[int, list[int]] = {}
    for shot in shots:
        passing: list[int] = []
        for line_idx in range(10):
            try:
                out = hirexsr_get_profile_py(
                    shot=int(shot),
                    line=line_idx,
                    quiet=True,
                    dc_shift=0.0,
                    tht=tht,
                    override=False,
                    seltime=None,
                )
                ok = quality_check_profile(
                    out,
                    min_timepoints=min_timepoints,
                    min_channels=min_channels,
                    max_verr_abs=max_verr_abs,
                    max_verr_rel=max_verr_rel,
                    max_tierr_abs=max_tierr_abs,
                    max_tierr_rel=max_tierr_rel,
                )
                if ok:
                    passing.append(line_idx)
            except Exception:
                pass
        results[int(shot)] = passing
        if not quiet:
            labels = [f"{l}:{_line_display_name(l, tht)}" for l in passing]
            status = ", ".join(labels) if labels else "(none)"
            print(f"profile {int(shot):>12}  {status}")
    return results


def _build_output_dict(results: dict[int, list[int]], tht: int = 0) -> dict:
    """Wrap results in the standard JSON output format."""
    line_name_map = _build_line_name_map(tht)
    return {
        "_line_name_comment": (
            "Line index to line name mapping from _line_config in hirexsr_lint_profile_py"
        ),
        "_line_name_map": line_name_map,
        "results": {str(shot): lines for shot, lines in results.items()},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Scan HIREX-SR shots for data quality and write JSON result files.\n\n"
            "Quality gates (applied per shot/line):\n"
            "  1. At least --min-timepoints time slices must have velocity data.\n"
            "  2. Each such time slice must have at least --min-channels good channels.\n"
            "  3. Optional error-bar thresholds further restrict 'good' points."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--csv",
        default="cmod_logbook_ntm_shots.csv",
        help="CSV file with shot numbers (one column, no header). Default: cmod_logbook_ntm_shots.csv",
    )
    p.add_argument(
        "--tht",
        type=int,
        default=0,
        help="Analysis tree suffix (0 = ANALYSIS, >0 = ANALYSIS<tht>). Default: 0",
    )
    p.add_argument(
        "--mode",
        choices=["lint", "profile", "both"],
        default="both",
        help="Which loader to scan. Default: both",
    )
    p.add_argument(
        "--lint-out",
        default="hirexsr_lint_quality.json",
        help="Output JSON path for line-integrated quality results.",
    )
    p.add_argument(
        "--profile-out",
        default="hirexsr_profile_quality.json",
        help="Output JSON path for inverted-profile quality results.",
    )

    # Quality gate parameters
    p.add_argument("--min-timepoints", type=int, default=5,
                   help="Minimum good time slices required. Default: 5")
    p.add_argument("--min-channels", type=int, default=5,
                   help="Minimum good spatial channels per time slice. Default: 5")
    p.add_argument("--max-verr-abs", type=float, default=10,
                   help="Max absolute velocity/omega error (km/s or kHz). Default: no limit")
    p.add_argument("--max-verr-rel", type=float, default=None,
                   help="Max relative velocity error |err|/|v|. Default: no limit")
    p.add_argument("--max-tierr-abs", type=float, default=1,
                   help="Max absolute Ti error [keV]. Default: no limit")
    p.add_argument("--max-tierr-rel", type=float, default=None,
                   help="Max relative Ti error |tierr|/|ti|. Default: no limit")

    p.add_argument("--quiet", action="store_true", help="Suppress per-shot progress output.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    shots = np.loadtxt(args.csv, dtype=int, delimiter=",")
    shots = shots.flatten().tolist()

    quality_kw = dict(
        tht=args.tht,
        quiet=args.quiet,
        min_timepoints=args.min_timepoints,
        min_channels=args.min_channels,
        max_verr_abs=args.max_verr_abs,
        max_verr_rel=args.max_verr_rel,
        max_tierr_abs=args.max_tierr_abs,
        max_tierr_rel=args.max_tierr_rel,
    )

    if args.mode in ("lint", "both"):
        print(f"Scanning {len(shots)} shots (line-integrated) ...")
        lint_results = scan_lint_quality(shots, **quality_kw)
        out_dict = _build_output_dict(lint_results, tht=args.tht)
        with open(args.lint_out, "w") as f:
            json.dump(out_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Saved lint quality results -> {args.lint_out}")

    if args.mode in ("profile", "both"):
        print(f"Scanning {len(shots)} shots (inverted profile) ...")
        prof_results = scan_profile_quality(shots, **quality_kw)
        out_dict = _build_output_dict(prof_results, tht=args.tht)
        with open(args.profile_out, "w") as f:
            json.dump(out_dict, f, indent=4, sort_keys=True, ensure_ascii=False)
        print(f"Saved profile quality results -> {args.profile_out}")


if __name__ == "__main__":
    main()
