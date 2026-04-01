"""Python rewrite of IDL `hirexsr_get_lint_profile` (from hirexsr_load_data.pro).

This module intentionally focuses on the line-integrated profile helper logic and
is kept separate from `hirexsr_load_result_py.py` for clarity.

Notes
-----
- The IDL routine builds on `hirexsr_load_momentptr` internals.
- Some C-Mod trees expose slightly different signal dimensions, so this rewrite
  uses robust shape handling and explicit fallbacks.
- A time-domain variant (`hirexsr_load_tlintptr`) is included as a minimal
  helper and can be extended once downstream usage is identified.

- Original IDL code and documentation:
    /usr/local/cmod/idl/HIREXSR/hirexsr_load_data.pro
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Any
import numpy as np
import mdsthin as mds

from hirexsr_plotting_py import _plot_lint_profile


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class LintProfileResult:
    """Output container for lint-profile quantities.

    Orientation conventions in this module:
    - `tau`: [nt_valid]
    - `v`, `verr`, `ti`, `tierr`, `rhotang`: [nt_valid, nch] (time-major)
    - `r_proj`: [nch, nt_valid] (channel-major; output of multi_interpol)
    - `r_ave`: [nch]
    """
    shot: int
    line: int
    tht: int
    tau: np.ndarray
    v: np.ndarray
    verr: np.ndarray
    emiss: np.ndarray
    emisserr: np.ndarray
    ti: np.ndarray
    tierr: np.ndarray
    rhotang: np.ndarray
    r_proj: np.ndarray
    r_ave: np.ndarray
    lam_o: float
    z: int


@dataclass
class TLintResult:
    """Time-domain organization analogous to IDL `hirexsr_load_tlintptr` output intent."""

    tau: np.ndarray
    v_t: np.ndarray
    verr_t: np.ndarray
    emiss_t: np.ndarray
    emisserr_t: np.ndarray
    ti_t: np.ndarray
    tierr_t: np.ndarray
    rhotang_t: np.ndarray


@dataclass
class MomentsData:
    """Container for loaded moments data from spectroscopy tree (mirrors IDL hirexsr_load_moments output)."""
    mom: np.ndarray
    err: np.ndarray
    pmom: np.ndarray
    perr: np.ndarray
    tau: np.ndarray
    rhotang: np.ndarray
    bfrac: np.ndarray
    scale: np.ndarray
    fitcase: np.ndarray
    pos: np.ndarray
    u: np.ndarray
    tpos: np.ndarray
    dlam: np.ndarray
    double: np.ndarray
    tree: str


# -----------------------------
# MDS helpers
# -----------------------------
def openTree(shotno: int, treeName: str = "CMOD"):
    conn = mds.Connection("alcdata")
    conn.openTree(treeName, shotno)
    return conn


def _analysis_tag(tht: int) -> str:
    return f"ANALYSIS{tht}" if tht > 0 else "ANALYSIS"


def _line_config(line: int, tht: int) -> dict[str, Any]:
    """Mirror IDL line routing used in hirexsr_load_momentptr/load_moments."""
    astr = _analysis_tag(tht)
    he = rf"\SPECTROSCOPY::TOP.HIREXSR.{astr}.HELIKE.MOMENTS"
    hl = rf"\SPECTROSCOPY::TOP.HIREXSR.{astr}.HLIKE.MOMENTS"

    # lam_o and Z are chosen to follow the IDL logic for each line index.
    # Common values are used here for practicality in Python-side analysis.
    #   Ar (Z=18): W/X/Z/LYA1/J
    #   Ca (Z=20): W mapped through HLIKE.LYA1 for some analyses
    #   Mo (Z=42): MO4D
    table = {
        0: {"path": f"{he}.W:", "z": 18, "lam_o": 3.94912},
        1: {"path": f"{he}.X:", "z": 18, "lam_o": 3.96581},
        2: {"path": f"{he}.Z:", "z": 18, "lam_o": 3.99417},
        3: {"path": f"{hl}.LYA1:", "z": 18, "lam_o": 3.73114},
        4: {"path": f"{hl}.MO4D:", "z": 42, "lam_o": 3.73980},
        5: {"path": f"{hl}.J:", "z": 18, "lam_o": 3.77179},
        # IDL routes line 6/9 via HLIKE.LYA1 but uses Ca W-like wavelength.
        6: {"path": f"{hl}.LYA1:", "z": 20, "lam_o": 3.17730},
        7: {"path": f"{he}.Z:", "z": 18, "lam_o": 3.73114},
        8: {"path": f"{he}.X:", "z": 42, "lam_o": 3.73980},
        9: {"path": f"{hl}.LYA1:", "z": 20, "lam_o": 3.73114},
    }
    if line not in table:
        raise ValueError(f"Unsupported line={line}. Supported line indices: {sorted(table)}")
    return table[line]


def _line_display_name(line: int, tht: int) -> str:
    """Return compact human-readable line label such as 'helike.z'."""
    cfg = _line_config(line, tht)
    path = str(cfg["path"])  # e.g. ...HELIKE.MOMENTS.Z:
    family = "helike" if ".HELIKE." in path.upper() else "hlike"
    if ".MOMENTS." in path.upper():
        line_name = path.split(".MOMENTS.")[-1].rstrip(":").lower()
    else:
        line_name = str(line)
    return f"{family}.{line_name}"


def _atomic_mass_amu(z: int) -> float:
    masses = {
        18: 39.948,  # Ar
        20: 40.078,  # Ca
        42: 95.95,   # Mo
    }
    if z not in masses:
        raise ValueError(f"No atomic-mass mapping configured for Z={z}")
    return masses[z]


# -----------------------------------------------
# Python rewrite of IDL hirexsr_load_moments
# -----------------------------------------------
def hirexsr_load_moments_py(shot: int, line: int, tht: int = 0) -> MomentsData:
    """Python rewrite of IDL `hirexsr_load_moments`.
    
    Loads moments, errors, profile moments, and auxiliary data from the 
    spectroscopy MDSplus tree for a given shot, line, and analysis tree.
    
    Mirrors the IDL function behavior exactly, including fallback logic for
    optional nodes (fitcase, tree name, dlam, double).
    
    Parameters
    ----------
    shot : int
        Shot number
    line : int
        Line index (0-9, routed via _line_config)
    tht : int, optional
        Analysis tree suffix (0 for ANALYSIS, >0 for ANALYSIS<tht>)
        
    Returns
    -------
    MomentsData
        Dataclass containing all loaded arrays and metadata
        
    Raises
    ------
    RuntimeError
        If moments data cannot be loaded from tree
    """
    cfg = _line_config(line, tht)
    path = cfg["path"]
    
    conn = openTree(shot, "spectroscopy")
    try:
        # Load primary data
        mom_raw = conn.get(f"{path}MOM")
        if mom_raw is None:
            raise RuntimeError(f"Cannot load {path}MOM for shot {shot}, line {line}")
        mom = np.asarray(mom_raw.data())
        
        # Load error array
        err_raw = conn.get(f"{path}ERR")
        err = np.asarray(err_raw.data()) if err_raw is not None else np.zeros_like(mom)
        
        # Load dimension vectors from various expressions
        # IDL: rhotang=mdsvalue('dim_of('+path+'MOM,0)')
        try:
            rhotang_raw = conn.get(f"dim_of({path}MOM,0)")
            rhotang = np.asarray(rhotang_raw.data()) if rhotang_raw is not None else np.array([])
        except Exception:
            rhotang = np.array([])
        
        # IDL: pmom=mdsvalue('dim_of('+path+'MOM,2)')
        try:
            pmom_raw = conn.get(f"dim_of({path}MOM,2)")
            pmom = np.asarray(pmom_raw.data()) if pmom_raw is not None else np.array([])
        except Exception:
            pmom = np.array([])
        
        # IDL: bfrac=mdsvalue('dim_of('+path+'MOM,3)')
        try:
            bfrac_raw = conn.get(f"dim_of({path}MOM,3)")
            bfrac = np.asarray(bfrac_raw.data()) if bfrac_raw is not None else np.array([])
        except Exception:
            bfrac = np.array([])
        
        # IDL: fitcase=mdsvalue('dim_of('+path+'MOM,4)',/quiet,status=fitstat)
        #      with fallback if not found
        try:
            fitcase_raw = conn.get(f"dim_of({path}MOM,4)")
            fitcase = np.asarray(fitcase_raw.data()) if fitcase_raw is not None else None
        except Exception:
            fitcase = None
        
        if fitcase is None:
            # Fallback: use bfrac shape and set to -1 where bfrac==0
            if bfrac.size > 0:
                fitcase = np.zeros_like(bfrac, dtype=float)
                fitcase[bfrac == 0] = -1
            else:
                fitcase = np.array([])
        
        # IDL: perr=mdsvalue('dim_of('+path+'ERR,2)')
        try:
            perr_raw = conn.get(f"dim_of({path}ERR,2)")
            perr = np.asarray(perr_raw.data()) if perr_raw is not None else np.array([])
        except Exception:
            perr = np.array([])
        
        # IDL: scale=mdsvalue('dim_of('+path+'ERR,3)')
        try:
            scale_raw = conn.get(f"dim_of({path}ERR,3)")
            scale = np.asarray(scale_raw.data()) if scale_raw is not None else np.array([])
        except Exception:
            scale = np.array([])
        
        # IDL: tau=mdsvalue('dim_of('+path+'MOM,1)',quiet=quiet)
        try:
            tau_raw = conn.get(f"dim_of({path}MOM,1)")
            tau = np.asarray(tau_raw.data()) if tau_raw is not None else np.array([])
        except Exception:
            tau = np.array([])
        
        # IDL: pos=mdsvalue(path+'POS',quiet=quiet)
        try:
            pos_raw = conn.get(f"{path}POS")
            pos = np.asarray(pos_raw.data()) if pos_raw is not None else np.array([])
        except Exception:
            pos = np.array([])
        
        # IDL: u=mdsvalue(path+'U',quiet=quiet)
        try:
            u_raw = conn.get(f"{path}U")
            u = np.asarray(u_raw.data()) if u_raw is not None else np.array([])
        except Exception:
            u = np.array([])
        
        # IDL: tpos=mdsvalue('dim_of('+path+'POS,0)',quiet=quiet)
        try:
            tpos_raw = conn.get(f"dim_of({path}POS,0)")
            tpos = np.asarray(tpos_raw.data()) if tpos_raw is not None else np.array([])
        except Exception:
            tpos = np.array([])
        
        # IDL: tree=mdsvalue('dim_of('+path+'POS,1)',quiet=quiet,status=treestatus)
        #      IF NOT treestatus THEN tree='analysis'
        try:
            tree_raw = conn.get(f"dim_of({path}POS,1)")
            if tree_raw is not None:
                tree_val = tree_raw.data()
                tree = str(tree_val) if tree_val is not None else 'analysis'
            else:
                tree = 'analysis'
        except Exception:
            tree = 'analysis'
        
        # IDL: dlam=mdsvalue(path+'DLAM',quiet=quiet)
        try:
            dlam_raw = conn.get(f"{path}DLAM")
            dlam = np.asarray(dlam_raw.data()) if dlam_raw is not None else np.array([])
        except Exception:
            dlam = np.array([])
        
        # IDL: double=mdsvalue(path+'DOUBLE',quiet=quiet)
        try:
            double_raw = conn.get(f"{path}DOUBLE")
            double = np.asarray(double_raw.data()) if double_raw is not None else np.array([])
        except Exception:
            double = np.array([])
        
    finally:
        conn.closeAllTrees()
    
    return MomentsData(
        mom=mom,
        err=err,
        pmom=pmom,
        perr=perr,
        tau=tau,
        rhotang=rhotang,
        bfrac=bfrac,
        scale=scale,
        fitcase=fitcase,
        pos=pos,
        u=u,
        tpos=tpos,
        dlam=dlam,
        double=double,
        tree=tree,
    )


# -----------------------------
# Numeric helpers
# -----------------------------
def multi_interpol(rmid: np.ndarray, rpsi: np.ndarray, efit_time: np.ndarray, psinorm: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Python rewrite of IDL `multi_interpol`.

    Projects EFIT `rmid(rpsi, t)` values onto `(psinorm, times)` samples.
    """
    psinorm = np.asarray(psinorm)
    times = np.asarray(times)
    efit_time = np.asarray(efit_time)
    rpsi = np.asarray(rpsi)
    rmid = np.asarray(rmid)

    if psinorm.ndim != 2:
        raise ValueError(f"psinorm must be 2D [nch,nt], got {psinorm.shape}")

    nch, nt = psinorm.shape
    out = np.full((nch, nt), np.nan, dtype=float)

    # Ensure rmid is [nt_efit, npsi] for this interpolation routine.
    if rmid.ndim != 2:
        raise ValueError(f"rmid must be 2D, got {rmid.shape}")
    if rmid.shape[0] != efit_time.size and rmid.shape[1] == efit_time.size:
        rmid = rmid.T

    if rmid.shape[0] != efit_time.size:
        raise ValueError(
            f"Cannot align rmid with efit_time: rmid shape={rmid.shape}, len(efit_time)={efit_time.size}"
        )

    for j in range(nt):
        t = times[j]

        # Find lower/upper EFIT indices around sample time.
        hi = int(np.searchsorted(efit_time, t, side="left"))
        if hi <= 0 or hi >= efit_time.size:
            continue
        lo = hi - 1

        dt = efit_time[hi] - efit_time[lo]
        if dt == 0:
            frac = 0.0
        else:
            frac = (t - efit_time[lo]) / dt

        # Linear-in-time interpolation of Rmid(psi).
        rmid_t = rmid[lo, :] + frac * (rmid[hi, :] - rmid[lo, :])

        # Interpolate onto requested psi samples for this time slice.
        out[:, j] = np.interp(psinorm[:, j], rpsi, rmid_t, left=np.nan, right=np.nan)

    return out


def _normalize_mom_shape(arr: np.ndarray) -> np.ndarray:
    """Normalize raw moments/errors to mdsthin ordering [3, nt, nch].

    IDL treats these arrays as [nch, nt, 3], while mdsthin typically reads them
    as [3, nt, nch].  We keep the mdsthin-native convention throughout this
    module to avoid repeated axis flips and ambiguity.
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D moment array, got {arr.shape}")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[-1] == 3:
        # [nt, nch, 3] -> [3, nt, nch]
        return np.moveaxis(arr, -1, 0)
    if arr.shape[1] == 3:
        # [nt, 3, nch] -> [3, nt, nch]
        return np.moveaxis(arr, 1, 0)
    raise ValueError(f"Cannot normalize to [3, nt, nch] from shape {arr.shape}")


def _validate_mom_time_axis(mom_arr: np.ndarray, nt: int) -> np.ndarray:
    """Validate/align moments array to [3, nt, nch] given tau length `nt`."""
    arr = np.asarray(mom_arr)
    if arr.ndim != 3 or arr.shape[0] != 3:
        raise ValueError(f"Expected shape (3, A, B), got {arr.shape}")
    if arr.shape[1] == nt:
        return arr  # already [3, nt, nch]
    if arr.shape[2] == nt:
        # [3, nch, nt] -> [3, nt, nch]
        return np.swapaxes(arr, 1, 2)
    raise ValueError(
        f"Cannot match nt={nt} to shape {arr.shape}; "
        "neither axis 1 nor axis 2 matches the tau dimension."
    )


def _extract_tau(moments_data: MomentsData) -> np.ndarray:
    """Extract the shot time vector from moments data.

    IDL hirexsr_load_moments sets::

        tau = mdsvalue('dim_of(' + path + 'MOM,1)')   ; time axis in seconds

    This is the axis returned by moments_data.tau (dim_of(MOM,1)).
    Invalid / unfilled time slots are marked as -1 in this vector.

    Note: moments_data.u is the **instrument étendue** [m²·sr] loaded from
    the :U node.  Despite its size matching nch in some trees, it is NOT a
    time axis and must not be used as tau.  The IDL hirexsr_get_lint_profile
    function never uses :U for time.
    """
    tau = moments_data.tau
    if tau.ndim > 1:
        tau = np.squeeze(tau)
    if tau.size > 0:
        return tau
    # Fallback: return sample indices along axis 1 of the raw mom array.
    # (axis 1 = time in Python's (3, nt, nch) layout from mdsthin)
    if moments_data.mom.ndim >= 3:
        return np.arange(moments_data.mom.shape[1], dtype=float)
    raise RuntimeError("Cannot determine time axis from moments data")


def _extract_rhotang(moments_data: MomentsData, nch: int, nt: int) -> np.ndarray:
    """Extract rhotang, normalised to shape [nch, nt].

    IDL hirexsr_load_moments sets::

        rhotang = mdsvalue('dim_of(' + path + 'MOM,0)')   ; [nch, nt] in IDL

    Python mdsthin reads this column-major array as (nt, nch).  We keep this as
    [nt, nch] to match the [3, nt, nch] moments convention used in this module.
    """
    rhotang = moments_data.rhotang
    if rhotang.size == 0:
        return np.full((nt, nch), np.nan, dtype=float)
    if rhotang.ndim == 2:
        if rhotang.shape == (nt, nch):
            return rhotang
        if rhotang.shape == (nch, nt):
            return rhotang.T           # transpose (nch, nt) -> (nt, nch)
    if rhotang.ndim == 1 and rhotang.size == nch:
        return np.repeat(rhotang[None, :], nt, axis=0)
    return np.full((nt, nch), np.nan, dtype=float)


def _mask_pos_sentinels(arr: np.ndarray, sentinel: float = -1.0) -> np.ndarray:
    """Replace sentinel values in POS array with NaN.
    
    MDSplus trees mark invalid/unfilled data with sentinel values (typically -1.0).
    This ensures that geometric factor calculations skip invalid entries.
    """
    result = np.asarray(arr, dtype=float)
    result[result == sentinel] = np.nan
    return result


def _compute_geom_factor(pos: np.ndarray, nt: int, nch: int) -> np.ndarray:
    """Compute geometric factor 1/(2*pi*R*cos(theta)) as [nt, nch].

    Supports common POS layouts seen in tree reads:
    - [nch, 4]
    - [4, nch]
    - [4, nt, nch]
    - [4, nch, nt]
    - [nt, nch, 4]
    - [nch, nt, 4]

    Automatically masks sentinel values (-1.0) in R and theta columns to NaN,
    as these mark invalid/unfilled MDSplus data that would otherwise corrupt
    the geometric correction.

    If no known layout matches, returns ones (no geometric correction).
    """
    geom = np.ones((nt, nch), dtype=float)
    p = _mask_pos_sentinels(np.asarray(pos))

    if p.ndim == 2:
        if p.shape == (nch, 4):
            denom = 2.0 * np.pi * p[:, 2] * np.cos(p[:, 3])
            geom_ch = np.where(denom != 0, 1.0 / denom, np.nan)
            return np.repeat(geom_ch[None, :], nt, axis=0)
        if p.shape == (4, nch):
            denom = 2.0 * np.pi * p[2, :] * np.cos(p[3, :])
            geom_ch = np.where(denom != 0, 1.0 / denom, np.nan)
            return np.repeat(geom_ch[None, :], nt, axis=0)
        return geom

    if p.ndim == 3:
        # (4, nt, nch)
        if p.shape[0] == 4 and p.shape[1] == nt and p.shape[2] == nch:
            denom = 2.0 * np.pi * p[2, :, :] * np.cos(p[3, :, :])
            return np.where(denom != 0, 1.0 / denom, np.nan)
        # (4, nch, nt)
        if p.shape[0] == 4 and p.shape[1] == nch and p.shape[2] == nt:
            q = np.swapaxes(p, 1, 2)
            denom = 2.0 * np.pi * q[2, :, :] * np.cos(q[3, :, :])
            return np.where(denom != 0, 1.0 / denom, np.nan)
        # (nt, nch, 4)
        if p.shape[0] == nt and p.shape[1] == nch and p.shape[2] == 4:
            denom = 2.0 * np.pi * p[:, :, 2] * np.cos(p[:, :, 3])
            return np.where(denom != 0, 1.0 / denom, np.nan)
        # (nch, nt, 4)
        if p.shape[0] == nch and p.shape[1] == nt and p.shape[2] == 4:
            q = np.swapaxes(p, 0, 1)
            denom = 2.0 * np.pi * q[:, :, 2] * np.cos(q[:, :, 3])
            return np.where(denom != 0, 1.0 / denom, np.nan)

    return geom


def _run_velocity_checks(
    lam_o: float,
    mom1_valid: np.ndarray,
    v_valid: np.ndarray,
    v_nogeom_valid: np.ndarray,
    geom_valid: np.ndarray,
    rhotang_valid: np.ndarray,
    pos: np.ndarray,
) -> None:
    """Print focused diagnostics to investigate velocity offset/trend issues."""
    print("=== velocity diagnostics ===")
    dlam = lam_o - mom1_valid
    print(
        f"delta-lambda stats [Ang]: min={np.nanmin(dlam):.3e}, "
        f"max={np.nanmax(dlam):.3e}, median={np.nanmedian(dlam):.3e}"
    )
    print(
        f"v_no_geom stats [km/s]: min={np.nanmin(v_nogeom_valid):.3e}, "
        f"max={np.nanmax(v_nogeom_valid):.3e}, median={np.nanmedian(v_nogeom_valid):.3e}"
    )
    print(
        f"v_with_geom stats [km/s]: min={np.nanmin(v_valid):.3e}, "
        f"max={np.nanmax(v_valid):.3e}, median={np.nanmedian(v_valid):.3e}"
    )
    print(
        f"geom stats: min={np.nanmin(geom_valid):.3e}, "
        f"max={np.nanmax(geom_valid):.3e}, median={np.nanmedian(geom_valid):.3e}, "
        f"neg_frac={np.mean(geom_valid < 0):.3f}"
    )

    # Per-time slope of v vs rhotang. Positive slope means v increases with rhotang.
    slopes_geom = []
    slopes_nogeom = []
    for it in range(v_valid.shape[0]):
        x = rhotang_valid[it, :]
        y_geom = v_valid[it, :]
        y_nogeom = v_nogeom_valid[it, :]
        ok_geom = np.isfinite(x) & np.isfinite(y_geom)
        ok_nogeom = np.isfinite(x) & np.isfinite(y_nogeom)
        if np.sum(ok_geom) >= 4:
            slopes_geom.append(np.polyfit(x[ok_geom], y_geom[ok_geom], 1)[0])
        if np.sum(ok_nogeom) >= 4:
            slopes_nogeom.append(np.polyfit(x[ok_nogeom], y_nogeom[ok_nogeom], 1)[0])

    if slopes_geom:
        sg = np.asarray(slopes_geom)
        print(
            f"slope(v_with_geom vs rhotang): median={np.nanmedian(sg):.3e}, "
            f"pos={np.sum(sg > 0)}/{sg.size}, neg={np.sum(sg < 0)}/{sg.size}"
        )
    if slopes_nogeom:
        sn = np.asarray(slopes_nogeom)
        print(
            f"slope(v_no_geom vs rhotang): median={np.nanmedian(sn):.3e}, "
            f"pos={np.sum(sn > 0)}/{sn.size}, neg={np.sum(sn < 0)}/{sn.size}"
        )

    # POS sanity checks for [nch,4] style trees.
    p = np.asarray(pos)
    if p.ndim == 2 and p.shape[1] >= 4:
        rcol = p[:, 2]
        tcol = p[:, 3]
        
        # Count sentinel values that will be masked during geom computation
        r_sentinel_count = int(np.sum(rcol == -1.0))
        t_sentinel_count = int(np.sum(tcol == -1.0))
        
        print(
            f"POS[:,2] (R): sentinel_count={r_sentinel_count}, "
            f"range={[np.nanmin(rcol), np.nanmax(rcol)]}"
        )
        print(
            f"POS[:,3] (theta): sentinel_count={t_sentinel_count}, "
            f"range=[{np.nanmin(tcol):.3e}, {np.nanmax(tcol):.3e}]"
        )
        
        # Interpret cos(theta) assuming valid (non-sentinel) values
        rcol_valid = rcol[rcol != -1.0]
        tcol_valid = tcol[tcol != -1.0]
        
        if tcol_valid.size > 0:
            cos_rad = np.cos(tcol_valid)
            cos_deg = np.cos(np.deg2rad(tcol_valid))
            print(
                f"cos(theta) range [valid only] radians: [{np.nanmin(cos_rad):.3e}, {np.nanmax(cos_rad):.3e}]"
            )
            print(
                f"cos(theta) range [valid only] degrees: [{np.nanmin(cos_deg):.3e}, {np.nanmax(cos_deg):.3e}]"
            )


# -----------------------------
# Main rewrite
# -----------------------------
def hirexsr_get_lint_profile_py(
    shot: int,
    line: int = 2,
    tht: int = 0,
    use_idl_profile_moments: bool = False,
    debug_velocity_checks: bool = False,
) -> LintProfileResult:
    """Python rewrite of IDL `hirexsr_get_lint_profile`.

    Returns line-integrated profile quantities on (nt_valid, nch) grids, where
    nch is the number of spatial detector channels and nt_valid is the number of
    valid time points (tau != -1) returned by hirexsr_load_moments.

    Array layouts:
    - `v`, `verr`, `ti`, `tierr`, `rhotang`: [nt_valid, nch] (time-major)
    - `r_proj` from `multi_interpol`: [nch, nt_valid] (channel-major)
    - `r_ave`: channel-average over time, shape [nch]

    Parameters
    ----------
    use_idl_profile_moments : bool
        If True, use profile moments from pmom/perr (IDL-equivalent to
        momentptr columns 11/12/14/15). If False, use raw mom/err moments
        [1]/[2] as before.
    debug_velocity_checks : bool
        If True, print detailed diagnostics to investigate velocity-sign,
        offset, and trend behavior before/after geometric correction.
    """
    c = 2.99e8
    e = 1.602e-19
    mconv = 1.661e-27

    cfg = _line_config(line, tht)
    lam_o = float(cfg["lam_o"])
    # lam_o = 3.73114 # Hardcoded test
    z = int(cfg["z"])

    # Load EFIT data from analysis tree.
    conn_a = openTree(shot, "analysis")
    try:
        rmid = np.asarray(conn_a.get(r"\ANALYSIS::EFIT_RMID").data())
        efit_time = np.asarray(conn_a.get(r"dim_of(\ANALYSIS::EFIT_RMID,0)").data())
        rpsi = np.asarray(conn_a.get(r"dim_of(\ANALYSIS::EFIT_RMID,1)").data())
    finally:
        conn_a.closeAllTrees()

    # Load all moments data via Python rewrite of IDL hirexsr_load_moments.
    moments = hirexsr_load_moments_py(shot, line, tht)

    # Step 1: extract tau from dim_of(MOM,1) — the time axis in seconds.
    # This is IDL's tau (invalid slots are -1).  :U is étendue [m²·sr], not time.
    tau = _extract_tau(moments)   # shape (nt,);  nt = total time slots
    nt = tau.size

    # Step 2: normalize/validate moments as [3, nt, nch].
    mom = _validate_mom_time_axis(_normalize_mom_shape(moments.mom), nt)
    err = _validate_mom_time_axis(_normalize_mom_shape(moments.err), nt)
    nch = mom.shape[2]   # number of spatial detector channels (e.g. 32)
    pos = moments.pos

    # Step 3: rhotang [nt, nch].
    rhotang = _extract_rhotang(moments, nch, nt)

    valid = np.where(tau != -1)[0]
    if valid.size == 0:
        raise RuntimeError("No valid times found (tau == -1 for all samples)")

    subtime = tau[valid]

    # Choose moment source.
    # - Default: raw mom/err indices [1],[2]
    # - Optional IDL-equivalent: profile moments pmom/perr (momentptr 11/12/14/15)
    source = "mom/err"
    if use_idl_profile_moments:
        try:
            pmom = _validate_mom_time_axis(_normalize_mom_shape(moments.pmom), nt)
            perr = _validate_mom_time_axis(_normalize_mom_shape(moments.perr), nt)
            mom0 = pmom[0, :, :]
            mom1 = pmom[1, :, :]
            mom2 = pmom[2, :, :]
            err0 = perr[0, :, :]
            err1 = perr[1, :, :]
            err2 = perr[2, :, :]
            source = "pmom/perr (IDL 11/12/14/15 equivalent)"
        except Exception:
            mom0 = mom[0, :, :]
            mom1 = mom[1, :, :]
            mom2 = mom[2, :, :]
            err0 = err[0, :, :]
            err1 = err[1, :, :]
            err2 = err[2, :, :]
            source = "mom/err (fallback; pmom/perr unavailable)"
    else:
        mom0 = mom[0, :, :]
        mom1 = mom[1, :, :]
        mom2 = mom[2, :, :]
        err0 = err[0, :, :]
        err1 = err[1, :, :]
        err2 = err[2, :, :]

    # Geometric denominator 1 / (2π R cos θ) from POS.
    # This now supports POS layouts including [nch,4], matching the shot case
    # where IDL-style jpos[:,2] / jpos[:,3] should be applied.
    geom = _compute_geom_factor(pos, nt=nt, nch=nch)

    mass = _atomic_mass_amu(z)
    conv_factor = (lam_o / c) ** 2 * (e * 1.0e3 / (mass * mconv))

    if use_idl_profile_moments: v_nogeom = -1.0 * (lam_o - mom1) * c / lam_o * 1.0e-3
    else:  v_nogeom =  (mom1/mom0) * c / lam_o * 1.0e-3
    v = v_nogeom * geom
    verr = np.abs(err1) * c / lam_o * np.abs(geom) * 1.0e-3
    emiss = mom0
    emisserr = np.abs(err0)
    if use_idl_profile_moments: ti = (mom2 ** 2) / conv_factor
    else:  ti = (mom2 / mom0) / conv_factor
    tierr = 2.0 * np.abs(err2) * np.sqrt(mom2 ** 2) / conv_factor

    # Geometry diagnostics: confirm geometric correction is actually active.
    finite_geom_mask = np.isfinite(geom)
    finite_geom_count = int(np.sum(finite_geom_mask))
    total_geom_count = int(geom.size)
    nonunity_geom_count = int(np.sum(finite_geom_mask & (np.abs(geom - 1.0) > 1.0e-12)))
    if finite_geom_count > 0:
        geom_min = float(np.nanmin(geom))
        geom_max = float(np.nanmax(geom))
    else:
        geom_min = np.nan
        geom_max = np.nan

    # Diagnostic output the data sizes before filtering to valid times.
    print(f"Loaded moments data: mom shape={mom.shape}, err shape={err.shape}, pos shape={pos.shape}")
    print(f"Extracted tau shape={tau.shape}, rhotang shape={rhotang.shape}")
    print(
        f"Computed emiss shape={emiss.shape}, emisserr shape={emisserr.shape}, "
        f"v shape={v.shape}, verr shape={verr.shape}, ti shape={ti.shape}, tierr shape={tierr.shape}"
    )
    print(
        f"Geom stats: finite={finite_geom_count}/{total_geom_count}, "
        f"non-unity={nonunity_geom_count}, range=[{geom_min:.3e}, {geom_max:.3e}]"
    )
    print(f"Moment source: {source}")
    print(f"Number of valid time points (tau != -1): {valid.size} out of {nt} total")
    # Keep only valid times (IDL behavior).
    v_nogeom = v_nogeom[valid, :]
    geom = geom[valid, :]
    v = v[valid, :]
    verr = verr[valid, :]
    emiss = emiss[valid, :]
    emisserr = emisserr[valid, :]
    ti = ti[valid, :]
    tierr = tierr[valid, :]
    rhotang = rhotang[valid, :]

    if debug_velocity_checks:
        _run_velocity_checks(
            lam_o=lam_o,
            mom1_valid=mom1[valid, :],
            v_valid=v,
            v_nogeom_valid=v_nogeom,
            geom_valid=geom,
            rhotang_valid=rhotang,
            pos=pos,
        )

    # multi_interpol expects psinorm as [nch, nt], so transpose from [nt, nch].
    # Result r_proj is [nch, nt_valid] (channel-major by construction).
    r_proj = multi_interpol(rmid, rpsi, efit_time, rhotang.T, subtime)
    valid_counts = np.sum(np.isfinite(r_proj), axis=1)
    sum_vals = np.nansum(r_proj, axis=1)
    # Time-average projected major radius for each channel -> [nch].
    r_ave = np.full(r_proj.shape[0], np.nan, dtype=float)
    has_vals = valid_counts > 0
    r_ave[has_vals] = sum_vals[has_vals] / valid_counts[has_vals]

    return LintProfileResult(
        shot=shot,
        line=line,
        tht=tht,
        tau=subtime,
        v=v,
        verr=verr,
        emiss=emiss,
        emisserr=emisserr,
        ti=ti,
        tierr=tierr,
        rhotang=rhotang,
        r_proj=r_proj,
        r_ave=r_ave,
        lam_o=lam_o,
        z=z,
    )


def hirexsr_load_tlintptr_py(shot: int, line: int = 2, tht: int = 0) -> TLintResult:
    """Simple time-domain reorganizer inspired by IDL `hirexsr_load_tlintptr`.

    This now mirrors the native output ordering, where arrays are already
    [nt, nch].
    """
    out = hirexsr_get_lint_profile_py(shot=shot, line=line, tht=tht)
    return TLintResult(
        tau=out.tau,
        v_t=out.v,
        verr_t=out.verr,
        emiss_t=out.emiss,
        emisserr_t=out.emisserr,
        ti_t=out.ti,
        tierr_t=out.tierr,
        rhotang_t=out.rhotang,
    )





def _print_summary(out: LintProfileResult) -> None:
    print("=== hirexsr_get_lint_profile_py summary ===")
    print(f"shot={out.shot}, line={out.line}, tht={out.tht}, Z={out.z}, lam_o={out.lam_o}")
    print(f"tau shape: {out.tau.shape}  [nt_valid]")
    print(f"emiss shape: {out.emiss.shape}  [nt_valid, nch]")
    print(f"v shape: {out.v.shape}  [nt_valid, nch]")
    print(f"ti shape: {out.ti.shape}  [nt_valid, nch]")
    print(f"rhotang shape: {out.rhotang.shape}  [nt_valid, nch]")
    print(f"r_proj shape: {out.r_proj.shape}  [nch, nt_valid]")
    print(f"r_ave shape: {out.r_ave.shape}  [nch]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python rewrite of hirexsr_get_lint_profile")
    parser.add_argument("--shot", type=int, default=1120906030)#1120808020)
    parser.add_argument("--line", type=int, default=7)
    parser.add_argument("--tht", type=int, default=0)
    parser.add_argument(
        "--use-idl-profile-moments",
        action="store_true",
        help="Use pmom/perr (IDL momentptr 11/12/14/15 equivalent) for v/Ti",
        default=False
    )
    parser.add_argument(
        "--debug-velocity-checks",
        action="store_true",
        help="Print detailed diagnostics for velocity/geometric-factor behavior",
    )
    parser.add_argument("--plot", dest="plot", action="store_true", default=True, help="Show plots")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plots")
    parser.add_argument("--every-nth", type=int, default=2, help="Plot every nth time point")
    parser.add_argument(
        "--x-axis",
        choices=["r_proj", "rhotang"],
        default="rhotang",
        help="Radial coordinate for plotting (default: rhotang)",
    )
    args = parser.parse_args()

    result = hirexsr_get_lint_profile_py(
        shot=args.shot,
        line=args.line,
        tht=args.tht,
        use_idl_profile_moments=args.use_idl_profile_moments,
        debug_velocity_checks=args.debug_velocity_checks,
    )
    _print_summary(result)
    if args.plot:
        _plot_lint_profile(result, every_nth=args.every_nth, x_axis=args.x_axis)

    print("Done.")
