"""Python rewrite of IDL `hirexsr_load_result.pro`.

This module loads three products for a HIREXSR line:
1) radial profiles
2) moments
3) line-integrated profiles

The implementation follows the original IDL logic where possible, but reads
MDSplus data directly in Python via a live MDS connection.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Any

import numpy as np

# Assumes new mds is installed (via pip install mdsthin)
import mdsthin as mds


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class ProfileResult:
    prof: dict[str, np.ndarray]
    proferr: dict[str, np.ndarray]
    tau: np.ndarray
    psinorm: np.ndarray
    rnorm: np.ndarray
    Rmid: np.ndarray
    Rmagx: np.ndarray
    a: np.ndarray
    prof1: dict[str, np.ndarray] | None = None
    proferr1: dict[str, np.ndarray] | None = None


@dataclass
class MomentResult:
    mom: np.ndarray
    momerr: np.ndarray
    tau: np.ndarray


@dataclass
class LineIntResult:
    lint: dict[str, np.ndarray]
    linterr: dict[str, np.ndarray]
    tau: np.ndarray
    psinorm: np.ndarray
    rnorm: np.ndarray
    Rmid: np.ndarray
    Rmagx: np.ndarray
    a: np.ndarray


# -----------------------------
# Connection and naming helpers
# -----------------------------
def openTree(shotno: int, treeName: str = "CMOD"):
    """Open an MDSplus tree using the same connection pattern as get_Cmod_Data.py."""
    conn = mds.Connection("alcdata")
    conn.openTree(treeName, shotno)
    return conn


def _line_to_label(line: int | str) -> str:
    """Map IDL-style line inputs to spectroscopy line labels."""
    if isinstance(line, str):
        return line.strip().lower()

    mapping = {
        1: "w",
        2: "z",
        3: "lya1",
        # Extended mapping often used in HIREX tooling.
        4: "x",
        5: "mo4d",
        6: "j",
    }
    if line not in mapping:
        raise ValueError(f"Unsupported line={line}. Supported integer lines: {sorted(mapping)}")
    return mapping[line]


def _candidate_line_labels(line: int | str) -> list[str]:
    """Return line candidates to try when a requested line has no data."""
    if isinstance(line, str):
        label = line.strip().lower()
        return [label]

    primary = _line_to_label(line)
    # Keep requested mapping first, then try common alternatives.
    common = ["z", "w", "x", "lya1", "j", "mo4d"]
    out = [primary] + [ll for ll in common if ll != primary]
    return out


def _line_to_branch(line_label: str) -> str:
    if line_label in {"w", "x", "z"}:
        return "helike"
    return "hlike"


def _analysis_root(tht: int) -> str:
    root = r"\spectroscopy::top.hirexsr.analysis"
    if tht > 0:
        root += str(tht)
    return root


def _try_get_data(conn, exprs: list[str]) -> tuple[np.ndarray, str]:
    """Return first successful `conn.get(expr).data()` among candidates."""
    last_exc: Exception | None = None
    for expr in exprs:
        try:
            data = conn.get(expr).data()
            return np.asarray(data), expr
        except Exception as exc:  # pragma: no cover - depends on remote tree layout
            last_exc = exc
    raise RuntimeError(f"All node expressions failed: {exprs}; last error={last_exc}")


def _try_get_dim(conn, expr: str, dim_idx: int = 0) -> np.ndarray:
    """Get time/radial axis via dim_of(TDI), returning a numpy array."""
    return np.asarray(conn.get(f"dim_of({expr},{dim_idx})").data())


# -----------------------------
# Numeric helpers
# -----------------------------
def _ensure_psi_tau(arr: np.ndarray, npsi: int, ntau: int) -> np.ndarray:
    """Force array orientation to [psi, tau] when possible."""
    arr = np.asarray(arr)
    if arr.shape == (npsi, ntau):
        return arr
    if arr.shape == (ntau, npsi):
        return arr.T
    raise ValueError(f"Array shape {arr.shape} does not match expected (npsi,ntau)=({npsi},{ntau})")


def _ensure_profile_cube(arr: np.ndarray, nfields: int = 4, nt: int | None = None) -> np.ndarray:
    """Normalize profile cube shape to [psi, tau, field].

    Accepts [field, psi, tau], [psi, tau, field], or [tau, psi, field].
    When *nt* is provided the time-axis size is used to disambiguate
    [tau, psi, field] from the correct [psi, tau, field] layout, which is
    necessary when raw MDSplus reads return [nt, npsi, nfield].
    """
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D profile array, got shape {arr.shape}")

    if arr.shape[0] == nfields:
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[-1] != nfields:
        raise ValueError(f"Cannot infer profile axis ordering from shape {arr.shape}")

    # arr is now [?, ?, field].  When nt is known, swap axes 0 and 1 if time
    # is currently on axis 0 (i.e. MDSplus returned [nt, npsi, nfield]).
    if nt is not None and arr.shape[0] == nt and arr.shape[1] != nt:
        arr = np.swapaxes(arr, 0, 1)

    return arr


def _interp2d_grid(z_xy: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """Bilinear-style interpolation on a rectangular grid.

    Mirrors the IDL `/grid` behavior by returning shape [len(x1), len(y1)].
    Steps:
    1) interpolate along y for each original x
    2) interpolate along x for each new y
    """
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    z_xy = np.asarray(z_xy)

    if z_xy.shape != (x0.size, y0.size):
        raise ValueError(
            f"z_xy shape {z_xy.shape} must be (len(x0),len(y0))={(x0.size,y0.size)}"
        )

    zy = np.empty((x0.size, y1.size), dtype=float)
    for i in range(x0.size):
        zy[i, :] = np.interp(y1, y0, z_xy[i, :], left=np.nan, right=np.nan)

    out = np.empty((x1.size, y1.size), dtype=float)
    for j in range(y1.size):
        out[:, j] = np.interp(x1, x0, zy[:, j], left=np.nan, right=np.nan)

    return out


def _subset_good_times(tau: np.ndarray, tgood: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered tau and matching index array."""
    if tgood is None:
        keep = np.arange(tau.size)
        return tau, keep

    tgood = np.asarray(tgood).squeeze()
    if tgood.dtype == bool:
        keep = np.where(tgood)[0]
    else:
        keep = np.where(tgood == 1)[0]

    return tau[keep], keep


# -----------------------------
# Core loaders
# -----------------------------
def _load_efit_mapping(shot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load EFIT arrays used to map psi->r/a and psi->Rmid in time."""
    conn = openTree(shot, "analysis")
    try:
        efit_tau = np.asarray(conn.get(r"dim_of(\efit_aeqdsk:rmagx)").data())
        efit_rmidd_raw = np.asarray(conn.get(r"\ANALYSIS::EFIT_RMID").data())

        # Ensure efit_rmidd is shaped [psi, tau] before interpolation.
        if efit_rmidd_raw.ndim != 2:
            raise ValueError(f"Unexpected EFIT_RMID shape {efit_rmidd_raw.shape}; expected 2D")
        if efit_rmidd_raw.shape[1] == efit_tau.size:
            efit_rmidd = efit_rmidd_raw
        elif efit_rmidd_raw.shape[0] == efit_tau.size:
            efit_rmidd = efit_rmidd_raw.T
        else:
            raise ValueError(
                "Cannot align EFIT_RMID with EFIT time axis: "
                f"EFIT_RMID shape={efit_rmidd_raw.shape}, len(efit_tau)={efit_tau.size}"
            )

        efit_psi = np.asarray(conn.get(r"dim_of(\ANALYSIS::EFIT_RMID,1)").data())
        if efit_psi.size != efit_rmidd.shape[0]:
            # Some trees expose psi on dim 0 instead of dim 1.
            efit_psi_alt = np.asarray(conn.get(r"dim_of(\ANALYSIS::EFIT_RMID,0)").data())
            if efit_psi_alt.size == efit_rmidd.shape[0]:
                efit_psi = efit_psi_alt
            else:
                raise ValueError(
                    "Cannot align EFIT psi axis with EFIT_RMID: "
                    f"len(dim1)={efit_psi.size}, len(dim0)={efit_psi_alt.size}, "
                    f"npsi={efit_rmidd.shape[0]}"
                )

        efit_rmagx = np.asarray(conn.get(r"\efit_aeqdsk:rmagx").data()) / 100.0
        efit_aout = np.asarray(conn.get(r"\efit_aeqdsk:aout").data()) / 100.0
        efit_rout = np.asarray(conn.get(r"\efit_aeqdsk:rout").data()) / 100.0
        efit_a = efit_rout + efit_aout - efit_rmagx

        efit_roa = np.zeros_like(efit_rmidd, dtype=float)
        for it in range(efit_tau.size):
            efit_roa[:, it] = (efit_rmidd[:, it] - efit_rmagx[it]) / efit_a[it]
    finally:
        conn.closeAllTrees()

    return efit_psi, efit_tau, efit_roa, efit_rmidd, efit_rmagx, efit_a


def _load_profile(shot: int, line_label: str, tht: int) -> dict[str, Any]:
    """Load raw profile arrays from spectroscopy tree."""
    conn = openTree(shot, "spectroscopy")
    base = _analysis_root(tht)
    branch = _line_to_branch(line_label)

    node_base = f"{base}.{branch}.profiles.{line_label}"
    try:
        pro, pro_expr = _try_get_data(conn, [f"{node_base}:pro"])
        proerr, _ = _try_get_data(conn, [f"{node_base}:proerr"])
        rho, rho_expr = _try_get_data(conn, [f"{node_base}:rho"])

        # Common optional nodes in this branch; if missing, defaults are used.
        tgood = None
        for expr in [f"{node_base}:tgood", f"{base}.{branch}.profiles:tgood"]:
            try:
                tgood = np.asarray(conn.get(expr).data())
                break
            except Exception:
                pass

        tinst = 0.0
        for expr in [f"{node_base}:tinst", f"{base}.{branch}.profiles:tinst"]:
            try:
                tinst = float(np.asarray(conn.get(expr).data()).squeeze())
                break
            except Exception:
                pass

        # Infer expected profile time length from profile array shape.
        pro_arr = np.asarray(pro)
        if pro_arr.ndim != 3:
            raise ValueError(f"Unexpected profile array shape {pro_arr.shape}")
        if pro_arr.shape[0] == 4:
            nt_expected = pro_arr.shape[2]
        elif pro_arr.shape[-1] == 4:
            nt_expected = pro_arr.shape[1]
        else:
            # Conservative fallback if profile field axis cannot be identified yet.
            nt_expected = pro_arr.shape[-1]

        tau = None
        tau_expr_candidates = [
            f"dim_of({rho_expr},0)",
            f"dim_of({rho_expr},1)",
            f"dim_of({pro_expr},0)",
            f"dim_of({pro_expr},1)",
            f"dim_of({pro_expr},2)",
        ]
        for dexpr in tau_expr_candidates:
            try:
                cand = np.asarray(conn.get(dexpr).data()).squeeze()
                if cand.ndim == 1 and cand.size == nt_expected:
                    tau = cand
                    break
            except Exception:
                pass

        if tau is None:
            # Last resort: integer sample index axis.
            tau = np.arange(nt_expected, dtype=float)
    finally:
        conn.closeAllTrees()

    return {
        "pro": pro,
        "proerr": proerr,
        "rho": rho,
        "tau": tau,
        "tgood": tgood,
        "tinst": tinst,
        "pro_expr": pro_expr,
    }


def _load_moment(shot: int, line_label: str, tht: int) -> MomentResult:
    """Load and normalize moment arrays to IDL-compatible shape [nch,3,nt]."""
    conn = openTree(shot, "spectroscopy")
    base = _analysis_root(tht)
    branch = _line_to_branch(line_label)

    node_base = f"{base}.{branch}.moments.{line_label}"
    try:
        mom_raw, mom_expr = _try_get_data(conn, [f"{node_base}:mom"])
        momerr_raw, _ = _try_get_data(conn, [f"{node_base}:err"])

        # In many trees, :U is the moment time-base and contains -1 for invalid slots.
        tau = None
        for expr in [f"{node_base}:u", f"dim_of({mom_expr},2)", f"dim_of({mom_expr},1)", f"dim_of({mom_expr},0)"]:
            try:
                cand = np.asarray(conn.get(expr).data()).squeeze()
                if cand.ndim == 1:
                    tau = cand
                    break
            except Exception:
                pass
    finally:
        conn.closeAllTrees()

    # Heuristic shape normalization.
    # Expected fields are 0th/1st/2nd moments.
    mom = np.asarray(mom_raw)
    momerr = np.asarray(momerr_raw)

    # Common layout from MDS: [3, nch, nt] -> [nch, 3, nt]
    if mom.ndim == 3 and mom.shape[0] == 3:
        mom = np.moveaxis(mom, 0, 1)
    if momerr.ndim == 3 and momerr.shape[0] == 3:
        momerr = np.moveaxis(momerr, 0, 1)

    # If layout is [nch, nt, 3], move to [nch, 3, nt]
    if mom.ndim == 3 and mom.shape[-1] == 3:
        mom = np.moveaxis(mom, -1, 1)
    if momerr.ndim == 3 and momerr.shape[-1] == 3:
        momerr = np.moveaxis(momerr, -1, 1)

    if mom.ndim != 3 or mom.shape[1] != 3:
        raise ValueError(f"Unexpected moment shape {mom.shape}; expected [nch,3,nt] after normalization")
    if momerr.ndim != 3 or momerr.shape[1] != 3:
        raise ValueError(
            f"Unexpected moment error shape {momerr.shape}; expected [nch,3,nt] after normalization"
        )

    nt = mom.shape[2]
    if tau is None:
        tau = np.arange(nt, dtype=float)
    tau = np.asarray(tau).squeeze()
    if tau.ndim != 1:
        tau = np.arange(nt, dtype=float)

    # Match IDL behavior: keep tau != -1 and slice data accordingly.
    if tau.size >= nt:
        tau_use = tau[:nt]
    else:
        tau_use = np.pad(tau, (0, nt - tau.size), constant_values=-1)
    good = np.where(tau_use != -1)[0]
    if good.size > 0:
        mom = mom[:, :, good]
        momerr = momerr[:, :, good]
        tau_use = tau_use[good]
    else:
        tau_use = np.arange(nt, dtype=float)

    return MomentResult(mom=mom, momerr=momerr, tau=tau_use)


def _load_lineint(shot: int, line_label: str, tht: int) -> LineIntResult | None:
    """Load line-integrated arrays if available.

    The original IDL uses `hirexsr_load_mlintptr`. Different C-Mod deployments
    expose this data under slightly different node names, so we try several.
    """
    conn = openTree(shot, "spectroscopy")
    base = _analysis_root(tht)
    branch = _line_to_branch(line_label)

    candidates = [
        f"{base}.{branch}.lineint.{line_label}:mlint",
        f"{base}.{branch}.mlint.{line_label}:mlint",
        f"{base}.{branch}.moments.{line_label}:mlint",
        f"{base}.{branch}.profiles.{line_label}:mlint",
    ]

    try:
        try:
            mlint, expr = _try_get_data(conn, candidates)
        except RuntimeError:
            return None

        # Expected fields roughly follow the IDL indexing used in the loop:
        # 0:freq,1:freq_err,2:ti,3:ti_err,4:psi,5:rmid,6:vpol,7:vpol_err,
        # 8:emiss,9:emiss_err,10:roa
        tau = _try_get_dim(conn, expr, dim_idx=2)
    finally:
        conn.closeAllTrees()

    arr = np.asarray(mlint)

    # Normalize to [nch, nfield, nt]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected mlint shape {arr.shape}; expected 3D")
    if arr.shape[1] < 11 and arr.shape[0] >= 11:
        arr = np.moveaxis(arr, 0, 1)
    if arr.shape[1] < 11:
        raise ValueError(f"Unexpected mlint shape {arr.shape}; need at least 11 fields on axis 1")

    freq = arr[:, 0, :]
    freq_err = arr[:, 1, :]
    ti = arr[:, 2, :]
    ti_err = arr[:, 3, :]
    psi = arr[:, 4, :]
    rmid = arr[:, 5, :]
    vpol = arr[:, 6, :]
    vpol_err = arr[:, 7, :]
    emiss = arr[:, 8, :]
    emiss_err = arr[:, 9, :]
    roa = arr[:, 10, :]

    omega = freq * 2.0 * np.pi
    omega_err = freq_err * 2.0 * np.pi
    vtor = omega * rmid
    vtor_err = omega_err * rmid

    # Approximate minor radius and axis from outer/inner points, matching IDL.
    a = (rmid[-1, :] - rmid[0, :]) / (roa[-1, :] - roa[0, :])
    rmagx = rmid[0, :] - a * roa[0, :]

    lint = {
        "emiss": emiss,
        "vtor": vtor,
        "vpol": vpol,
        "ti": ti,
        "Omega": omega,
        "freq": freq,
    }
    linterr = {
        "emiss_err": emiss_err,
        "vtor_err": vtor_err,
        "vpol_err": vpol_err,
        "ti_err": ti_err,
        "Omega_err": omega_err,
        "freq_err": freq_err,
    }
    return LineIntResult(
        lint=lint,
        linterr=linterr,
        tau=tau,
        psinorm=psi,
        rnorm=roa,
        Rmid=rmid,
        Rmagx=rmagx,
        a=a,
    )


# -----------------------------
# Public API
# -----------------------------
def hirexsr_load_result_py(
    shot: int,
    line: int | str,
    tht: int = 0,
    ti_fit: bool | None = None,
    omega_fit: bool | None = None,
) -> tuple[ProfileResult, MomentResult, LineIntResult | None]:
    """Python equivalent of IDL `hirexsr_load_result`.

    Parameters
    ----------
    shot
        C-Mod shot number.
    line
        Either an integer line id (1=W,2=Z,3=LYA1,4=X,5=MO4D,6=J) or label string.
    tht
        THACO analysis index.
    ti_fit, omega_fit
        Accepted for API compatibility. The original IDL loads external .sav fit
        files interactively. That behavior is intentionally not replicated here.

    Returns
    -------
    (profile, moment, lineint)
        Dataclass objects that mirror the original IDL output structure.
    """
    if ti_fit:
        print("[info] ti_fit requested but external .sav fit loading is not implemented in this Python rewrite.")
    if omega_fit:
        print("[info] omega_fit requested but external .sav fit loading is not implemented in this Python rewrite.")

    line_label = _line_to_label(line)

    # EFIT mapping arrays from analysis tree.
    efit_psi, efit_tau, efit_roa, efit_rmidd, efit_rmagx, efit_a = _load_efit_mapping(shot)

    # Spectroscopy profile arrays. If requested line has no data for this shot,
    # try common fallback labels to remain practical across legacy line mappings.
    raw = None
    selected_label = None
    profile_errors: list[str] = []
    for cand in _candidate_line_labels(line):
        try:
            raw = _load_profile(shot, cand, tht)
            selected_label = cand
            break
        except Exception as exc:
            profile_errors.append(f"{cand}: {exc}")

    if raw is None or selected_label is None:
        raise RuntimeError(
            "Could not load HIREX profile data for any candidate line label. "
            f"Tried: {_candidate_line_labels(line)}. Errors: {profile_errors}"
        )

    if selected_label != line_label:
        print(
            f"[info] Requested line '{line_label}' has no data for this shot; "
            f"using '{selected_label}' instead."
        )

    line_label = selected_label
    pro = _ensure_profile_cube(raw["pro"])        # [psi, tau, field]
    proerr = _ensure_profile_cube(raw["proerr"])  # [psi, tau, field]

    rho = np.asarray(raw["rho"])
    tau = np.asarray(raw["tau"]).squeeze()
    tgood = raw["tgood"]
    tinst = float(raw["tinst"])

    # Normalize rho to [psi, tau] and get psi coordinate from first time slice.
    if rho.ndim == 1:
        rho2d = np.repeat(rho[:, None], tau.size, axis=1)
    elif rho.ndim == 2:
        # rho often arrives as [psi,tau] or [tau,psi].
        if rho.shape[1] == tau.size:
            rho2d = rho
        elif rho.shape[0] == tau.size:
            rho2d = rho.T
        else:
            raise ValueError(f"Unexpected rho shape {rho.shape} for tau size {tau.size}")
    else:
        raise ValueError(f"Unexpected rho dimensionality: {rho.shape}")

    # Filter good-time samples (IDL: tau=tau(where(tgood eq 1))).
    tau_good, keep = _subset_good_times(tau, tgood)
    pro = pro[:, keep, :]
    proerr = proerr[:, keep, :]
    rho2d = rho2d[:, keep]

    psi = rho2d[:, 0]

    # Detect IDL m=1 split behavior using middle-point monotonicity check.
    mid = (psi.size - 1) // 2
    has_m1 = not (psi[mid] < psi[mid + 1])

    if has_m1:
        psi_base = psi[: mid + 1]
        pro0 = pro[: mid + 1, :, :]
        proerr0 = proerr[: mid + 1, :, :]
        pro1 = pro[mid + 1 : 2 * (mid + 1), :, :]
        proerr1 = proerr[mid + 1 : 2 * (mid + 1), :, :]
        psi_m1 = psi[mid + 1 : mid + 1 + pro1.shape[0]]
    else:
        psi_base = psi
        pro0 = pro
        proerr0 = proerr
        pro1 = None
        proerr1 = None
        psi_m1 = None

    # Map EFIT quantities to (psi,tau) grid (IDL: interp2d(...,/grid)).
    efit_roa_xy = _ensure_psi_tau(efit_roa, efit_psi.size, efit_tau.size)
    efit_rmidd_xy = _ensure_psi_tau(efit_rmidd, efit_psi.size, efit_tau.size)
    roa = _interp2d_grid(efit_roa_xy, efit_psi, efit_tau, psi_base, tau_good)
    rmid = _interp2d_grid(efit_rmidd_xy, efit_psi, efit_tau, psi_base, tau_good)
    if psi_m1 is not None:
        rmid_m1 = _interp2d_grid(efit_rmidd_xy, efit_psi, efit_tau, psi_m1, tau_good)
    else:
        rmid_m1 = None
    rmagx = np.interp(tau_good, efit_tau, efit_rmagx, left=np.nan, right=np.nan)
    a = np.interp(tau_good, efit_tau, efit_a, left=np.nan, right=np.nan)

    twopi = 2.0 * np.pi
    prof = {
        "emiss": pro0[:, :, 0],
        "vtor": pro0[:, :, 1] * twopi * rmid,
        "vpol": pro0[:, :, 2],
        "ti": pro0[:, :, 3] - tinst,
        "Omega": pro0[:, :, 1] * twopi,
        "freq": pro0[:, :, 1],
    }
    proferr = {
        "emiss_err": proerr0[:, :, 0],
        "vtor_err": proerr0[:, :, 1] * twopi * rmid,
        "vpol_err": proerr0[:, :, 2],
        "ti_err": proerr0[:, :, 3],
        "Omega_err": proerr0[:, :, 1] * twopi,
        "freq_err": proerr0[:, :, 1],
    }

    prof1_dict = None
    proferr1_dict = None
    if pro1 is not None and proerr1 is not None:
        if rmid_m1 is None:
            rmid_m1 = rmid
        prof1_dict = {
            "emiss": pro1[:, :, 0],
            "vtor": pro1[:, :, 1] * twopi * rmid_m1,
            "vpol": pro1[:, :, 2],
            "ti": pro1[:, :, 3] - tinst,
            "Omega": pro1[:, :, 1] * twopi,
            "freq": pro1[:, :, 1],
        }
        proferr1_dict = {
            "emiss_err": proerr1[:, :, 0],
            "vtor_err": proerr1[:, :, 1] * twopi * rmid_m1,
            "vpol_err": proerr1[:, :, 2],
            "ti_err": proerr1[:, :, 3],
            "Omega_err": proerr1[:, :, 1] * twopi,
            "freq_err": proerr1[:, :, 1],
        }

    profile = ProfileResult(
        prof=prof,
        proferr=proferr,
        tau=tau_good,
        psinorm=psi_base,
        rnorm=roa,
        Rmid=rmid,
        Rmagx=rmagx,
        a=a,
        prof1=prof1_dict,
        proferr1=proferr1_dict,
    )

    moment = _load_moment(shot, line_label, tht)
    lineint = _load_lineint(shot, line_label, tht)

    return profile, moment, lineint


def _shape_or_none(arr: Any) -> Any:
    return None if arr is None else tuple(np.asarray(arr).shape)


def _print_summary(profile: ProfileResult, moment: MomentResult, lineint: LineIntResult | None) -> None:
    print("=== hirexsr_load_result_py summary ===")
    print(f"profile.tau shape: {profile.tau.shape}")
    print(f"profile.psinorm shape: {profile.psinorm.shape}")
    print(f"profile.prof['freq'] shape: {profile.prof['freq'].shape}")
    print(f"profile.proferr['freq_err'] shape: {profile.proferr['freq_err'].shape}")
    print(f"profile.prof1 exists: {profile.prof1 is not None}")
    print(f"moment.mom shape: {moment.mom.shape}")
    print(f"moment.momerr shape: {moment.momerr.shape}")
    print(f"moment.tau shape: {moment.tau.shape}")

    if lineint is None:
        print("lineint: not available from current MDSplus node layout")
    else:
        print(f"lineint.tau shape: {lineint.tau.shape}")
        print(f"lineint.psinorm shape: {lineint.psinorm.shape}")
        print(f"lineint.lint['freq'] shape: {lineint.lint['freq'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python rewrite of hirexsr_load_result.pro")
    parser.add_argument("--shot", type=int, default=1140221012, help="C-Mod shot number")
    parser.add_argument("--line", default=6, help="Line index or label (default: 6)")
    parser.add_argument("--tht", type=int, default=1, help="THACO analysis index")
    args = parser.parse_args()

    line_in: int | str
    try:
        line_in = int(args.line)
    except ValueError:
        line_in = str(args.line)

    profile_out, moment_out, lineint_out = hirexsr_load_result_py(
        shot=args.shot,
        line=line_in,
        tht=args.tht,
    )
    _print_summary(profile_out, moment_out, lineint_out)
