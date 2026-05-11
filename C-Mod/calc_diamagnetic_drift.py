#!/usr/bin/env python3
"""
Compute electron and ion diamagnetic drift frequencies with time-dependent inputs.

Assumptions implemented:
- B_t is available as a full surface in major radius and time: [nR, nt]
- Zeff is 1D over time: [nt]
- ne, Te, Ti are 2D: [nspace, nt]
- psi_n and q are 2D in time as well: [nspace, nt] (or [nq, nt] for q grid)
- Flux-surface circumference is approximated as circular with C = 2*pi*r_minor.

The ion density is computed as ni = ne / Zeff(t).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys
import warnings

import eqtools as eq
import matplotlib.pyplot as plt
plt.ion()
import numpy as np

# Thomson scattering laser Te, Ne profiles
from get_Cmod_Data import YAG

# HIREXSR package necessary for V_phi(r), Z_eff -> n_i(r) profile data
from HIREXSR_py.hirexsr_get_profile_py import hirexsr_get_profile_py as _hxsr  # type: ignore[reportMissingImports]
from HIREXSR_py.zeff_neo_python import zeff_neo  # type: ignore[reportMissingImports]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    }
)


E_CHARGE = 1.602176634e-19  # [C]
TWO_PI = 2.0 * np.pi
C_MOD_R0_M = 0.68
C_MOD_A_MINOR_M = 0.22


@dataclass
class ProfileData:
    time_diag: np.ndarray          # [nt]
    r_major_diag_m: np.ndarray     # [ndiag, nt]
    ne_m3: np.ndarray           # [nspace, nt]
    te_eV: np.ndarray           # [nspace, nt]
    ti_eV: np.ndarray           # [nspace, nt]
    zeff: np.ndarray            # [nt]
    psi_hx: np.ndarray | None         # [npsi_hx, nt]
    rho_hx: np.ndarray | None         # [npsi_hx, nt]
    omega_tor_rad_s: np.ndarray | None   # [npsi_hx, nt]  toroidal rotation from HIREXSR
    omega_tor_err_rad_s: np.ndarray | None  # [npsi_hx, nt]  1-sigma error


@dataclass
class EquilibriumData:
    time_s: np.ndarray          # [nt_eq]
    psi_n_q: np.ndarray         # [nq, nt_eq]
    q_profile: np.ndarray       # [nq, nt_eq]
    b_t_t: np.ndarray           # [nR, nt_eq] full toroidal field surface B_t(R,t)
    r_major_full_m: np.ndarray  # [nR] full major-radius grid (inboard to outboard)
    r_minor_q_m: np.ndarray     # [nq, nt_eq]
    r_major_q_m: np.ndarray     # [nq, nt_eq]
    elongation_t: np.ndarray    # [nt_eq]
    circumference_q_m: np.ndarray  # [nq, nt_eq]
    q95: np.ndarray             # [nt_eq]
    b_pol_q_T: np.ndarray       # [nq, nt_eq]  poloidal field on q-grid, computed at midplane

@dataclass
class DiamagneticResult:
    time_s: np.ndarray
    psi_n: np.ndarray           # [nspace, nt]
    q: np.ndarray               # [nspace, nt]
    circumference_m: np.ndarray  # [nspace, nt]
    dpe_dr_pa_m: np.ndarray      # [nspace, nt]
    dpi_dr_pa_m: np.ndarray      # [nspace, nt]
    ne_m3: np.ndarray           # [nspace, nt]
    ni_m3: np.ndarray           # [nspace, nt]
    te_eV: np.ndarray           # [nspace, nt]
    ti_eV: np.ndarray           # [nspace, nt]
    f_star_e_Hz: np.ndarray     # [nspace, nt]
    f_star_i_Hz: np.ndarray     # [nspace, nt]
    omega_star_e_rad_s: np.ndarray
    omega_star_i_rad_s: np.ndarray
    f_star_e_tor_Hz: np.ndarray  # [nspace, nt]  toroidal drift frequency (uses B_pol)
    f_star_i_tor_Hz: np.ndarray  # [nspace, nt]
    omega_star_e_tor_rad_s: np.ndarray
    omega_star_i_tor_rad_s: np.ndarray

def do_equilibrium_dimensionality_check(
    time_s: np.ndarray,
    psi_n_q: np.ndarray,
    q_profile: np.ndarray,
    b_t_t: np.ndarray,
    r_major_full_m: np.ndarray,
    r_minor_q_m: np.ndarray,
    r_out_q: np.ndarray,
    elongation_t: np.ndarray,
    circumference_q_m: np.ndarray,
    q95: np.ndarray
):
    # Ensure dimensionality is correct for eqtools loaded equilibrium profiles
    nt_eq = time_s.size
    assert psi_n_q.shape == q_profile.shape, f"Expected psi_n_q and q_profile to have the same shape, got {psi_n_q.shape} vs {q_profile.shape}"
    assert b_t_t.shape == (r_major_full_m.size, nt_eq), (
        f"Expected b_t_t to have shape ({r_major_full_m.size}, {nt_eq}), got {b_t_t.shape}"
    )
    assert r_minor_q_m.shape == q_profile.shape, f"Expected r_minor_q_m to have the same shape as q_profile, got {r_minor_q_m.shape} vs {q_profile.shape}"
    assert r_out_q.shape == q_profile.shape, f"Expected r_out_q to have the same shape as q_profile, got {r_out_q.shape} vs {q_profile.shape}"
    assert elongation_t.shape == (nt_eq,), f"Expected elongation_t to have shape ({nt_eq},), got {elongation_t.shape}"
    assert circumference_q_m.shape == q_profile.shape, f"Expected circumference_q_m to have the same shape as q_profile, got {circumference_q_m.shape} vs {q_profile.shape}"
    assert q95.shape == (nt_eq,), f"Expected q95 to have shape ({nt_eq},), got {q95.shape}"

def do_profile_dimensionality_check(t_diag, r_all, te_eV, ti_eV, ne_m3, psi_hx, zeff_t):
    # Ensure dimensionality is correct for YAG-loaded profiles
    nt_diag = t_diag.size
    nspace = r_all.shape[0]
    assert r_all.shape == (nspace, nt_diag), f"Expected r_all to have shape ({nspace}, {nt_diag}), got {r_all.shape}"
    assert te_eV.shape == (nspace, nt_diag), f"Expected te_eV to have shape ({nspace}, {nt_diag}), got {te_eV.shape}"
    assert ti_eV.shape[1] ==  nt_diag, f"Expected ti_eV to have time shape ( {nt_diag}), got {ti_eV.shape}"
    assert ne_m3.shape == (nspace, nt_diag), f"Expected ne_m3 to have shape ({nspace}, {nt_diag}), got {ne_m3.shape}"
    # if psi_hx is not None:
    #     assert psi_hx.shape == (nspace, nt_diag), f"Expected psi_hx to have shape ({nspace}, {nt_diag}), got {psi_hx.shape}"
    assert zeff_t.shape == (nt_diag,), f"Expected zeff_t to have shape ({nt_diag},), got {zeff_t.shape}"

def _safe_keep_channels(nchan: int, drop: list[int]) -> np.ndarray:
    drop_valid = [i for i in drop if 0 <= i < nchan]
    return np.delete(np.arange(nchan), drop_valid)


def _interpolate_below_threshold_radial(
    values_st: np.ndarray,
    r_major_st: np.ndarray,
    min_value: float,
) -> np.ndarray:
    """Replace low-value samples using radial interpolation at each time slice.

    Any non-finite value or value below min_value is treated as invalid and
    replaced using 1D interpolation in major radius for that time slice.
    """
    values = np.asarray(values_st, dtype=float)
    rvals = np.asarray(r_major_st, dtype=float)
    if values.shape != rvals.shape:
        raise ValueError(f"Interpolation shape mismatch: {values.shape} vs {rvals.shape}")

    out = np.array(values, dtype=float, copy=True)
    nspace, nt = out.shape

    for it in range(nt):
        y = out[:, it]
        x = rvals[:, it]

        valid = np.isfinite(y) & np.isfinite(x) & (y >= min_value)
        if np.all(valid):
            continue

        if np.count_nonzero(valid) >= 2:
            xv = x[valid]
            yv = y[valid]
            order = np.argsort(xv)
            xv = xv[order]
            yv = yv[order]

            # np.interp requires increasing x; duplicate x values are collapsed.
            keep = np.concatenate(([True], np.diff(xv) > 1e-12))
            xv = xv[keep]
            yv = yv[keep]

            if xv.size >= 2:
                out[:, it] = np.interp(x, xv, yv)
            else:
                fill_val = max(float(yv[0]), float(min_value))
                out[:, it] = np.full(nspace, fill_val, dtype=float)
        elif np.count_nonzero(valid) == 1:
            fill_val = max(float(y[valid][0]), float(min_value))
            out[:, it] = np.full(nspace, fill_val, dtype=float)
        else:
            out[:, it] = np.full(nspace, float(min_value), dtype=float)

    return out


def _smooth_1d_robust_local_poly(
    x: np.ndarray,
    y: np.ndarray,
    span_fraction: float = 0.45,
    poly_order: int = 2,
    robust_iters: int = 2,
) -> np.ndarray:
    """Robust local polynomial smoothing for sparse 1D radial profiles."""
    x_in = np.asarray(x, dtype=float)
    y_in = np.asarray(y, dtype=float)
    valid = np.isfinite(x_in) & np.isfinite(y_in)

    if np.count_nonzero(valid) < max(3, poly_order + 1):
        return np.array(y_in, dtype=float, copy=True)

    xv = x_in[valid]
    yv = y_in[valid]
    order = np.argsort(xv)
    xs = xv[order]
    ys = yv[order]

    keep = np.concatenate(([True], np.diff(xs) > 1e-12))
    xs = xs[keep]
    ys = ys[keep]

    n = xs.size
    if n < max(3, poly_order + 1):
        return np.array(y_in, dtype=float, copy=True)

    nwin = int(np.ceil(max(0.2, min(1.0, span_fraction)) * n))
    nwin = max(poly_order + 2, nwin)
    if nwin % 2 == 0:
        nwin += 1
    nwin = min(nwin, n)

    robust_w = np.ones(n, dtype=float)
    yhat = np.array(ys, dtype=float, copy=True)

    half = nwin // 2
    for _ in range(max(1, robust_iters + 1)):
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, lo + nwin)
            lo = max(0, hi - nwin)

            xw = xs[lo:hi]
            yw = ys[lo:hi]
            rw = robust_w[lo:hi]

            dist = np.abs(xw - xs[i])
            dmax = max(np.max(dist), 1e-12)
            lw = (1.0 - (dist / dmax) ** 3) ** 3
            w = lw * rw

            if np.count_nonzero(w > 1e-12) < (poly_order + 1):
                yhat[i] = np.average(yw, weights=np.maximum(w, 1e-12))
                continue

            xc = xw - xs[i]
            deg = min(poly_order, xw.size - 1)
            v = np.vstack([xc ** k for k in range(deg + 1)]).T
            ws = np.sqrt(np.maximum(w, 1e-12))
            vw = v * ws[:, np.newaxis]
            yw_w = yw * ws
            coeff, *_ = np.linalg.lstsq(vw, yw_w, rcond=None)
            yhat[i] = coeff[0]

        resid = ys - yhat
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-12
        u = resid / (6.0 * sigma)
        robust_w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)

    y_smooth_valid = np.interp(xv, xs, yhat)
    y_out = np.array(y_in, dtype=float, copy=True)
    y_out[valid] = y_smooth_valid

    if np.any(~valid):
        y_out[~valid] = np.interp(x_in[~valid], xs, yhat)

    return y_out


def _smooth_profiles_by_radius(
    values_st: np.ndarray,
    r_major_st: np.ndarray,
    span_fraction: float = 0.45,
    poly_order: int = 2,
    robust_iters: int = 2,
) -> np.ndarray:
    """Smooth each time-slice on its major-radius grid with robust local fits."""
    values = np.asarray(values_st, dtype=float)
    rvals = np.asarray(r_major_st, dtype=float)
    if values.shape != rvals.shape:
        raise ValueError(f"Smoothing shape mismatch: {values.shape} vs {rvals.shape}")

    out = np.array(values, dtype=float, copy=True)
    for it in range(out.shape[1]):
        out[:, it] = _smooth_1d_robust_local_poly(
            rvals[:, it],
            out[:, it],
            span_fraction=span_fraction,
            poly_order=poly_order,
            robust_iters=robust_iters,
        )
    return out


def _clean_low_te_ne_profiles(
    te_keV_st: np.ndarray,
    ne_m3_st: np.ndarray,
    r_major_st: np.ndarray,
    te_floor_keV: float = 0.5,
    ne_floor_m3: float = 0.5e19,
    smooth_after_cleanup: bool = True,
    smooth_span_fraction: float = 0.45,
    smooth_poly_order: int = 2,
    smooth_robust_iters: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean low Te/Ne points via interpolation, then robust radial smoothing."""
    te_clean = _interpolate_below_threshold_radial(te_keV_st, r_major_st, min_value=te_floor_keV)
    ne_clean = _interpolate_below_threshold_radial(ne_m3_st, r_major_st, min_value=ne_floor_m3)

    if smooth_after_cleanup:
        te_clean = _smooth_profiles_by_radius(
            te_clean,
            r_major_st,
            span_fraction=smooth_span_fraction,
            poly_order=smooth_poly_order,
            robust_iters=smooth_robust_iters,
        )
        ne_clean = _smooth_profiles_by_radius(
            ne_clean,
            r_major_st,
            span_fraction=smooth_span_fraction,
            poly_order=smooth_poly_order,
            robust_iters=smooth_robust_iters,
        )

    return te_clean, ne_clean




def _ellipse_circumference(a_minor: np.ndarray, kappa: np.ndarray | float) -> np.ndarray:
    """Ellipse circumference via Ramanujan's second approximation.

    Parameters
    ----------
    a_minor : ndarray
        Minor radius [m].
    kappa : ndarray or float
        Elongation, so semi-major axis is b = kappa * a.
    """
    a = np.maximum(np.asarray(a_minor, dtype=float), 1e-6)
    k = np.maximum(np.asarray(kappa, dtype=float), 1e-6)
    b = np.maximum(k * a, 1e-6)

    h = ((a - b) ** 2) / np.maximum((a + b) ** 2, 1e-12)
    return np.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + np.sqrt(np.maximum(4.0 - 3.0 * h, 1e-12))))


def _load_zeff_timeseries(shot: int, time_diag: np.ndarray) -> np.ndarray:
    """Load Zeff(t) from zeff_neo and interpolate onto a target timebase."""
    t = np.asarray(time_diag, dtype=float).squeeze()
    if t.size < 2:
        return np.ones_like(t)

    dt = 0.1 # for Z_eff, resample after calculation into YAG timebase
    trange = (float(np.nanmin(t)), float(np.nanmax(t)))

    if zeff_neo is None:
        print("Warning: zeff_neo not available; using Zeff(t)=1 fallback.")
        return np.ones_like(t)

    try:
        # Zeff_neo has a consistency check, will raise exception if values are invalid
        zeff_out = zeff_neo(
            shot=shot,
            dt=dt,
            trange=trange,
            plot=False,
            verbose=False,
        )
        zeff_vals, zeff_times = zeff_out[0], zeff_out[1]


        zeff_interp = np.interp(t, zeff_times, zeff_vals)
        
        return zeff_interp

    except Exception as exc:
        print(f"Warning: falling back to Zeff(t)=1 due to zeff_neo error: {exc}")
        return np.ones_like(t)


def _load_ti_from_hirexsr(
    shot: int,
    time_diag: np.ndarray,
    r_diag: np.ndarray,
    line: int = 2,
    tht: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Ti(psi_n, t) from HIREXSR inverted profiles and interpolate onto the
    (r_grid_yag, time_yag) grid used by ProfileData.  Returns fallback_ti_eV on failure.

    HIREXSR stores Ti in keV; the result is converted to eV.
    The HIREXSR psinorm convention (sqrt of normalised poloidal flux) matches
    the psi_n convention used throughout this module.

    """



    if _hxsr is None:
        raise RuntimeError("HIREXSR profile loader is unavailable in this environment.")

    result = _hxsr(shot=shot, line=line, tht=tht, quiet=True)
    ti_keV = np.asarray(result.ti, dtype=float)
    psi_hx = np.asarray(result.psi, dtype=float)
    t_hx = np.asarray(result.time, dtype=float).squeeze()
    rho_hx = np.asarray(result.r_maj, dtype=float)
    omega_hx = np.asarray(result.omg, dtype=float)
    omega_err_hx = np.asarray(result.omgerr, dtype=float)

    if psi_hx.ndim == 1:
        psi_hx = np.repeat(psi_hx[:, np.newaxis], t_hx.size, axis=1)

    tierr = getattr(result, "tierr", None)
    emisserr = getattr(result, "emisserr", None)

    valid = np.isfinite(ti_keV) & np.isfinite(psi_hx)
    if tierr is not None:
        valid &= np.asarray(tierr, dtype=float) < 0.5
    if emisserr is not None:
        valid &= np.asarray(emisserr, dtype=float) < 0.5

    ti_keV = np.where(valid, np.clip(ti_keV, 0.0, 20.0), 0.0)
    psi_hx = np.clip(psi_hx, 0.0, 1.0)

    # Enforce that the boundary is at least psi=1, and Ti=0 at the boundary, to avoid unphysical extrapolation in the diamagnetic drift calculation.
    psi_hx = np.concatenate((psi_hx, np.ones((1, psi_hx.shape[1]))), axis=0)
    rho_hx = np.concatenate((rho_hx, .91*np.ones((1, rho_hx.shape[1]))), axis=0)
    ti_keV = np.concatenate((ti_keV, np.zeros((1, ti_keV.shape[1]))), axis=0)

    ti_eV_t = _interp_space_time_in_time(ti_keV * 1e3, t_hx, time_diag)
    psi_hx_t = _interp_space_time_in_time(psi_hx, t_hx, time_diag)
    rho_hx_t = _interp_space_time_in_time(rho_hx, t_hx, time_diag)
    # omega_hx / omega_err_hx do not get the boundary row appended (no physical
    # constraint at the edge for rotation), so interpolate on the original shape.
    omega_hx_t = _interp_space_time_in_time(omega_hx, t_hx, time_diag)
    omega_err_hx_t = _interp_space_time_in_time(omega_err_hx, t_hx, time_diag)
    return ti_eV_t, np.clip(psi_hx_t, 0.0, 1.0), rho_hx_t, omega_hx_t, omega_err_hx_t



def load_profiles_for_shot(
    shot: int,
    dropChansMain: list[int] | None = None,
    dropChansEdge: list[int] | None = None,
    fallback_ti_eV: float | None = None,
    te_floor_keV: float = 0.5,
    ne_floor_m3: float = 0.5e19,
    smooth_after_cleanup: bool = True,
    smooth_span_fraction: float = 0.45,
    smooth_poly_order: int = 2,
    smooth_robust_iters: int = 2,
    line: int = 2,
    tht: int = 0,
) -> ProfileData:
    """
    Load YAG Thomson profiles as channel-by-time arrays.

    Follows YAG.return_Profile behavior by concatenating edge channels when available,
    then sorting by mapped major radius.

    Notes:
    - Te from YAG is treated as keV and converted to eV.
    - Ti is loaded from HIREXSR He-like inversion (hirexsr_get_profile_py, line=2)
      and interpolated onto the YAG psi_n / time grid.  Falls back to Ti = Te if
      HIREXSR data are not available.
    - Set hard limits on Ti errors to avoid unphysical values propagating into the diamagnetic drift calculation.
    - Use the Te spatial channel location as the diagnostic nspace grid
    - Zeff(t) is loaded from zeff_neo_python and interpolated to YAG time.

    """

    # Get electron density and temperature profiles from YAG-Thomson scattering
    if dropChansMain is None:
        dropChansMain = [3]
    if dropChansEdge is None:
        dropChansEdge = [0, 1, 2, 3]

    yag = YAG(shot)

    t_diag = np.asarray(yag.time, dtype=float).squeeze()
    nt_diag = t_diag.size

    te_core = np.asarray(yag.Te, dtype=float)
    ne_core = np.asarray(yag.Ne, dtype=float)
    r_core = np.asarray(yag.R_Map, dtype=float)

    keep_core = _safe_keep_channels(te_core.shape[0], dropChansMain)
    te_core = te_core[keep_core, :]
    ne_core = ne_core[keep_core, :]
    r_core = r_core[keep_core, :]

    has_edge = all(
        hasattr(yag, attr)
        for attr in ["Te_Edge", "Ne_Edge", "R_Map_Edge", "time_Edge"]
    )

    if has_edge:
        t_edge = np.asarray(yag.time_Edge, dtype=float).squeeze()
        nt_edge = t_edge.size

        te_edge = np.asarray(yag.Te_Edge, dtype=float)
        ne_edge = np.asarray(yag.Ne_Edge, dtype=float)
        r_edge = np.asarray(yag.R_Map_Edge, dtype=float)

        keep_edge = _safe_keep_channels(te_edge.shape[0], dropChansEdge)
        te_edge = te_edge[keep_edge, :]
        ne_edge = ne_edge[keep_edge, :]
        r_edge = r_edge[keep_edge, :]

        # Interpolate edge data onto the core timebase, then concatenate channels.
        te_edge = _interp_space_time_in_time(te_edge, t_edge, t_diag)
        ne_edge = _interp_space_time_in_time(ne_edge, t_edge, t_diag)
        r_edge = _interp_space_time_in_time(r_edge, t_edge, t_diag)

        te_all = np.concatenate((te_core, te_edge), axis=0)
        ne_all = np.concatenate((ne_core, ne_edge), axis=0)
        r_all = np.concatenate((r_core, r_edge), axis=0)
    else:
        te_all = te_core
        ne_all = ne_core
        r_all = r_core

    # Match return_Profile behavior by sorting channels in mapped radius.
    r_mean = np.nanmean(r_all, axis=1)
    order = np.argsort(r_mean)

    te_all = te_all[order, :]
    ne_all = ne_all[order, :]
    r_all = r_all[order, :]

    te_all, ne_all = _clean_low_te_ne_profiles(
        te_all,
        ne_all,
        r_all,
        te_floor_keV=te_floor_keV,
        ne_floor_m3=ne_floor_m3,
        smooth_after_cleanup=smooth_after_cleanup,
        smooth_span_fraction=smooth_span_fraction,
        smooth_poly_order=smooth_poly_order,
        smooth_robust_iters=smooth_robust_iters,
    )

    te_eV = te_all * 1e3
    ne_m3 = ne_all

    # Get Ti from HIREXSR, with times mapped to YAG. Keep spatial points mapped to psi, interpolate on equilibrium grid later
    if not fallback_ti_eV:
        try:
            ti_eV, psi_hx, rho_hx, omega_tor, omega_tor_err = _load_ti_from_hirexsr(
                shot,
                t_diag,
                r_diag=r_all,
                line=line,
                tht=tht,
            )
        except Exception as exc:
            print(f"Warning: Ti from HIREXSR unavailable, falling back to Ti=Te. Reason: {exc}")
            ti_eV = te_eV
            psi_hx = None
            rho_hx = None
            omega_tor = None
            omega_tor_err = None
    else:        
        print("Warning: using Ti = Te*correction fallback instead of HIREXSR data.")
        ti_eV = te_eV * fallback_ti_eV
        psi_hx = None
        rho_hx = None
        omega_tor = None
        omega_tor_err = None
    # Load Zeff(t) from zeff_neo_python, interpolated to YAG timebase.
    zeff_t = _load_zeff_timeseries(shot, t_diag)

    do_profile_dimensionality_check(t_diag, r_all, te_eV, ti_eV, ne_m3, psi_hx, zeff_t)

    return ProfileData(
        time_diag=t_diag,
        r_major_diag_m=r_all,
        ne_m3=ne_m3,
        te_eV=te_eV,
        ti_eV=ti_eV,
        zeff=zeff_t,
        psi_hx=psi_hx,
        rho_hx=rho_hx,
        omega_tor_rad_s=omega_tor,
        omega_tor_err_rad_s=omega_tor_err,
    )


def load_equilibrium_for_shot(shot: int, tree: str = "efit20", diagnosticPlot=False) -> EquilibriumData:
    """
    Load equilibrium data from eqtools EFIT tree and convert to time-dependent q, psi_n, and B_t.
    """
    eq_f = eq.CModEFIT.CModEFITTree(shot, tree=tree)

    t_eq = np.asarray(eq_f.getTimeBase(), dtype=float).squeeze()
    nt_eq = t_eq.size

    q_raw = np.asarray(eq_f.getQProfile(), dtype=float).T
    psi_raw = np.asarray(eq_f.getRmidPsi(), dtype=float).T

    psi_n_q = _normalize_psi_to_psi_n(psi_raw)

    r_grid_full_m = np.asarray(eq_f.getRGrid(), dtype=float).squeeze()

    # Build full toroidal-field surface B_t(R,t) from F(t,...) / R.
    # F from eqtools may be scalar, [nt], [nt,nR], or [nR,nt], so normalize to [nR,nt].
    f_raw = np.asarray(eq_f.getF(), dtype=float)
    if f_raw.ndim == 0:
        b_t_t = np.full((r_grid_full_m.size, nt_eq), float(f_raw)) / r_grid_full_m[:, np.newaxis]
    elif f_raw.ndim == 1:
        if f_raw.size == nt_eq:
            b_t_t = f_raw[np.newaxis, :] / r_grid_full_m[:, np.newaxis]
        elif f_raw.size == r_grid_full_m.size:
            b_t_t = np.repeat((f_raw / np.maximum(r_grid_full_m, 1e-8))[:, np.newaxis], nt_eq, axis=1)
        else:
            raise ValueError(f"Unexpected getF() shape {f_raw.shape} for nt={nt_eq}, nR={r_grid_full_m.size}")
    elif f_raw.ndim == 2:
        if f_raw.shape == (nt_eq, r_grid_full_m.size):
            b_t_t = (f_raw / r_grid_full_m[np.newaxis, :]).T
        elif f_raw.shape == (r_grid_full_m.size, nt_eq):
            b_t_t = f_raw / r_grid_full_m[:, np.newaxis]
        else:
            raise ValueError(f"Unexpected getF() 2D shape {f_raw.shape} for nt={nt_eq}, nR={r_grid_full_m.size}")
    else:
        raise ValueError(f"Unsupported getF() dimensionality: {f_raw.shape}")

    q95 = eq_f.getQ95()

    r_mag_t = np.asarray(eq_f.getMagR(), dtype=float).squeeze()
    # if r_mag_raw.ndim == 0:
    #     r_mag_t = np.full(nt_eq, float(r_mag_raw))
    # else:
    #     r_mag_t = r_mag_raw


    r_out_q = eq_f.getRmidPsi().T # Ensure dimensions are [space, time]

    r_minor_q_m = np.clip(np.abs(r_out_q - r_mag_t[np.newaxis, :]), 1e-3, None)

    elongation_t = np.asarray(eq_f.getElongation(), dtype=float).squeeze()
    # if elongation_t.ndim == 0:
    #     elongation_t = np.full(nt_eq, float(elongation_t))
    # else:
    #     elongation_t = kappa_raw
    elongation_t = np.clip(elongation_t, 1.0, None)

    circumference_q_m = _ellipse_circumference(r_minor_q_m, elongation_t[np.newaxis, :])

    # Ensure dimensionality is correct
    do_equilibrium_dimensionality_check(
        t_eq,
        psi_n_q,
        q_raw,
        b_t_t,
        r_grid_full_m,
        r_minor_q_m,
        r_out_q,
        elongation_t,
        circumference_q_m,
        q95,
    )
  
    if diagnosticPlot:
        plot_magnetic_equilibrium(r_out_q, _compute_b_pol_midplane(eq_f, r_out_q, nt_eq), t_eq, eq_f)

    return EquilibriumData(
        time_s=t_eq,
        psi_n_q=psi_n_q,
        q_profile=q_raw,
        b_t_t=b_t_t,
        r_major_full_m=r_grid_full_m,
        r_minor_q_m=r_minor_q_m,
        r_major_q_m=r_out_q,
        elongation_t=elongation_t,
        circumference_q_m=circumference_q_m,
        q95=q95,
        b_pol_q_T=_compute_b_pol_midplane(eq_f, r_out_q, nt_eq),
    )


def plot_magnetic_equilibrium(r_out_q: np.ndarray, b_pol_q_T: np.ndarray, t_eq: np.ndarray, eq_f: eq.CModEFIT.CModEFITTree):
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,2,figsize=(7,2.5),layout='constrained', sharey=True);
    ctr1 = axs[0].contourf(r_out_q,np.tile(t_eq, (r_out_q.shape[0], 1)),b_pol_q_T,levels=20,zorder=-5);
    axs[0].set_xlabel('R-major [m]');
    axs[0].set_ylabel('Time [s]');
    plt.colorbar(ctr1,label=r'$B_\theta$ [T]', ax=axs[0]);
    axs[0].set_rasterization_zorder(-1);
    axs[0].grid();

    ctr2 = axs[1].contourf(eq_f.getRGrid(),t_eq,(eq_f.getF()/eq_f.getRGrid()),levels=20,zorder=-5);
    axs[1].set_xlabel('R-major [m]');
    # axs[1].set_ylabel('Time [s]');
    plt.colorbar(ctr2,label=r'$B_\phi$ [T]', ax=axs[1]);
    axs[1].set_rasterization_zorder(-1);
    axs[1].grid();
    

    fig.savefig('../output_plots/C_Mod_B_profiles.pdf',transparent=True);
    
    plt.show()

def _compute_b_pol_midplane(
    eq_f,
    r_major_q_m: np.ndarray,
    nt_eq: int,
) -> np.ndarray:
    """Compute the poloidal magnetic field at the outboard midplane on the q-grid.

    At the midplane (Z=0) up-down symmetry gives dPsi/dZ = 0, so B_r = 0 and
    B_p = B_z = (1/R) * dPsi/dR.

    The 2D flux grid from eqtools has shape [nt, nZ, nR] (or [nZ, nR] when nt=1).
    The result is interpolated onto the outboard midplane major-radius grid used
    for the q-profile, giving shape [nq, nt_eq].
    """
    psi_2d = np.asarray(eq_f.getFluxGrid(), dtype=float)  # [nt, nZ, nR]
    R_grid = np.asarray(eq_f.getRGrid(length_unit='m'), dtype=float)   # [nR]
    Z_grid = np.asarray(eq_f.getZGrid(length_unit='m'), dtype=float)   # [nZ]

    iz_mid = int(np.argmin(np.abs(Z_grid)))

    # Extract midplane psi: ensure shape [nR, nt_eq]
    if psi_2d.ndim == 3:
        psi_mid = psi_2d[:, iz_mid, :].T  # [nR, nt]
    else:
        # Fallback for single-time case
        psi_mid = psi_2d[iz_mid, :][:, np.newaxis]  # [nR, 1]
        psi_mid = np.repeat(psi_mid, nt_eq, axis=1)

    # B_z = (1/R) * dPsi/dR at midplane; B_r = 0 by symmetry
    dPsi_dR = np.gradient(psi_mid, R_grid, axis=0)       # [nR, nt]
    B_pol_mid = dPsi_dR / R_grid[:, np.newaxis]  # [nR, nt]

    # Interpolate onto the outboard midplane R grid used for q-profile [nq, nt]
    nq = r_major_q_m.shape[0]
    b_pol_q = np.empty((nq, nt_eq), dtype=float)
    for it in range(nt_eq):
        b_pol_q[:, it] = np.interp(r_major_q_m[:, it], R_grid, B_pol_mid[:, it])

    # plt.figure();plt.contour(r_major_q_m,np.tile(np.arange(nt_eq),(len(r_major_q_m),1)),b_pol_q,levels=20);
    # plt.xlabel('R [m]');plt.ylabel('Time [s]');
    # plt.colorbar(label=r'$B_\theta$ [T]');plt.show()
    return b_pol_q


def _normalize_psi_to_psi_n(psi_space_time: np.ndarray) -> np.ndarray:
    psi_n = np.empty_like(psi_space_time)
    for it in range(psi_space_time.shape[1]):
        psi_col = psi_space_time[:, it]
        denom = psi_col[-1] - psi_col[0]
        if np.isclose(denom, 0.0):
            raise ValueError("Cannot normalize psi: zero span in a time slice.")
        psi_n[:, it] = np.sqrt(np.clip((psi_col - psi_col[0]) / denom, 0.0, 1.0))
    return psi_n


def _interp_space_time_in_time(
    data_st: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
) -> np.ndarray:
    if np.array_equal(t_src, t_dst):
        return data_st

    out = np.empty((data_st.shape[0], t_dst.size), dtype=float)
    for i in range(data_st.shape[0]):
        out[i, :] = np.interp(t_dst, t_src, data_st[i, :])
    return out


def _interp_1d_sorted(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    return np.interp(x_new, xs, ys)


def _interp_columns_to_grid(
    y_src: np.ndarray,
    x_src: np.ndarray,
    x_dst: np.ndarray,
) -> np.ndarray:
    """Interpolate each time column of y(x,t) onto x_dst(:,t)."""
    if y_src.shape != x_src.shape:
        raise ValueError(
            f"Source value/coordinate shape mismatch: {y_src.shape} vs {x_src.shape}."
        )
    if y_src.shape[1] != x_dst.shape[1]:
        raise ValueError(
            f"Time dimension mismatch for interpolation: {y_src.shape} vs {x_dst.shape}."
        )

    out = np.empty_like(x_dst, dtype=float)
    for it in range(x_dst.shape[1]):
        out[:, it] = _interp_1d_sorted(x_dst[:, it], x_src[:, it], y_src[:, it])
    return out


def _map_ti_psi_to_diagnostic_radius_grid(
    ti_eV_t: np.ndarray,
    rho_hx_t: np.ndarray,
    psi_hx_t: np.ndarray,
    psi_n_q_t: np.ndarray,
    r_major_q_t: np.ndarray,
    r_major_diag_t: np.ndarray,
) -> np.ndarray:
    """Map Ti from HIREXSR psi grid to diagnostic radial grid via equilibrium grids.

    Mapping chain (all at the same timebase):
      1) (psi_hx_t -> psi_n_q_t)
      2) (r_major_q_t -> r_major_diag_t)
    """
    # ti_q_eV = _interp_columns_to_grid(ti_eV_t, psi_hx_t, psi_n_q_t)
    # interp_out = _interp_columns_to_grid(ti_q_eV, r_major_q_t, r_major_diag_t)
    ti_q_eV = _interp_columns_to_grid(ti_eV_t, rho_hx_t, r_major_diag_t)
    return ti_q_eV


def _gradient_axis0_nonuniform(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Vectorized first derivative along axis 0 for per-column nonuniform x grids."""
    if y.shape != x.shape:
        raise ValueError(f"Gradient shape mismatch: {y.shape} vs {x.shape}.")
    if y.shape[0] < 2:
        raise ValueError("Need at least two radial points to compute gradients.")

    grad = np.empty_like(y, dtype=float)
    dx_lo = np.maximum(x[1, :] - x[0, :], 1e-12)
    dx_hi = np.maximum(x[-1, :] - x[-2, :], 1e-12)

    grad[0, :] = (y[1, :] - y[0, :]) / dx_lo
    grad[-1, :] = (y[-1, :] - y[-2, :]) / dx_hi

    if y.shape[0] > 2:
        dx_mid = np.maximum(x[2:, :] - x[:-2, :], 1e-12)
        grad[1:-1, :] = (y[2:, :] - y[:-2, :]) / dx_mid

    return grad


def _gradient_axis0_nonuniform_polyfit(
    y: np.ndarray,
    x: np.ndarray,
    fit_points: int = 5,
    poly_order: int = 3,
) -> np.ndarray:
    """Higher-order first derivative via local polynomial fits along axis 0.

    For each radial point and time slice, fit a local polynomial y(x) and
    evaluate dy/dx at the target point. This is typically less noisy than
    low-order finite differences on sparse grids.
    """
    if y.shape != x.shape:
        raise ValueError(f"Gradient shape mismatch: {y.shape} vs {x.shape}.")
    if y.shape[0] < 2:
        raise ValueError("Need at least two radial points to compute gradients.")

    nrad, nt = y.shape
    grad = np.empty_like(y, dtype=float)

    nfit = int(max(3, fit_points))
    if nfit % 2 == 0:
        nfit += 1
    nfit = min(nfit, nrad if nrad % 2 == 1 else max(3, nrad - 1))
    pord = int(max(1, min(poly_order, nfit - 1)))

    half = nfit // 2
    for it in range(nt):
        xcol = np.asarray(x[:, it], dtype=float)
        ycol = np.asarray(y[:, it], dtype=float)

        for i in range(nrad):
            lo = max(0, i - half)
            hi = min(nrad, lo + nfit)
            lo = max(0, hi - nfit)

            xw = xcol[lo:hi]
            yw = ycol[lo:hi]

            order = np.argsort(xw)
            xs = xw[order]
            ys = yw[order]

            uniq = np.concatenate(([True], np.diff(xs) > 1e-12))
            xs = xs[uniq]
            ys = ys[uniq]

            if xs.size < 2:
                grad[i, it] = 0.0
                continue

            deg = min(pord, xs.size - 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.RankWarning)
                coeff = np.polyfit(xs, ys, deg=deg)
            dcoeff = np.polyder(coeff)
            grad[i, it] = np.polyval(dcoeff, xcol[i])

    return grad


def _smooth_radial_profiles_axis0(y: np.ndarray, passes: int = 2) -> np.ndarray:
    """Apply binomial smoothing along the radial axis.

    Uses a 5-point kernel [1, 4, 6, 4, 1]/16 when possible, with a 3-point
    fallback for short radial grids.
    """
    if passes <= 0 or y.shape[0] < 3:
        return y

    out = np.array(y, dtype=float, copy=True)
    for _ in range(passes):
        if out.shape[0] >= 5:
            padded = np.pad(out, ((2, 2), (0, 0)), mode="edge")
            out = (
                padded[:-4, :]
                + 4.0 * padded[1:-3, :]
                + 6.0 * padded[2:-2, :]
                + 4.0 * padded[3:-1, :]
                + padded[4:, :]
            ) / 16.0
        else:
            padded = np.pad(out, ((1, 1), (0, 0)), mode="edge")
            out = 0.25 * padded[:-2, :] + 0.5 * padded[1:-1, :] + 0.25 * padded[2:, :]
    return out


def _smooth_profiles_for_gradients_axis0(
    y: np.ndarray,
    passes: int = 2,
    method: str = "spline",
    window_length: int = 9,
    polyorder: int = 3,
    spline_s_factor: float = 0.03,
) -> np.ndarray:
    """Smooth radial profiles with derivative-friendly filters.

    Preferred method is a smoothing spline fit to keep gradients physical
    without over-flattening profile structure.

    Supported methods: "spline", "savgol", "binomial".
    """
    if passes <= 0 or y.shape[0] < 3:
        return np.array(y, dtype=float, copy=True)

    y_in = np.array(y, dtype=float, copy=True)
    method_l = method.lower()

    if method_l == "binomial":
        return _smooth_radial_profiles_axis0(y_in, passes=passes)

    if method_l not in {"spline", "savgol"}:
        raise ValueError(
            f"Unsupported smoothing method '{method}'. Use 'spline', 'savgol', or 'binomial'."
        )

    try:
        from scipy.interpolate import UnivariateSpline
        from scipy.signal import savgol_filter
    except Exception:
        return _smooth_radial_profiles_axis0(y_in, passes=passes)

    out = np.array(y_in, dtype=float, copy=True)
    nrad = out.shape[0]
    x = np.arange(nrad, dtype=float)

    if method_l == "savgol":
        wl = min(window_length, nrad if nrad % 2 == 1 else nrad - 1)
        if wl < 3:
            return _smooth_radial_profiles_axis0(out, passes=passes)
        if wl <= polyorder:
            wl = polyorder + 2
            if wl % 2 == 0:
                wl += 1
            if wl > nrad:
                return _smooth_radial_profiles_axis0(out, passes=passes)

        for _ in range(passes):
            out = savgol_filter(out, window_length=wl, polyorder=polyorder, axis=0, mode="interp")
        return out

    # Smoothing-fit path (default): spline fit per time slice.
    for it in range(out.shape[1]):
        y_col = out[:, it]

        y_scale = np.nanmedian(np.abs(y_col - np.nanmedian(y_col)))
        if not np.isfinite(y_scale) or y_scale <= 0.0:
            y_scale = max(np.nanstd(y_col), 1e-9)
        s_val = max(float(spline_s_factor), 0.0) * nrad * (y_scale ** 2)

        try:
            spline = UnivariateSpline(x, y_col, k=3, s=s_val)
            out[:, it] = spline(x)
        except Exception:
            # Keep a stable fallback per-column if spline fit fails.
            out[:, it] = _smooth_radial_profiles_axis0(y_col[:, np.newaxis], passes=passes)[:, 0]

    return out


def _safe_log_gradient_sorted(y: np.ndarray, x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    order = np.argsort(x)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    xs = np.maximum(x[order], 1e-8)
    ys = np.maximum(y[order], eps)

    grad_sorted = np.gradient(ys, xs, edge_order=2)
    return grad_sorted[inv_order]


def _plot_compute_grid_diagnostics(
    time_s: np.ndarray,
    selected_times_s: list[float],
    psi_n_q: np.ndarray,
    psi_diag_mapped: np.ndarray,
    q_profile: np.ndarray,
    q95_t: np.ndarray,
    r_major_diag_m: np.ndarray,
    pe_diag_pa: np.ndarray,
    pe_diag_pa_smoothed: np.ndarray,
    pi_diag_pa: np.ndarray,
    pi_diag_pa_smoothed: np.ndarray,
    ne_diag_m3: np.ndarray,
    ne_diag_m3_smoothed: np.ndarray,
    ni_diag_m3: np.ndarray,
    ni_diag_m3_smoothed: np.ndarray,
    te_diag_eV: np.ndarray,
    te_diag_eV_smoothed: np.ndarray,
    ti_diag_eV: np.ndarray,
    ti_diag_eV_smoothed: np.ndarray,
    ti_ev_t: np.ndarray,
    rho_ti_t: np.ndarray,
    r_major_q_m: np.ndarray,
    r_minor_q_m: np.ndarray,
    shot: int | None = None,
) -> None:
    """Diagnostic figure for q(psi), pressure, mapping, and n/T on diag radial grid."""
    if len(selected_times_s) == 0:
        return

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8.5),
        num=(f"Drift Grid Diagnostics: shot {shot}" if shot is not None else "Drift Grid Diagnostics"),
    )
    ax_q = axes[0, 0]
    ax_p = axes[0, 1]
    ax_map = axes[0, 2]
    ax_ne = axes[1, 0]
    ax_ni = axes[1, 1]
    ax_empty = axes[1, 2]

    ax_te = ax_ne.twinx()
    ax_ti = ax_ni.twinx()
    ax_rminor = ax_empty

    cmap = plt.get_cmap("tab10")

    for i_sel, t_sel in enumerate(selected_times_s):
        it = int(np.argmin(np.abs(time_s - t_sel)))
        color = cmap(i_sel % 10)
        time_label = f"t={time_s[it]:.2f}s"

        order_q = np.argsort(psi_n_q[:, it])
        psi_sorted = psi_n_q[:, it][order_q]
        q_sorted = q_profile[:, it][order_q]
        r_q_sorted = r_major_q_m[:, it][order_q]
        r_minor_sorted = r_minor_q_m[:, it][order_q]

        order_r = np.argsort(r_major_diag_m[:, it])
        r_diag_sorted = r_major_diag_m[:, it][order_r]
        pe_sorted = pe_diag_pa[:, it][order_r]
        pe_sm_sorted = pe_diag_pa_smoothed[:, it][order_r]
        pi_sorted = pi_diag_pa[:, it][order_r]
        pi_sm_sorted = pi_diag_pa_smoothed[:, it][order_r]
        ne_sorted = ne_diag_m3[:, it][order_r]
        ne_sm_sorted = ne_diag_m3_smoothed[:, it][order_r]
        ni_sorted = ni_diag_m3[:, it][order_r]
        ni_sm_sorted = ni_diag_m3_smoothed[:, it][order_r]
        te_sorted = te_diag_eV[:, it][order_r]
        te_sm_sorted = te_diag_eV_smoothed[:, it][order_r]
        ti_sorted = ti_diag_eV[:, it][order_r]
        ti_sm_sorted = ti_diag_eV_smoothed[:, it][order_r]
        ti_ev_t_sorted = ti_ev_t[:, it]
        rho_ti_t_sorted = rho_ti_t[:, it]

        order_map = np.argsort(psi_diag_mapped[:, it])
        psi_diag_sorted = psi_diag_mapped[:, it][order_map]
        r_diag_map_sorted = r_major_diag_m[:, it][order_map]

        ax_q.plot(psi_sorted, q_sorted, color=color, lw=1.8, label=time_label)
        ax_q.axhline(float(q95_t[it]), color=color, ls=":", lw=1.2)

        ax_p.plot(r_diag_sorted, pe_sorted / 101325, color=color, lw=1.4, marker='*', alpha=0.65, label=rf"$p_e$ base ({time_label})")
        ax_p.plot(r_diag_sorted, pe_sm_sorted / 101325, color=color, lw=2.0, ls="-", label=rf"$p_e$ extra-sm ({time_label})")
        ax_p.plot(r_diag_sorted, pi_sorted / 101325, color=color, lw=1.4, ls="--", marker='*', alpha=0.65, label=rf"$p_i$ base ({time_label})")
        ax_p.plot(r_diag_sorted, pi_sm_sorted / 101325, color=color, lw=2.0, ls=":", label=rf"$p_i$ extra-sm ({time_label})")

        ax_map.plot(psi_sorted, r_q_sorted, color=color, lw=1.8, label=rf"eq grid ({time_label})")
        ax_map.plot(psi_diag_sorted, r_diag_map_sorted, color=color, lw=1.2, ls=":", label=rf"diag grid ({time_label})")

        ax_ne.plot(r_diag_sorted, ne_sorted * 1e-20, color=color, lw=1.2, alpha=0.55, label=rf"$n_e$ base ({time_label})")
        ax_ne.plot(r_diag_sorted, ne_sm_sorted * 1e-20, color=color, lw=2.0, ls="-", label=rf"$n_e$ extra-sm ({time_label})")
        ax_te.plot(r_diag_sorted, te_sorted * 1e-3, color=color, lw=1.2, ls="--", alpha=0.55, label=rf"$T_e$ base ({time_label})")
        ax_te.plot(r_diag_sorted, te_sm_sorted * 1e-3, color=color, lw=1.8, ls="-.", label=rf"$T_e$ extra-sm ({time_label})")

        ax_ni.plot(r_diag_sorted, ni_sorted * 1e-20, color=color, lw=1.2, alpha=0.55, label=rf"$n_i$ base ({time_label})")
        ax_ni.plot(r_diag_sorted, ni_sm_sorted * 1e-20, color=color, lw=2.0, ls="-", label=rf"$n_i$ extra-sm ({time_label})")
        ax_ti.plot(r_diag_sorted, ti_sorted * 1e-3, color=color, lw=1.2, ls="--", alpha=0.55, label=rf"$T_i$ base ({time_label})")
        ax_ti.plot(r_diag_sorted, ti_sm_sorted * 1e-3, color=color, lw=1.8, ls="-.", label=rf"$T_i$ extra-sm ({time_label})")
        ax_ti.plot( rho_ti_t_sorted, ti_ev_t_sorted * 1e-3, color=color, lw=1.4, ls=":", label=rf"$T_i,\,\rho$ ({time_label})")
        ax_rminor.plot(psi_sorted, r_minor_sorted, color=color, lw=1.8, label=time_label)

    ax_q.set_title(r"$q(\psi_N)$ and $q_{95}(t)$")
    ax_q.set_xlabel(r"$\psi_N$")
    ax_q.set_ylabel(r"$q$")
    ax_q.grid(alpha=0.3)
    ax_q.legend(fontsize=8, ncol=1)

    ax_p.set_title(r"Pressure on Original Radial Grid")
    ax_p.set_xlabel(r"$R_{\mathrm{major}}$ [m]")
    ax_p.set_ylabel(r"$p$ [atm]")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=7, ncol=1, handlelength=4)

    ax_map.set_title(r"Radial Mapping: $R(\psi_N)$")
    ax_map.set_xlabel(r"$\psi_N$")
    ax_map.set_ylabel(r"$R_{\mathrm{major}}$ [m]")
    ax_map.grid(alpha=0.3)
    ax_map.legend(fontsize=7, ncol=1)

    ax_ne.set_title(r"Electron $n_e, T_e$ on Diagnostic $R$")
    ax_ne.set_xlabel(r"$R_{\mathrm{major}}$ [m]")
    ax_ne.set_ylabel(r"$n_e$ [$10^{20}\,\mathrm{m}^{-3}$]")
    ax_te.set_ylabel(r"$T_e$ [keV]")
    ax_ne.grid(alpha=0.3)

    h1, l1 = ax_ne.get_legend_handles_labels()
    h2, l2 = ax_te.get_legend_handles_labels()
    ax_ne.legend(h1 + h2, l1 + l2, fontsize=7, ncol=1)

    ax_ni.set_title(r"Ion $n_i, T_i$ on Diagnostic $R$")
    ax_ni.set_xlabel(r"$R_{\mathrm{major}}$ [m]")
    ax_ni.set_ylabel(r"$n_i$ [$10^{20}\,\mathrm{m}^{-3}$]")
    ax_ti.set_ylabel(r"$T_i$ [keV]")
    ax_ni.grid(alpha=0.3)

    h3, l3 = ax_ni.get_legend_handles_labels()
    h4, l4 = ax_ti.get_legend_handles_labels()
    ax_ni.legend(h3 + h4, l3 + l4, fontsize=7, ncol=1)

    ax_rminor.set_title(r"Minor Radius vs $\psi_N$")
    ax_rminor.set_xlabel(r"$\psi_N$")
    ax_rminor.set_ylabel(r"$r_{\mathrm{minor}}$ [m]")
    ax_rminor.grid(alpha=0.3)
    ax_rminor.legend(fontsize=8, ncol=1)

    fig.tight_layout()
    plt.show(block=False)


def compute_diamagnetic_drift_frequencies(
    profiles: ProfileData,
    equilibrium: EquilibriumData,
    ion_charge_state: float = 1.0,
    selected_times_s: list[float] | None = None,
    shot: int | None = None,
    do_diagnostic_plot: bool = True,
    smooth_profiles_before_pressure: bool = False,
    profile_smoothing_passes: int = 2,
    profile_smoothing_method: str = "spline",
    pressure_smoothing_passes: int = 2,
    pressure_gradient_method: str = "polyfit",
    pressure_gradient_fit_points: int = 5,
    pressure_gradient_poly_order: int = 3,
) -> DiamagneticResult:
    """
        Compute diamagnetic drift frequencies from pressure-gradient drift velocity
        and flux-surface circumference on the equilibrium psi grid.

            v_*s = (1 / (q_s n_s B^2)) * d(p_s)/dr x B
            f_*s = v_*s / C(psi, t)
            omega_*s = 2*pi*f_*s

        Electron density and temperature are interpolated from the diagnostic major-radius
        grid onto the equilibrium major-radius grid. Ion temperature is interpolated from
        the HIREXSR psi grid onto the same equilibrium psi grid.
    """
    t_diag = np.asarray(profiles.time_diag, dtype=float).squeeze()
    nt_diag = t_diag.size
    has_hx_grid = profiles.psi_hx is not None

    r_major_diag_m = np.asarray(profiles.r_major_diag_m, dtype=float)
    ne_m3 = np.asarray(profiles.ne_m3, dtype=float)
    te_eV = np.asarray(profiles.te_eV, dtype=float)
    ti_eV = np.asarray(profiles.ti_eV, dtype=float)
    if has_hx_grid:
        psi_hx_src = profiles.psi_hx
        if psi_hx_src is None:
            raise ValueError("psi_hx unexpectedly None while has_hx_grid=True")
        psi_hx = np.asarray(psi_hx_src, dtype=float)
    else:
        psi_hx = r_major_diag_m

    zeff_raw = np.asarray(profiles.zeff, dtype=float).squeeze()
    if zeff_raw.ndim == 0:
        zeff_t_diag = np.full(nt_diag, float(zeff_raw))
    else:
        zeff_t_diag = zeff_raw

    if ion_charge_state <= 0.0:
        raise ValueError("ion_charge_state must be positive.")


    t_eq = np.asarray(equilibrium.time_s, dtype=float).squeeze()
    psi_n_q = np.asarray(equilibrium.psi_n_q, dtype=float)
    q_profile = np.asarray(equilibrium.q_profile, dtype=float)
    q95_raw = np.asarray(equilibrium.q95, dtype=float).squeeze()
    if q95_raw.ndim == 0:
        q95_t = np.full(t_eq.size, float(q95_raw))
    else:
        q95_t = q95_raw
    q95_t = np.maximum(q95_t, 0.0)

    b_t_rt = np.asarray(equilibrium.b_t_t, dtype=float)
    r_major_full_m = np.asarray(equilibrium.r_major_full_m, dtype=float).squeeze()
    r_minor_q_t_eq = np.asarray(equilibrium.r_minor_q_m, dtype=float)
    r_major_q_t_eq = np.asarray(equilibrium.r_major_q_m, dtype=float)
    circumference_q_t_eq = np.asarray(equilibrium.circumference_q_m, dtype=float)
    b_pol_q_t_eq = np.asarray(equilibrium.b_pol_q_T, dtype=float)

    # Use equilibrium time as the master output timebase, and map diagnostics onto it.
    t = t_eq
    nt = t.size

    r_major_diag_t = _interp_space_time_in_time(r_major_diag_m, t_diag, t)
    ne_m3_t = _interp_space_time_in_time(ne_m3, t_diag, t)
    te_eV_t = _interp_space_time_in_time(te_eV, t_diag, t)
    ti_eV_t = _interp_space_time_in_time(ti_eV, t_diag, t)
    rho_ti_t = (
        _interp_space_time_in_time(np.asarray(profiles.rho_hx, dtype=float), t_diag, t)
        if profiles.rho_hx is not None
        else None
    )
    psi_hx_t = _interp_space_time_in_time(psi_hx, t_diag, t) if psi_hx is not None else r_major_diag_t
    zeff_t = np.interp(t, t_diag, zeff_t_diag)

    psi_n_q_t = psi_n_q
    q_profile_t = q_profile
    r_minor_q_t = r_minor_q_t_eq
    r_major_q_t = r_major_q_t_eq
    circumference_q_t = circumference_q_t_eq
    b_pol_q_t = b_pol_q_t_eq
    r_major_full_t = np.repeat(r_major_full_m[:, np.newaxis], nt, axis=1)
    b_t_q_t = _interp_columns_to_grid(b_t_rt, r_major_full_t, r_major_q_t)

    ne_q = _interp_columns_to_grid(ne_m3_t, r_major_diag_t, r_major_q_t)
    te_q_eV = _interp_columns_to_grid(te_eV_t, r_major_diag_t, r_major_q_t)
    psi_diag_mapped = _interp_columns_to_grid(psi_n_q_t, r_major_q_t, r_major_diag_t)
    if has_hx_grid:
        ti_q_eV = _interp_columns_to_grid(ti_eV_t, psi_hx_t, psi_n_q_t)
        if rho_ti_t is None:
            rho_ti_t = psi_hx_t
        ti_diag_eV_t = _map_ti_psi_to_diagnostic_radius_grid(
            ti_eV_t=ti_eV_t,
            rho_hx_t=rho_ti_t,
            psi_hx_t=psi_hx_t,
            psi_n_q_t=psi_n_q_t,
            r_major_q_t=r_major_q_t,
            r_major_diag_t=r_major_diag_t,
        )
    else:
        ti_q_eV = _interp_columns_to_grid(ti_eV_t, r_major_diag_t, r_major_q_t)
        ti_diag_eV_t = ti_eV_t
        if rho_ti_t is None:
            rho_ti_t = psi_diag_mapped

    ne_q_raw = ne_q
    te_q_eV_raw = te_q_eV
    ti_q_eV_raw = ti_q_eV
    ni_q_raw = ne_q_raw / np.maximum(zeff_t[np.newaxis, :], 1e-8)

    if smooth_profiles_before_pressure:
        ne_q = _smooth_profiles_for_gradients_axis0(
            ne_q_raw,
            passes=profile_smoothing_passes,
            method=profile_smoothing_method,
        )
        te_q_eV = _smooth_profiles_for_gradients_axis0(
            te_q_eV_raw,
            passes=profile_smoothing_passes,
            method=profile_smoothing_method,
        )
        ti_q_eV = _smooth_profiles_for_gradients_axis0(
            ti_q_eV_raw,
            passes=profile_smoothing_passes,
            method=profile_smoothing_method,
        )
    else:
        ne_q = np.array(ne_q_raw, dtype=float, copy=True)
        te_q_eV = np.array(te_q_eV_raw, dtype=float, copy=True)
        ti_q_eV = np.array(ti_q_eV_raw, dtype=float, copy=True)

    ni_q = ne_q / np.maximum(zeff_t[np.newaxis, :], 1e-8)

    # Build diagnostic-grid quantities for plotting and gradient computation.
    ni_diag_t = ne_m3_t / np.maximum(zeff_t[np.newaxis, :], 1e-8)
    ne_diag_sm = _smooth_profiles_for_gradients_axis0(
        ne_m3_t,
        passes=profile_smoothing_passes,
        method=profile_smoothing_method,
    ) if smooth_profiles_before_pressure else np.array(ne_m3_t, dtype=float, copy=True)
    te_diag_sm = _smooth_profiles_for_gradients_axis0(
        te_eV_t,
        passes=profile_smoothing_passes,
        method=profile_smoothing_method,
    ) if smooth_profiles_before_pressure else np.array(te_eV_t, dtype=float, copy=True)
    ti_diag_sm = _smooth_profiles_for_gradients_axis0(
        ti_diag_eV_t,
        passes=profile_smoothing_passes,
        method=profile_smoothing_method,
    ) if smooth_profiles_before_pressure else np.array(ti_diag_eV_t, dtype=float, copy=True)
    ni_diag_sm = ne_diag_sm / np.maximum(zeff_t[np.newaxis, :], 1e-8)

    pe_diag_pa = ne_m3_t * (te_eV_t * E_CHARGE)
    pi_diag_pa = ni_diag_t * (ti_diag_eV_t * E_CHARGE)
    pe_diag_pa_sm = ne_diag_sm * (te_diag_sm * E_CHARGE)
    pi_diag_pa_sm = ni_diag_sm * (ti_diag_sm * E_CHARGE)

    q_e = -E_CHARGE
    q_i = ion_charge_state * E_CHARGE
    grad_method = pressure_gradient_method.lower()

    # Compute dp/dR on the diagnostic major-radius grid where profiles are naturally
    # spaced and well-conditioned. Since r_minor = R_major - R_0 (circular approximation),
    # dp/dr_minor = dp/dR_major exactly. The resulting gradient is then interpolated onto
    # the equilibrium q-grid, avoiding compression artefacts near the magnetic axis.
    # Use base (load-cleaned) pressure traces for gradients; extra-sm traces are for
    # diagnostic comparison only.
    if grad_method == "polyfit":
        dpe_dR_diag = _gradient_axis0_nonuniform_polyfit(
            pe_diag_pa,
            r_major_diag_t,
            fit_points=pressure_gradient_fit_points,
            poly_order=pressure_gradient_poly_order,
        )
        dpi_dR_diag = _gradient_axis0_nonuniform_polyfit(
            pi_diag_pa,
            r_major_diag_t,
            fit_points=pressure_gradient_fit_points,
            poly_order=pressure_gradient_poly_order,
        )
    elif grad_method == "finite-diff":
        dpe_dR_diag = _gradient_axis0_nonuniform(pe_diag_pa, r_major_diag_t)
        dpi_dR_diag = _gradient_axis0_nonuniform(pi_diag_pa, r_major_diag_t)
    else:
        raise ValueError(
            f"Unsupported pressure_gradient_method '{pressure_gradient_method}'. "
            "Use 'polyfit' or 'finite-diff'."
        )

    # Interpolate dp/dR from the diagnostic grid onto the equilibrium q-grid.
    dpe_dr = _interp_columns_to_grid(dpe_dR_diag, r_major_diag_t, r_major_q_t)
    dpi_dr = _interp_columns_to_grid(dpi_dR_diag, r_major_diag_t, r_major_q_t)

    # Use v_* = (1/(q n B^2)) (grad p x B), resolved into component magnitudes.
    # For radial grad p and B = B_t e_tor + B_p e_pol:
    #   |v_*pol| ~ (dp/dr) * B_t / (q n B^2)
    #   |v_*tor| ~ (dp/dr) * B_p / (q n B^2)
    b_tot_sq = np.maximum(b_t_q_t * b_t_q_t + b_pol_q_t * b_pol_q_t, 1e-12)
    v_star_e = dpe_dr * b_t_q_t / (q_e * np.maximum(ne_q, 1e-12) * b_tot_sq)
    v_star_i = dpi_dr * b_t_q_t / (q_i * np.maximum(ni_q, 1e-12) * b_tot_sq)

    valid_circumference = circumference_q_t >= 0.10
    f_star_e = np.where(valid_circumference, v_star_e / circumference_q_t, np.nan)
    f_star_i = np.where(valid_circumference, v_star_i / circumference_q_t, np.nan)
    omega_star_e = TWO_PI * f_star_e
    omega_star_i = TWO_PI * f_star_i

    # Toroidal diamagnetic drift component from the same B/B^2 projection.
    v_star_e_tor = dpe_dr * b_pol_q_t / (q_e * np.maximum(ne_q, 1e-12) * b_tot_sq)
    v_star_i_tor = dpi_dr * b_pol_q_t / (q_i * np.maximum(ni_q, 1e-12) * b_tot_sq)
    toroidal_circ = TWO_PI * r_major_q_t  # [nq, nt]
    f_star_e_tor = v_star_e_tor / toroidal_circ
    f_star_i_tor = v_star_i_tor / toroidal_circ
    omega_star_e_tor = TWO_PI * f_star_e_tor
    omega_star_i_tor = TWO_PI * f_star_i_tor

    # Restrict computed drifts to q surfaces inside q95 at each time.
    q_valid = (q_profile_t > 0.0) & (q_profile_t <= q95_t[np.newaxis, :])
    f_star_e = np.where(q_valid, f_star_e, np.nan)
    f_star_i = np.where(q_valid, f_star_i, np.nan)
    omega_star_e = np.where(q_valid, omega_star_e, np.nan)
    omega_star_i = np.where(q_valid, omega_star_i, np.nan)
    dpe_dr = np.where(q_valid, dpe_dr, np.nan)
    dpi_dr = np.where(q_valid, dpi_dr, np.nan)
    q_profile_masked = np.where(q_valid, q_profile_t, np.nan)
    circumference_masked = np.where(q_valid, circumference_q_t, np.nan)
    f_star_e_tor = np.where(q_valid, f_star_e_tor, np.nan)
    f_star_i_tor = np.where(q_valid, f_star_i_tor, np.nan)
    omega_star_e_tor = np.where(q_valid, omega_star_e_tor, np.nan)
    omega_star_i_tor = np.where(q_valid, omega_star_i_tor, np.nan)

    if do_diagnostic_plot and selected_times_s is not None:
        _plot_compute_grid_diagnostics(
            time_s=t,
            selected_times_s=selected_times_s,
            psi_n_q=psi_n_q_t,
            psi_diag_mapped=psi_diag_mapped,
            q_profile=q_profile_t,
            q95_t=q95_t,
            r_major_diag_m=r_major_diag_t,
            pe_diag_pa=pe_diag_pa,
            pe_diag_pa_smoothed=pe_diag_pa_sm,
            pi_diag_pa=pi_diag_pa,
            pi_diag_pa_smoothed=pi_diag_pa_sm,
            ne_diag_m3=ne_m3_t,
            ne_diag_m3_smoothed=ne_diag_sm,
            ni_diag_m3=ni_diag_t,
            ni_diag_m3_smoothed=ni_diag_sm,
            te_diag_eV=te_eV_t,
            te_diag_eV_smoothed=te_diag_sm,
            ti_ev_t=ti_eV_t,
            rho_ti_t=rho_ti_t,
            ti_diag_eV=ti_diag_eV_t,
            ti_diag_eV_smoothed=ti_diag_sm,
            r_major_q_m=r_major_q_t,
            r_minor_q_m=r_minor_q_t,
            shot=shot,
        )

    return DiamagneticResult(
        time_s=t,
        psi_n=psi_n_q_t,
        q=q_profile_masked,
        circumference_m=circumference_masked,
        dpe_dr_pa_m=dpe_dr,
        dpi_dr_pa_m=dpi_dr,
        ne_m3=ne_q,
        ni_m3=ni_q,
        te_eV=te_q_eV,
        ti_eV=ti_q_eV,
        f_star_e_Hz=f_star_e,
        f_star_i_Hz=f_star_i,
        omega_star_e_rad_s=omega_star_e,
        omega_star_i_rad_s=omega_star_i,
        f_star_e_tor_Hz=f_star_e_tor,
        f_star_i_tor_Hz=f_star_i_tor,
        omega_star_e_tor_rad_s=omega_star_e_tor,
        omega_star_i_tor_rad_s=omega_star_i_tor,
    )


def _add_density_temperature_panel(
    ax_density,
    ax_temp,
    psi_n: np.ndarray,
    density_m3: np.ndarray,
    temp_eV: np.ndarray,
    label_prefix: str,
    color,
    time_label: str,
) -> None:
    order = np.argsort(psi_n)
    x = psi_n[order]

    density_plot = density_m3[order] * 1e-20
    temp_plot = temp_eV[order] * 1e-3

    ax_density.plot(x, density_plot, color=color, lw=1.8, ls='-', marker='*', label=f"$n_{label_prefix}$ ({time_label})")
    ax_temp.plot(x, temp_plot, color=color, lw=1.8, ls="--", marker='*', label=f"$T_{label_prefix}$ ({time_label})")


def plot_diamagnetic_vs_q_times(
    result: DiamagneticResult,
    profiles: ProfileData,
    equilibrium: EquilibriumData,
    shot: int,
    selected_times_s: list[float],
    doSave: str = '',
    f_lims_omega: list[float] = [-50, 50],
    f_lims_f: list[float] = [-15, 15],
    f_lims_f_tor: list[float] = [-1, 1],
    line: int = 2,
    tht: int = 0,
    max_err_omega: float = 10.0,
) -> None:
    """Production summary plot: pressure vs psi_n, combined drift frequencies vs q,
    and toroidal rotation (omega_tor) with error bars vs psi_n.

    Layout: 1 row x 3 columns.
    - Left:   Smoothed electron and ion pressure profiles vs normalised poloidal flux.
    - Centre: Electron and ion diamagnetic drift frequencies on the same axes vs q.
    - Right:  HIREXSR toroidal rotation with 1-sigma error bars vs psi_n,
              with q(psi_n) overlaid on the right-hand y-axis (cut at q_95).
    """
    """Production summary plot: pressure vs psi_n, combined poloidal and toroidal
    drift frequencies vs q, and toroidal rotation (omega_tor) with error bars vs psi_n.

    Layout: 1 row x 4 columns.
    - Panel 1: Smoothed electron and ion pressure profiles vs normalised poloidal flux.
    - Panel 2: Poloidal diamagnetic drift frequencies (uses B_t) vs q.
    - Panel 3: Toroidal diamagnetic drift frequencies (uses B_pol) vs q.
    - Panel 4: HIREXSR toroidal rotation with 1-sigma error bars vs psi_n,
               with q(psi_n) overlaid on the right-hand y-axis (cut at q_95).
    """
    if len(selected_times_s) == 0:
        raise ValueError("selected_times_s must contain at least one time.")

    t_res = result.time_s
    t_diag = np.asarray(profiles.time_diag, dtype=float).squeeze()
    t_eq = np.asarray(equilibrium.time_s, dtype=float).squeeze()

    fig, axes = plt.subplots(
        1, 4,
        figsize=(20, 5),
        num=f"Diamagnetic Drift Summary: shot {shot}",
    )
    ax_p = axes[0]
    ax_f = axes[1]
    ax_f_tor = axes[2]
    ax_omg = axes[3]
    ax_q_omg = ax_omg.twinx()  # right-hand q(psi_n) axis on omega panel

    cmap = plt.get_cmap("tab10")

    has_omega = (
        profiles.omega_tor_rad_s is not None
        and profiles.omega_tor_err_rad_s is not None
        and profiles.psi_hx is not None
    )

    for i_sel, t_sel in enumerate(selected_times_s):
        it_res = int(np.argmin(np.abs(t_res - t_sel)))
        it_diag = int(np.argmin(np.abs(t_diag - t_sel)))
        color = cmap(i_sel % 10)
        time_label = f"t={t_res[it_res]:.2f} s"

        # --- Pressure on psi_n grid (q-grid has psi_n) ---
        psi_col = result.psi_n[:, it_res]
        pe_col = result.ne_m3[:, it_res] * result.te_eV[:, it_res] * E_CHARGE
        pi_col = result.ni_m3[:, it_res] * result.ti_eV[:, it_res] * E_CHARGE
        order_psi = np.argsort(psi_col)
        psi_s = psi_col[order_psi]
        ax_p.plot(
            psi_s, pe_col[order_psi] / 101325, # Convert Pa to atm for plotting
            color=color, lw=2.0, ls="-",
            label=rf"$p_e$ ({time_label})",
        )
        ax_p.plot(
            psi_s, pi_col[order_psi] / 101325,
            color=color, lw=2.0, ls="--",
            label=rf"$p_i$ ({time_label})",
        )

        # --- Diamagnetic frequencies on q grid ---
        q_col = result.q[:, it_res]
        order_q = np.argsort(q_col)
        q_s = q_col[order_q]
        fe_khz = result.f_star_e_Hz[:, it_res][order_q] / 1e3
        fi_khz = result.f_star_i_Hz[:, it_res][order_q] / 1e3
        ax_f.plot(
            q_s, fe_khz,
            color=color, lw=2.0, ls="-",
            label=rf"$f_{{*e}}$ ({time_label})",
        )
        ax_f.plot(
            q_s, fi_khz,
            color=color, lw=2.0, ls="--",
            label=rf"$f_{{*i}}$ ({time_label})",
        )
        if f_lims_f:
            ax_f.set_ylim(f_lims_f)

        # --- Toroidal diamagnetic frequencies (B_pol) on q grid ---
        fe_tor_khz = result.f_star_e_tor_Hz[:, it_res][order_q] / 1e3
        fi_tor_khz = result.f_star_i_tor_Hz[:, it_res][order_q] / 1e3
        ax_f_tor.plot(
            q_s, fe_tor_khz,
            color=color, lw=2.0, ls="-",
            label=rf"$f_{{*e}}^{{\mathrm{{tor}}}}$ ({time_label})",
        )
        ax_f_tor.plot(
            q_s, fi_tor_khz,
            color=color, lw=2.0, ls="--",
            label=rf"$f_{{*i}}^{{\mathrm{{tor}}}}$ ({time_label})",
        )
        if f_lims_f_tor:
            ax_f_tor.set_ylim(f_lims_f_tor)

        # --- Toroidal rotation with error bars on psi_hx grid ---
        if has_omega:
            psi_hx_arr = profiles.psi_hx
            omg_arr = profiles.omega_tor_rad_s
            omg_err_arr = profiles.omega_tor_err_rad_s
            if psi_hx_arr is None or omg_arr is None or omg_err_arr is None:
                continue
            psi_hx_col = np.asarray(psi_hx_arr[:-1, it_diag], dtype=float) # Last point is artificial Psi=1.0 boundary
            omg_col = np.asarray(omg_arr[:, it_diag], dtype=float)
            omg_err_col = np.asarray(omg_err_arr[:, it_diag], dtype=float)
            order_hx = np.argsort(psi_hx_col)
            psi_hx_s = psi_hx_col[order_hx]
            omg_s = omg_col[order_hx] # kHz
            omg_err_s = omg_err_col[order_hx] 

            # Filter out points with errors exceeding the maximum allowed value
            valid_mask = omg_err_s <= max_err_omega
            psi_hx_s = psi_hx_s[valid_mask]
            omg_s = omg_s[valid_mask]
            omg_err_s = omg_err_s[valid_mask]
            ax_omg.errorbar(
                psi_hx_s, omg_s,
                yerr=omg_err_s,
                color=color, lw=1.8,
                elinewidth=0.9, capsize=3,
                label=rf"$\omega_{{\phi}}$ ({time_label})",
            )
            ax_omg.set_ylim(f_lims_omega)

        # --- q(psi_n) on right-hand axis, cut at q_95 ---
        it_eq = int(np.argmin(np.abs(t_eq - t_sel)))
        q95_val = float(np.asarray(equilibrium.q95, dtype=float).squeeze().flat[it_eq])
        psi_eq_col = np.asarray(equilibrium.psi_n_q[:, it_eq], dtype=float)
        q_eq_col = np.asarray(equilibrium.q_profile[:, it_eq], dtype=float)
        order_eq = np.argsort(psi_eq_col)
        psi_eq_s = psi_eq_col[order_eq]
        q_eq_s = q_eq_col[order_eq]
        mask_q95 = q_eq_s <= q95_val
        ax_q_omg.plot(
            psi_eq_s[mask_q95], q_eq_s[mask_q95],
            color=color, lw=1.4, ls=":",
            label=rf"$q$ ({time_label})",
        )

    ax_p.set_title(r"Pressure Profiles")
    ax_p.set_xlabel(r"$\psi_N$")
    ax_p.set_ylabel(r"$p$ [atm]")
    ax_p.axhline(0.0, color="k", ls=":", lw=0.8)
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=9, ncol=1)

    ax_f.set_title(r"Diamagnetic Drift Frequencies")
    ax_f.set_xlabel(r"$q(\psi_N,\,t)$")
    ax_f.set_ylabel(r"$f_*$ [kHz]")
    ax_f.axhline(0.0, color="k", ls="--", lw=1)
    ax_f.grid(alpha=0.3)
    ax_f.legend(fontsize=9, ncol=1)

    ax_f_tor.set_title(r"Toroidal Diamagnetic Drift Frequencies ($B_p$)")
    ax_f_tor.set_xlabel(r"$q(\psi_N,\,t)$")
    ax_f_tor.set_ylabel(r"$f_*^{\mathrm{tor}}$ [kHz]")
    ax_f_tor.axhline(0.0, color="k", ls="--", lw=1)
    ax_f_tor.grid(alpha=0.3)
    ax_f_tor.legend(fontsize=9, ncol=1)

    if has_omega:
        ax_omg.set_title(r"Toroidal Rotation (HIREXSR) \& $q(\psi_N)$")
        ax_omg.set_xlabel(r"$\psi_N$")
        ax_omg.set_ylabel(r"$\omega_\phi$ [kHz]")
        ax_q_omg.set_ylabel(r"$q$")
        ax_omg.axhline(0.0, color="k", ls=":", lw=0.8)
        ax_omg.grid(alpha=0.3)
        h1, l1 = ax_omg.get_legend_handles_labels()
        h2, l2 = ax_q_omg.get_legend_handles_labels()
        ax_omg.legend(h1 + h2, l1 + l2, fontsize=9, ncol=1)
    else:
        ax_omg.set_visible(False)

    fig.suptitle(f"Shot {shot}: diamagnetic drift and toroidal rotation")
    fig.tight_layout()
    

    if doSave:
        save_path = doSave + f"diamagnetic_drift_shot_{shot}_tht_{tht}_line_{line}.pdf"
        fig.savefig(save_path, transparent=True)
        print(f"Saved figure to {save_path}")

    plt.show(block=True)

def _demo_profiles() -> ProfileData:
    # Artificial timebase: 0.0 to 1.0 s in 0.1 s increments.
    time_s = np.arange(0.0, 1.01, 0.1)

    nspace = 96
    psi_base = np.linspace(0.02, 0.98, nspace)[:, np.newaxis]
    tau = (time_s / time_s.max())[np.newaxis, :]

    # Mild time-dependent breathing in psi_n profile.
    psi_n = psi_base + 0.015 * np.sin(2.0 * np.pi * tau) * (psi_base - 0.5)
    psi_n = np.clip(psi_n, 0.0, 1.0)

    r_minor_m = np.clip(C_MOD_A_MINOR_M * psi_n, 1e-3, C_MOD_A_MINOR_M)
    r_major_diag_m = C_MOD_R0_M + r_minor_m

    ne0 = 2.8e20
    te0 = 1200.0
    ti0 = 900.0

    time_mod_n = 1.0 + 0.10 * np.sin(2.0 * np.pi * tau)
    time_mod_t = 1.0 + 0.12 * np.cos(2.0 * np.pi * tau)

    ne_m3 = (ne0 * (1.0 - 0.88 * psi_n**1.4) + 3.0e19) * time_mod_n
    te_eV = (te0 * (1.0 - 0.92 * psi_n**1.6) + 60.0) * time_mod_t
    ti_eV = (ti0 * (1.0 - 0.85 * psi_n**1.4) + 45.0) * (
        1.0 + 0.08 * np.sin(2.0 * np.pi * tau + 0.7)
    )

    zeff = 1.7 + 0.15 * np.sin(2.0 * np.pi * time_s)

    return ProfileData(
        time_diag=time_s,
        r_major_diag_m=r_major_diag_m,
        ne_m3=ne_m3,
        te_eV=te_eV,
        ti_eV=ti_eV,
        zeff=zeff,
        psi_hx=psi_n,
        rho_hx=None,
        omega_tor_rad_s=None,
        omega_tor_err_rad_s=None,
    )


def _demo_equilibrium(time_s: np.ndarray) -> EquilibriumData:
    nq = 128

    psi_q_base = np.linspace(0.0, 1.0, nq)[:, np.newaxis]
    tau = (time_s / max(time_s.max(), 1e-6))[np.newaxis, :]

    psi_n_q = psi_q_base + 0.01 * np.cos(2.0 * np.pi * tau) * (psi_q_base - 0.5)
    psi_n_q = np.clip(psi_n_q, 0.0, 1.0)

    r_minor_q_m = np.clip(C_MOD_A_MINOR_M * psi_n_q, 1e-3, C_MOD_A_MINOR_M)
    elongation_t = np.full(time_s.size, 1.6, dtype=float)
    circumference_q_m = _ellipse_circumference(r_minor_q_m, elongation_t[np.newaxis, :])


    q_profile = 0.9 + 2.7 * psi_n_q**1.8 + 0.08 * np.sin(2.0 * np.pi * tau)
    b_t_time = 5.4 + 0.15 * np.cos(2.0 * np.pi * time_s)
    r_major_full_m = np.linspace(C_MOD_R0_M - C_MOD_A_MINOR_M, C_MOD_R0_M + C_MOD_A_MINOR_M, 196)
    b_t_t = b_t_time[np.newaxis, :] * (C_MOD_R0_M / np.maximum(r_major_full_m[:, np.newaxis], 1e-6))
    q95 = np.full(time_s.size, 3.2, dtype=float)

    return EquilibriumData(
        time_s=time_s,
        psi_n_q=psi_n_q,
        q_profile=q_profile,
        b_t_t=b_t_t,
        r_major_full_m=r_major_full_m,
        r_minor_q_m=r_minor_q_m,
        r_major_q_m=C_MOD_R0_M + r_minor_q_m,
        elongation_t=elongation_t,
        circumference_q_m=circumference_q_m,
        q95=q95,
        b_pol_q_T=_demo_b_pol(r_minor_q_m, C_MOD_R0_M + r_minor_q_m, b_t_time[np.newaxis, :], q_profile),
    )


def _demo_b_pol(
    r_minor_q_m: np.ndarray,
    r_major_q_m: np.ndarray,
    b_t_t: np.ndarray,
    q_profile: np.ndarray,
) -> np.ndarray:
    """Synthetic poloidal field from q = r * B_t / (R * B_p)  =>  B_p = r * B_t / (R * q)."""
    q_safe = np.maximum(np.abs(q_profile), 0.1)
    return r_minor_q_m * b_t_t / (r_major_q_m * q_safe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute time-dependent electron/ion diamagnetic frequencies and plot selected time slices versus q."
    )
    #1120906030
    # 1110316031
    parser.add_argument("--shot", type=int, default=1120906030, help="Shot number")
    parser.add_argument("--tree", type=str, default="efit20", help="EFIT tree name")
    parser.add_argument(
        "--zi",
        type=float,
        default=1.0,
        help="Ion charge state Z_i for q_i = Z_i e",
    )
    
    # Note: This argument comes into the command line as --plot times 0.6 1 1.3 (space-separated list of times), which is parsed into a list of floats.
    parser.add_argument(
        "--plot-times",
        nargs="+",
        type=float,
        default=[0.6, 1, 1.3],
        help="Selected times [s] to plot (nearest available times are used)",
    )
    parser.add_argument(
        "--demo",
        default=False,
        action="store_true",
        help="Use fully synthetic time-dependent demo data (0.0:0.1:1.0 s)",
    )

    parser.add_argument(
        "--f_lims_omega",
        nargs=2,
        type=float,
        default=[-50, 50],
        help="Y-axis limits for omega_* plot (kHz)",
    )

    parser.add_argument(
        "--doSave",
        default='',
        type=str,
        help="Save plots to file path",
    )

    parser.add_argument(
        "--tht",
        type=int,
        default=0,
        help = "THT branch number for HIREXSR MHD+ Tree"
    )

    parser.add_argument(
        "--line",
        type=int,
        default=2,
        help = "Diagnostic line name for HIREXSR MHD+ Tree (e.g. 'core', 'edge')"
    )

    parser.add_argument(
        "--max-omega-err",
        type=float,
        default=10.0,
        help="Maximum allowed error in omega_* [kHz]"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        profiles = _demo_profiles()
        equilibrium = _demo_equilibrium(profiles.time_diag)
    else:
        profiles = load_profiles_for_shot(args.shot, line = args.line, tht=args.tht)
        equilibrium = load_equilibrium_for_shot(args.shot, tree=args.tree)

    result = compute_diamagnetic_drift_frequencies(
        profiles=profiles,
        equilibrium=equilibrium,
        ion_charge_state=args.zi,
        selected_times_s=args.plot_times,
        shot=args.shot,
        do_diagnostic_plot=True,
    )

    plot_diamagnetic_vs_q_times(result, profiles, equilibrium, args.shot, args.plot_times, \
                                doSave=args.doSave, f_lims_omega=args.f_lims_omega, \
                                    tht=args.tht, line=args.line, max_err_omega=args.max_omega_err)


if __name__ == "__main__":
    main()
    print("Done")
