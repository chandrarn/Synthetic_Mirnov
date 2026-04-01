"""Python rewrite of IDL `HIREXSR_GET_PROFILE` from `hirexsr_load_data.pro`.

This loader returns inverted profile data from the spectroscopy tree and maps
its radial coordinate into projected major radius using EFIT.

Original IDL code and documentation:
/usr/local/cmod/idl/HIREXSR/hirexsr_load_data.pro
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Sequence

import numpy as np

from hirexsr_lint_profile_py import openTree, multi_interpol
from hirexsr_lint_profile_py import hirexsr_get_lint_profile_py
from hirexsr_lint_profile_py import _line_config
from hirexsr_load_result_py import _ensure_profile_cube
from hirexsr_plotting_py import _plot_inversion_profile, _plot_lint_profile, _plot_profile_vs_lint


@dataclass
class InversionProfileResult:
    shot: int
    lineid: str
    time: np.ndarray
    psi: np.ndarray
    r_maj: np.ndarray
    r_ave: np.ndarray
    emiss: np.ndarray
    emisserr: np.ndarray
    ti: np.ndarray
    tierr: np.ndarray
    omg: np.ndarray
    omgerr: np.ndarray
    rot: np.ndarray
    roterr: np.ndarray
    has_m1: bool
    emissm1: np.ndarray | None = None
    emisserrm1: np.ndarray | None = None
    tim1: np.ndarray | None = None
    tierrm1: np.ndarray | None = None
    omgm1: np.ndarray | None = None
    omgerrm1: np.ndarray | None = None
    rotm1: np.ndarray | None = None
    roterrm1: np.ndarray | None = None


def _analysis_initstring(tht: int | None) -> str:
    if tht is None or tht == 0:
        return r"\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS."
    return rf"\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS{int(tht)}."


def _line_from_index(line: int, tht: int) -> tuple[str, str]:
    """Map lint-style integer line index to profile tree node and compact line id."""
    cfg = _line_config(line, tht)
    path = str(cfg["path"]).upper()

    if ".HELIKE." in path:
        family = "HELIKE"
    elif ".HLIKE." in path:
        family = "HLIKE"
    else:
        raise ValueError(f"Could not infer line family from path: {cfg['path']}")

    if ".MOMENTS." not in path:
        raise ValueError(f"Unexpected line path format: {cfg['path']}")

    line_name = path.split(".MOMENTS.")[-1].rstrip(":")
    line_node = f"{family}.PROFILES.{line_name}"
    lineid = line_name.lower()
    return line_node, lineid


def _first_index_gt(vals: np.ndarray, threshold: float) -> int | None:
    idx = np.where(vals > threshold)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def _slice_seltime(
    emiss: np.ndarray,
    emisserr: np.ndarray,
    omg: np.ndarray,
    omgerr: np.ndarray,
    ti: np.ndarray,
    tierr: np.ndarray,
    time: np.ndarray,
    psi: np.ndarray,
    seltime: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if seltime is None:
        return emiss, emisserr, omg, omgerr, ti, tierr, time, psi

    if len(seltime) != 2:
        raise ValueError("seltime must have exactly two values: [time_low, time_high]")

    time_low = float(seltime[0])
    time_high = float(seltime[1])
    ilow = _first_index_gt(time, time_low)
    ihigh = _first_index_gt(time, time_high)

    if ilow is None or ihigh is None:
        raise ValueError("seltime bounds did not match available time range")
    if ihigh < ilow:
        raise ValueError("Invalid seltime range: upper bound occurs before lower bound")

    sl = slice(ilow, ihigh + 1)
    return emiss[:, sl], emisserr[:, sl], omg[:, sl], omgerr[:, sl], ti[:, sl], tierr[:, sl], time[sl], psi[:, sl]


def _load_spectroscopy_profile_data(
    shot: int,
    pro_expr: str,
    proerr_expr: str,
    override: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Open spectroscopy tree, validate module integrity, and load profile arrays.

    Returns
    -------
    invstruc : ndarray, shape [npsi, nt, nfield]
    invstrucerr : ndarray, shape [npsi, nt, nfield]
    psinorm : ndarray, shape [npsi] or [npsi, nt]
    time : ndarray, shape [nt]
    """
    conn = openTree(shot, "spectroscopy")
    try:
        # IDL safety check for module image sizes.
        mod_lengths = []
        for mod in ("MOD1", "MOD2", "MOD3", "MOD4"):
            try:
                val = conn.get(
                    rf"getnci(\SPECTROSCOPY::TOP.HIREX_SR.RAW_DATA:{mod}, \"length\")"
                )
                mod_lengths.append(float(np.asarray(val.data()).squeeze()))
            except Exception:
                mod_lengths.append(np.nan)

        finite_mods = np.asarray(mod_lengths)[np.isfinite(mod_lengths)]
        if finite_mods.size >= 2:
            mismatch = np.any(np.abs(finite_mods - finite_mods[0]) > 0)
            if mismatch and not override:
                raise RuntimeError(
                    "Likely module malfunction on this shot (module image sizes differ). "
                    "Pass override=True to force loading."
                )

        psi_expr = f"dim_of({pro_expr}, 0)"
        time_expr = f"dim_of({pro_expr}, 1)"

        invstruc_raw = np.asarray(conn.get(pro_expr).data())
        invstrucerr_raw = np.asarray(conn.get(proerr_expr).data())
        psinorm = np.asarray(conn.get(psi_expr).data())
        time = np.asarray(conn.get(time_expr).data()).squeeze()
    finally:
        conn.closeAllTrees()

    # Normalize to [npsi, nt, nfield]. Pass nt so axes are correctly oriented
    # when MDSplus returns [nt, npsi, nfield] rather than [npsi, nt, nfield].
    invstruc = _ensure_profile_cube(invstruc_raw, nt=time.size)
    invstrucerr = _ensure_profile_cube(invstrucerr_raw, nt=time.size)

    return invstruc, invstrucerr, psinorm, time


def hirexsr_get_profile_py(
    shot: int,
    line: int = 2,
    quiet: bool = False,
    dc_shift: float = 0.0,
    tht: int = 0,
    override: bool = False,
    seltime: Sequence[float] | None = None,
) -> InversionProfileResult:
    """Python rewrite of IDL `HIREXSR_GET_PROFILE`.

    Returns inverted profile quantities, including mapped major radius and
    toroidal rotation in km/s, with m=1 split handling matching the IDL logic.
    """
    initstring = _analysis_initstring(tht)
    line_node, lineid = _line_from_index(line=line, tht=tht)
    pro_expr = initstring + line_node + ":PRO"
    proerr_expr = initstring + line_node + ":PROERR"

    invstruc, invstrucerr, psinorm, time = _load_spectroscopy_profile_data(
        shot=shot,
        pro_expr=pro_expr,
        proerr_expr=proerr_expr,
        override=override,
    )

    # invstruc is [nch, nt, nfield] after normalization.
    nch = invstruc.shape[0]
    nt_full = invstruc.shape[1]
    valid = np.where(time != -1)[0]

    if valid.size == 0:
        raise RuntimeError("No valid times found in profile time base (all time == -1)")

    # Field ordering follows IDL layout in PRO/PROERR:
    # 0=emissivity, 1=rotation frequency (omega), 2=vpol, 3=Ti.
    emiss = invstruc[:, valid, 0]
    emisserr = invstrucerr[:, valid, 0]
    ti = invstruc[:, valid, 3]
    tierr = invstrucerr[:, valid, 3]
    omg = invstruc[:, valid, 1] + float(dc_shift)
    omgerr = invstrucerr[:, valid, 1]
    subtime = time[valid]

    if psinorm.ndim == 2:
        # MDSplus returns the time-varying psi axis transposed relative to IDL:
        # IDL stores [nch, nt], Python/mdsthin can read it as [nt, nch].
        # Normalize to [nch, nt_full] before slicing.
        if psinorm.shape == (nt_full, nch):
            psinorm = psinorm.T
        elif psinorm.shape != (nch, nt_full):
            raise ValueError(
                f"2D psinorm shape {psinorm.shape} is inconsistent with "
                f"expected (nch={nch}, nt={nt_full}) or its transpose."
            )
        subpsinorm = psinorm[:, valid]
    elif psinorm.ndim == 1:
        subpsinorm = np.repeat(psinorm[:, None], subtime.size, axis=1)
    else:
        raise ValueError(f"Unexpected psinorm shape {psinorm.shape}")

    emiss, emisserr, omg, omgerr, ti, tierr, subtime, subpsinorm = _slice_seltime(
        emiss=emiss,
        emisserr=emisserr,
        omg=omg,
        omgerr=omgerr,
        ti=ti,
        tierr=tierr,
        time=subtime,
        psi=subpsinorm,
        seltime=seltime,
    )

    conn_a = openTree(shot, "analysis")
    try:
        rmid = np.asarray(conn_a.get(r"\ANALYSIS::EFIT_RMID").data())
        efit_time = np.asarray(conn_a.get(r"dim_of(\ANALYSIS::EFIT_RMID,0)").data())
        rpsi = np.asarray(conn_a.get(r"dim_of(\ANALYSIS::EFIT_RMID,1)").data())
    finally:
        conn_a.closeAllTrees()

    r_proj = multi_interpol(rmid=rmid, rpsi=rpsi, efit_time=efit_time, psinorm=subpsinorm, times=subtime)

    # Convert omega -> toroidal velocity using v = 2*pi*R*omega.
    rot = 2.0 * np.pi * r_proj * omg
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(omg != 0, omgerr / omg, np.nan)
    roterr = rot * frac

    nt = subtime.size
    # Radial average of projected major radius over selected times.
    r_ave = np.nansum(r_proj, axis=1) / float(nt)

    if not quiet:
        print(f"Loaded inversion profile: shot={shot}, line={lineid}, nt={nt}, npsi={r_proj.shape[0]}")

    numspatial = subpsinorm.shape[0]
    has_m1 = False
    if numspatial >= 4 and numspatial % 2 == 0:
        half = numspatial // 2
        logictest1 = np.all(subpsinorm[0, :] == subpsinorm[half, :])
        logictest2 = np.all(subpsinorm[1, :] == subpsinorm[1 + half, :])
        has_m1 = bool(logictest1 and logictest2)

    if has_m1:
        half = numspatial // 2
        finindex = half - 1
        startindex = half

        base = slice(0, finindex + 1)
        m1 = slice(startindex, numspatial)

        return InversionProfileResult(
            shot=shot,
            lineid=lineid,
            time=subtime,
            psi=subpsinorm[base, :],
            r_maj=r_proj[base, :],
            r_ave=r_ave[base],
            emiss=emiss[base, :],
            emisserr=emisserr[base, :],
            ti=ti[base, :],
            tierr=tierr[base, :],
            omg=omg[base, :],
            omgerr=omgerr[base, :],
            rot=rot[base, :],
            roterr=roterr[base, :],
            has_m1=True,
            emissm1=emiss[m1, :],
            emisserrm1=emisserr[m1, :],
            tim1=ti[m1, :],
            tierrm1=tierr[m1, :],
            omgm1=omg[m1, :],
            omgerrm1=omgerr[m1, :],
            rotm1=rot[m1, :],
            roterrm1=roterr[m1, :],
        )

    return InversionProfileResult(
        shot=shot,
        lineid=lineid,
        time=subtime,
        psi=subpsinorm,
        r_maj=r_proj,
        r_ave=r_ave,
        emiss=emiss,
        emisserr=emisserr,
        ti=ti,
        tierr=tierr,
        omg=omg,
        omgerr=omgerr,
        rot=rot,
        roterr=roterr,
        has_m1=False,
    )


def _print_summary(out: InversionProfileResult) -> None:
    print("=== hirexsr_get_profile_py summary ===")
    print(f"shot={out.shot}, line={out.lineid}, has_m1={out.has_m1}")
    print(f"time shape: {out.time.shape}")
    print(f"psi shape: {out.psi.shape}")
    print(f"r_maj shape: {out.r_maj.shape}")
    print(f"rot shape: {out.rot.shape}")
    print(f"ti shape: {out.ti.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python rewrite of IDL HIREXSR_GET_PROFILE")
    parser.add_argument("--shot", type=int, default=1120906030, help="Shot number to load")
    parser.add_argument("--tht", type=int, default=0)
    parser.add_argument(
        "--line",
        type=int,
        default=2,
        help="Line index (lint-style mapping in _line_config; default 2=He-like Z)",
    )
    parser.add_argument("--dc-shift", type=float, default=0.0, help="Additive shift applied to omega")
    parser.add_argument("--override", action="store_true", help="Bypass module mismatch safety check")
    parser.add_argument(
        "--seltime",
        nargs=2,
        type=float,
        metavar=("T_LOW", "T_HIGH"),
        help="Time window selection in seconds",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plot", dest="plot", action="store_true", default=True, help="Show plots")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plots")
    parser.add_argument("--every-nth", type=int, default=10, help="Plot every nth time point")
    parser.add_argument(
        "--compare-lint",
        action="store_true",
        help="Also run line-integrated loader and generate profile-vs-lint comparison plots",
        default=True,
    )
    parser.add_argument("--lint-line", type=int, default=7, help="Line index for lint comparison")
    parser.add_argument(
        "--lint-use-idl-profile-moments",
        action="store_true",
        default=False,
        help="Use pmom/perr in lint comparison (IDL-like profile moments)",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="",
        help="Optional prefix for saving plot PDF(s); empty string disables saving",
    )
    parser.add_argument(
        "--x-axis",
        choices=["psi", "r_maj"],
        default="psi",
        help="Radial coordinate for plotting (default: psi)",
    )
    args = parser.parse_args()

    out = hirexsr_get_profile_py(
        shot=args.shot,
        line=args.line,
        quiet=args.quiet,
        dc_shift=args.dc_shift,
        tht=args.tht,
        override=args.override,
        seltime=args.seltime,
    )
    _print_summary(out)
    if args.plot:
        _plot_inversion_profile(
            out,
            tht=args.tht,
            every_nth=args.every_nth,
            x_axis=args.x_axis,
            doSave=args.save_prefix,
        )

    if args.compare_lint:
        lint = hirexsr_get_lint_profile_py(
            shot=args.shot,
            line=args.lint_line,
            tht=args.tht,
            use_idl_profile_moments=args.lint_use_idl_profile_moments,
        )
        _plot_lint_profile(
            lint,
            every_nth=args.every_nth,
            x_axis="r_proj" if args.x_axis == "r_maj" else "rhotang",
        )
        _plot_profile_vs_lint(
            prof=out,
            lint=lint,
            tht=args.tht,
            x_axis=args.x_axis,
            every_nth=args.every_nth,
            doSave=args.save_prefix,
        )

    print('Done.')
