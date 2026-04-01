"""Shared plotting helpers for HIREX-SR profile and lint-profile results."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def _plot_inversion_profile(
    out: Any,
    tht: int,
    every_nth: int = 10,
    x_axis: str = "psi",
    doSave: str = "",
    doOmega: bool = True,
    w_limit: float = 100.0,
    Ti_limit: float = 10.0,
    emiss_lim: float = 3.0,
    specific_timepoint: float | None = None,
) -> None:
    """Plot emissivity, rotation, and Ti versus chosen radial coordinate."""
    if x_axis not in {"psi", "r_maj"}:
        raise ValueError(f"Unsupported x_axis='{x_axis}'. Use 'psi' or 'r_maj'.")

    nt = out.time.size
    time_indices = list(range(0, nt, max(1, every_nth)))
    if not time_indices:
        print("No time points to plot.")
        return
    if specific_timepoint is not None:
        closest_idx = np.argmin(np.abs(out.time - specific_timepoint))
        time_indices = [closest_idx]

    line_display = out.lineid
    try:
        from hirexsr_get_profile_py import _line_from_index
        from hirexsr_lint_profile_py import _line_display_name

        for i in range(10):
            if _line_from_index(i, tht)[1] == out.lineid:
                line_display = _line_display_name(i, tht)
                break
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    fig.suptitle(
        f"shot {out.shot}  line {line_display}  tht {tht} profile  (every {max(1, every_nth)} time pts)"
    )

    cmap = plt.get_cmap("viridis", max(len(time_indices), 1))
    for idx, it in enumerate(time_indices):
        color = cmap(idx / max(len(time_indices) - 1, 1))
        label = f"t={out.time[it]:.3f} s"
        if x_axis == "psi":
            xvals = out.psi[:, it]
            x_label = r"$\Psi_N$"
        else:
            xvals = out.r_maj[:, it]
            x_label = r"$R_{maj}$ [m]"

        yrot = out.omg[:, it] if doOmega else out.rot[:, it]
        yrot_err = out.omgerr[:, it] if doOmega else out.roterr[:, it]
        axes[0].errorbar(xvals, yrot, yerr=yrot_err, color=color, label=label)
        axes[1].errorbar(xvals, out.ti[:, it], yerr=out.tierr[:, it], color=color, label=label)
        axes[2].errorbar(xvals, out.emiss[:, it], yerr=out.emisserr[:, it], color=color, label=label)

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(r"$\omega$ [kHz]" if doOmega else r"$v_\phi$ [km/s]")
    axes[0].set_title("Inverted Toroidal Rotation")
    axes[0].grid(True)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].set_ylim([-w_limit, w_limit] if w_limit is not None else None)

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(r"$T_i$ [keV]")
    axes[1].set_title("Inverted Ion Temperature")
    axes[1].grid(True)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].set_ylim([0, Ti_limit] if Ti_limit is not None else None)

    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Emissivity [a.u.]")
    axes[2].set_title("Inverted Emissivity")
    axes[2].grid(True)
    axes[2].legend(fontsize=8, loc="best")
    axes[2].set_ylim([0, emiss_lim] if emiss_lim is not None else None)

    plt.show()
    if doSave:
        fig.savefig(f"{doSave}shot{out.shot}_line{out.lineid}_tht{tht}_{x_axis}.pdf", transparent=True)
        print(f"Saved plot to {doSave}_shot{out.shot}_line{out.lineid}_tht{tht}_{x_axis}.pdf")


def _curve_mask(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_abs_max: float | None = None,
) -> np.ndarray:
    """Return mask for valid/finite curve points with optional bounds."""
    m = np.isfinite(x) & np.isfinite(y) & (x != -1.0)
    if yerr is not None:
        m &= np.isfinite(yerr)
    if x_bounds is not None:
        m &= (x >= x_bounds[0]) & (x <= x_bounds[1])
    if y_abs_max is not None:
        m &= np.abs(y) <= y_abs_max
    return m


def _plot_profile_vs_lint(
    prof: Any,
    lint: Any,
    tht: int,
    x_axis: str = "psi",
    every_nth: int = 5,
    doSave: str = "",
    omega_abs_max: float = 100.0,
    ti_abs_max: float = 10.0,
    emiss_abs_max: float | None = None,
    v_lims: tuple[float, float] | None = (-40.0, 40.0),
    ti_lims: tuple[float, float] | None = (0.0, 6.0),
    emiss_lims: tuple[float, float] | None = (0.0, 3.0),
    specific_timepoint: float | None = None,
) -> None:
    """Overlay inversion-profile and lint-profile measurements at matched times."""
    if x_axis not in {"psi", "r_maj"}:
        raise ValueError(f"Unsupported x_axis='{x_axis}'. Use 'psi' or 'r_maj'.")

    if prof.time.size == 0 or lint.tau.size == 0:
        print("No time points available for profile-vs-lint comparison.")
        return

    prof_time_idx = list(range(0, prof.time.size, max(1, every_nth)))
    if not np.any(prof_time_idx):
        print("No profile time points selected for comparison.")
        return
    if specific_timepoint is not None:
        prof_time_idx = [int(np.argmin(np.abs(prof.time - specific_timepoint)))]

    line_name = str(getattr(lint, "line", "?"))
    try:
        from hirexsr_lint_profile_py import _line_display_name

        line_name = _line_display_name(lint.line, lint.tht)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    fig.suptitle(
        f"shot {prof.shot}  line {line_name} profile vs lint comparison  "
        f"tht {tht}  (every {max(1, every_nth)} profile times)"
    )

    cmap = plt.get_cmap("tab20", max(len(prof_time_idx), 1))

    for i, ip in enumerate(prof_time_idx):
        t_prof = prof.time[ip]
        il = int(np.argmin(np.abs(lint.tau - t_prof)))
        t_lint = lint.tau[il]
        color = cmap(i / max(len(prof_time_idx) - 1, 1))

        if x_axis == "psi":
            x_prof = prof.psi[:, ip]
            x_lint = lint.rhotang[il, :]
            x_bounds = (0.0, 1.2)
            x_label = r"$\Psi_N$ / rhotang"
        else:
            x_prof = prof.r_maj[:, ip]
            x_lint = lint.r_proj[:, il]
            x_bounds = None
            x_label = r"$R$ [m]"

        m_prof_w = _curve_mask(x_prof, prof.omg[:, ip], prof.omgerr[:, ip], x_bounds=x_bounds, y_abs_max=omega_abs_max)
        m_lint_w = _curve_mask(x_lint, lint.v[il, :], lint.verr[il, :], x_bounds=x_bounds, y_abs_max=omega_abs_max)

        axes[0].errorbar(
            x_prof[m_prof_w],
            prof.omg[:, ip][m_prof_w],
            yerr=prof.omgerr[:, ip][m_prof_w],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            alpha=0.85,
            label=f"profile t={t_prof:.3f}",
        )
        axes[0].errorbar(
            x_lint[m_lint_w],
            lint.v[il, :][m_lint_w],
            yerr=lint.verr[il, :][m_lint_w],
            color=color,
            linestyle="--",
            marker="x",
            markersize=3,
            alpha=0.85,
            label=f"lint t={t_lint:.3f}",
        )

        m_prof_ti = _curve_mask(x_prof, prof.ti[:, ip], prof.tierr[:, ip], x_bounds=x_bounds, y_abs_max=ti_abs_max)
        m_lint_ti = _curve_mask(x_lint, lint.ti[il, :], lint.tierr[il, :], x_bounds=x_bounds, y_abs_max=ti_abs_max)
        axes[1].errorbar(
            x_prof[m_prof_ti],
            prof.ti[:, ip][m_prof_ti],
            yerr=prof.tierr[:, ip][m_prof_ti],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            alpha=0.85,
        )
        axes[1].errorbar(
            x_lint[m_lint_ti],
            lint.ti[il, :][m_lint_ti],
            yerr=lint.tierr[il, :][m_lint_ti],
            color=color,
            linestyle="--",
            marker="x",
            markersize=3,
            alpha=0.85,
        )

        m_prof_em = _curve_mask(x_prof, prof.emiss[:, ip], prof.emisserr[:, ip], x_bounds=x_bounds, y_abs_max=emiss_abs_max)
        m_lint_em = _curve_mask(x_lint, lint.emiss[il, :], lint.emisserr[il, :], x_bounds=x_bounds, y_abs_max=emiss_abs_max)
        axes[2].errorbar(
            x_prof[m_prof_em],
            prof.emiss[:, ip][m_prof_em],
            yerr=prof.emisserr[:, ip][m_prof_em],
            color=color,
            linestyle="-",
            marker="o",
            markersize=3,
            alpha=0.85,
        )
        axes[2].errorbar(
            x_lint[m_lint_em],
            lint.emiss[il, :][m_lint_em],
            yerr=lint.emisserr[il, :][m_lint_em],
            color=color,
            linestyle="--",
            marker="x",
            markersize=3,
            alpha=0.85,
        )

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(r"$\omega$ [kHz]")
    axes[0].set_title("Omega: Profile (solid) vs Lint (dashed)")
    axes[0].grid(True)
    axes[0].legend(fontsize=7, loc="best")
    if v_lims is not None:
        axes[0].set_ylim(v_lims)

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(r"$T_i$ [keV]")
    axes[1].set_title("Ti: Profile (solid) vs Lint (dashed)")
    axes[1].grid(True)
    axes[1].set_ylim(bottom=0.0)
    if ti_lims is not None:
        axes[1].set_ylim(ti_lims)

    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Emissivity [a.u.]")
    axes[2].set_title("Emissivity: Profile (solid) vs Lint (dashed)")
    axes[2].grid(True)
    axes[2].set_ylim(bottom=0.0)
    if emiss_lims is not None:
        axes[2].set_ylim(emiss_lims)

    plt.show()
    if doSave:
        out_path = f"{doSave}shot{prof.shot}_line{prof.lineid}_tht{tht}_{x_axis}_profile_vs_lint.pdf"
        fig.savefig(out_path, transparent=True)
        print(f"Saved comparison plot to {out_path}")


def _plot_lint_profile(
    out: Any,
    every_nth: int = 10,
    x_axis: str = "r_proj",
    specific_timepoint: float | list[float] | None = None,
    w_lim: float = 150,
    ti_lim: float = 10,
    emiss_lim: float = 3,
    doSave: str = "",
) -> None:
    """Plot v, Ti, and emissivity versus chosen radial coordinate."""
    tau = out.tau
    v = out.v
    ti = out.ti
    emiss = out.emiss
    rhotang = out.rhotang
    r_proj = out.r_proj

    if x_axis not in {"r_proj", "rhotang"}:
        raise ValueError(f"Unsupported x_axis='{x_axis}'. Use 'r_proj' or 'rhotang'.")

    nt_valid = tau.size
    time_indices = list(range(0, nt_valid, every_nth))
    if not time_indices:
        print("No time points to plot.")
        return

    if specific_timepoint is not None:
        if not isinstance(specific_timepoint, list):
            specific_timepoint = [float(specific_timepoint)]
        time_indices = []
        for timepoint in specific_timepoint:
            time_indices.append(np.argmin(np.abs(tau - timepoint)))

    line_name = str(getattr(out, "line", "?"))
    try:
        from hirexsr_lint_profile_py import _line_display_name

        line_name = _line_display_name(out.line, out.tht)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
    fig.suptitle(
        f"shot {out.shot}  line {line_name}  tht {out.tht}  (every {every_nth} time pts)"
    )

    cmap = plt.get_cmap("viridis", max(len(time_indices), 1))
    for idx, it in enumerate(time_indices):
        color = cmap(idx / max(len(time_indices) - 1, 1))
        label = f"t={tau[it]:.3f} s"
        if x_axis == "r_proj":
            xvals = r_proj[:, it]
        else:
            xvals = rhotang[it, :]
        v_valid = np.isfinite(v[it, :]) & (np.abs(v[it, :]) < w_lim) & (xvals > 0)
        ti_valid = np.isfinite(ti[it, :]) & (ti[it, :] < ti_lim) & (xvals > 0)
        emiss_valid = np.isfinite(emiss[it, :]) & (emiss[it, :] >= 0) & (xvals > 0)
        axes[0].plot(xvals[v_valid], v[it, v_valid], color=color, label=label)
        axes[1].plot(xvals[ti_valid], ti[it, ti_valid], color=color, label=label)
        axes[2].plot(xvals[emiss_valid], emiss[it, emiss_valid], color=color, label=label)

    if x_axis == "r_proj":
        x_label = "Rproj [m]"
    else:
        x_label = r"$\Psi_N$"

    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(r"$\omega$  [kHz]")
    axes[0].set_title("Velocity")
    axes[0].legend(fontsize=7, loc="best")

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Ti  [keV]")
    axes[1].set_title("Ion Temperature")
    axes[1].legend(fontsize=7, loc="best")

    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Emissivity [a.u.]")
    axes[2].set_title("Emissivity")
    axes[2].legend(fontsize=7, loc="best")

    for ax in axes:
        ax.grid(True)
        ax.set_xlim(0, 1)
    axes[1].set_ylim([0, ti_lim])
    axes[2].set_ylim([0, emiss_lim])
    plt.tight_layout()
    plt.show()
    if doSave:
        fig.savefig(
            f"{doSave}lint_profile_shot{out.shot}_line{out.line}_tht{out.tht}_{x_axis}.pdf",
            transparent=True,
        )
        print(
            f"Saved plot to {doSave}lint_profile_shot{out.shot}_line{out.line}_tht{out.tht}_{x_axis}.pdf"
        )
