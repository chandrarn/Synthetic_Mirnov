import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

try:
    import MDSplus
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some environments
    MDSplus = None


def _time_first(arr, nt):
    """Return a view where time is axis 0 whenever it can be inferred."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.shape[0] == nt:
        return arr
    if arr.ndim >= 2 and arr.shape[1] == nt:
        return np.swapaxes(arr, 0, 1)
    return arr


def _safe_interp1(x, y, x_new):
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)
    if x.size == 0 or y.size == 0:
        return np.zeros_like(x_new, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(x_new, float(np.nan), dtype=float)
    return np.interp(x_new, x[finite], y[finite])


def _moving_average_time(arr, width=5):
    """Simple time smoothing equivalent to IDL smooth(..., [1, width])."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or width <= 1:
        return arr

    kernel = np.ones(width, dtype=float) / float(width)
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(arr[:, j], kernel, mode="same")
    return out


def _median_filter_axis1(arr, width=3):
    """Apply a simple odd-width median filter along axis 1 (channel axis)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or width <= 1 or width % 2 == 0:
        return arr

    pad = width // 2
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(arr)
    for j in range(arr.shape[1]):
        out[:, j] = np.nanmedian(padded[:, j : j + width], axis=1)
    return out


def _tri_interp(x_src, y_src, z_src, x_grid, y_grid):
    """
    Interpolate scattered (x, y, z) data onto a regular grid.

    Behavior is tuned to be closer to IDL trigrid:
    1) Linear interpolation on the triangulation interior.
    2) Nearest-neighbor fill for points outside the convex hull.
    """
    x_src = np.asarray(x_src, dtype=float).ravel()
    y_src = np.asarray(y_src, dtype=float).ravel()
    z_src = np.asarray(z_src, dtype=float).ravel()

    finite = np.isfinite(x_src) & np.isfinite(y_src) & np.isfinite(z_src)
    x_src = x_src[finite]
    y_src = y_src[finite]
    z_src = z_src[finite]

    if x_src.size < 3:
        return np.full_like(x_grid, np.nan, dtype=float)

    tri = mtri.Triangulation(x_src, y_src)
    interp = mtri.LinearTriInterpolator(tri, z_src)
    z_grid = interp(x_grid, y_grid)

    if np.ma.isMaskedArray(z_grid):
        z_grid = z_grid.filled(np.nan)

    z_grid = np.asarray(z_grid, dtype=float)

    # Fill outside-convex-hull points using nearest source sample.
    mask = ~np.isfinite(z_grid)
    if np.any(mask):
        xq = np.asarray(x_grid, dtype=float)[mask]
        yq = np.asarray(y_grid, dtype=float)[mask]
        nearest_vals = np.empty(xq.size, dtype=float)

        # Chunked nearest-neighbor search to avoid large temporary arrays.
        chunk = 4096
        for i0 in range(0, xq.size, chunk):
            i1 = min(i0 + chunk, xq.size)
            dx = xq[i0:i1, None] - x_src[None, :]
            dy = yq[i0:i1, None] - y_src[None, :]
            d2 = dx * dx + dy * dy
            nearest_vals[i0:i1] = z_src[np.argmin(d2, axis=1)]

        z_grid[mask] = nearest_vals

    return z_grid

def doConsistencyCheck(zeff, times):
    if zeff.ndim != 1 or times.ndim != 1 or zeff.size < 2:
        raise ValueError("zeff_neo returned insufficient Zeff samples.")

    finite = np.isfinite(zeff) & np.isfinite(times)
    if np.count_nonzero(finite) < 2:
        raise ValueError("zeff_neo returned non-finite Zeff values.")
    
    if np.any(zeff < 0.0):
        raise ValueError("zeff_neo returned negative Zeff values, which are unphysical.")

def _interp_zeff_from_ip_curve(zeff_grid, ip_curve, target_ip):
    """
    IDL-like 1D interpolation helper for final Zeff inference.

    Given Ip_neo(Zeff) and measured Ip, infer Zeff by monotonic interpolation,
    while guarding against non-finite values and duplicate Ip points.
    """
    zeff_grid = np.asarray(zeff_grid, dtype=float)
    ip_curve = np.asarray(ip_curve, dtype=float)

    finite = np.isfinite(zeff_grid) & np.isfinite(ip_curve)
    if finite.sum() < 2 or not np.isfinite(target_ip):
        return np.nan

    x = ip_curve[finite]
    y = zeff_grid[finite]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Collapse duplicates in x by averaging y, producing strictly increasing x.
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.zeros_like(xu)
    counts = np.zeros_like(xu)
    for i, g in enumerate(inv):
        yu[g] += y[i]
        counts[g] += 1.0
    yu = np.divide(yu, counts, out=np.copy(yu), where=counts > 0)

    if xu.size < 2:
        return np.nan

    return float(np.interp(target_ip, xu, yu, left=yu[0], right=yu[-1]))


def _interp_zeff_from_ip_curve_diag(zeff_grid, ip_curve, target_ip):
    """Diagnostic variant of _interp_zeff_from_ip_curve returning internals."""
    zeff_grid = np.asarray(zeff_grid, dtype=float)
    ip_curve = np.asarray(ip_curve, dtype=float)

    finite = np.isfinite(zeff_grid) & np.isfinite(ip_curve)
    diag = {
        "finite_count": int(np.count_nonzero(finite)),
        "input_zeff_grid": zeff_grid,
        "input_ip_curve": ip_curve,
        "target_ip": float(target_ip),
    }
    if finite.sum() < 2 or not np.isfinite(target_ip):
        diag["status"] = "insufficient_data"
        diag["zeff"] = np.nan
        return np.nan, diag

    x = ip_curve[finite]
    y = zeff_grid[finite]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    xu, inv = np.unique(x, return_inverse=True)
    yu = np.zeros_like(xu)
    counts = np.zeros_like(xu)
    for i, g in enumerate(inv):
        yu[g] += y[i]
        counts[g] += 1.0
    yu = np.divide(yu, counts, out=np.copy(yu), where=counts > 0)

    if xu.size < 2:
        diag["status"] = "degenerate_curve"
        diag["x_unique"] = xu
        diag["y_unique"] = yu
        diag["zeff"] = np.nan
        return np.nan, diag

    zeff_val = float(np.interp(target_ip, xu, yu, left=yu[0], right=yu[-1]))

    diag["status"] = "ok"
    diag["x_unique"] = xu
    diag["y_unique"] = yu
    diag["x_span"] = (float(np.min(xu)), float(np.max(xu)))
    diag["y_span"] = (float(np.min(yu)), float(np.max(yu)))
    i_right = int(np.searchsorted(xu, target_ip))
    i_left = max(0, i_right - 1)
    i_right = min(len(xu) - 1, i_right)
    if i_right == i_left:
        i_right = min(len(xu) - 1, i_left + 1)
    if i_right != i_left:
        local_slope = float((yu[i_right] - yu[i_left]) / (xu[i_right] - xu[i_left]))
    else:
        local_slope = np.nan
    x_min, x_max = float(np.min(xu)), float(np.max(xu))
    if x_max > x_min:
        target_position = float((target_ip - x_min) / (x_max - x_min))
    else:
        target_position = np.nan
    diag["local_slope"] = local_slope
    diag["target_position"] = target_position
    diag["zeff"] = zeff_val
    return zeff_val, diag


def _summarize_diag_bins(diag_bins):
    """Build cross-bin diagnostics to inspect shape/offset trends."""
    if len(diag_bins) == 0:
        return {
            "n_bins": 0,
            "ok_bins": 0,
            "failed_bins": 0,
        }

    statuses = [d.get("status", "unknown") for d in diag_bins]
    ok_mask = np.array([s == "ok" for s in statuses], dtype=bool)
    local_slopes = np.array([
        d.get("local_slope", np.nan) for d in diag_bins
    ], dtype=float)
    target_pos = np.array([
        d.get("target_position", np.nan) for d in diag_bins
    ], dtype=float)
    x_span_width = np.array([
        np.nan if "x_span" not in d else d["x_span"][1] - d["x_span"][0]
        for d in diag_bins
    ], dtype=float)

    summary = {
        "n_bins": int(len(diag_bins)),
        "ok_bins": int(np.count_nonzero(ok_mask)),
        "failed_bins": int(len(diag_bins) - np.count_nonzero(ok_mask)),
        "status_counts": {
            s: int(sum(1 for ss in statuses if ss == s)) for s in sorted(set(statuses))
        },
        "local_slope_mean": float(np.nanmean(local_slopes)),
        "local_slope_std": float(np.nanstd(local_slopes)),
        "target_position_mean": float(np.nanmean(target_pos)),
        "target_position_std": float(np.nanstd(target_pos)),
        "x_span_width_mean": float(np.nanmean(x_span_width)),
        "x_span_width_std": float(np.nanstd(x_span_width)),
    }

    # Helpful flags for bins where interpolation may be extrapolation-like.
    near_edge = np.where((target_pos < 0.05) | (target_pos > 0.95))[0]
    summary["near_edge_bins"] = [int(i) for i in near_edge]
    return summary


def _array_stats(arr):
    """Compact numeric summary for diagnostics."""
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    frac_finite = float(np.mean(finite)) if a.size > 0 else 0.0
    out = {
        "shape": tuple(int(i) for i in a.shape),
        "frac_finite": frac_finite,
    }
    if np.any(finite):
        out["min"] = float(np.nanmin(a))
        out["max"] = float(np.nanmax(a))
        out["mean"] = float(np.nanmean(a))
    else:
        out["min"] = np.nan
        out["max"] = np.nan
        out["mean"] = np.nan
    return out


def _compute_ohmic_power_proxy(t, psurf, ip, li, ip2, volume, nbbbs, rbbbs, zbbbs):
    """
    Approximate ohmic heating power from EFIT equilibrium quantities.

    Mirrors the IDL sequence:
    - vsurf = d(psi_edge)/dt
    - poynt = vsurf * Ip
    - wmag uses internal inductance and plasma volume
    - poh = |poynt - d(wmag)/dt|
    """
    vsurf = np.gradient(psurf, t)
    ntime = t.size
    len2 = np.full(ntime, 1.0e10, dtype=float)

    # len2 is the squared LCFS poloidal perimeter, used in wmag scaling.
    for i in range(ntime):
        nb = int(nbbbs[i]) if i < nbbbs.size else 0
        if nb > 0:
            rb = np.asarray(rbbbs[i, :nb], dtype=float)
            zb = np.asarray(zbbbs[i, :nb], dtype=float)
            dr = rb - np.roll(rb, -1)
            dz = zb - np.roll(zb, -1)
            len2[i] = np.sum(np.sqrt(dr**2 + dz**2)) ** 2

    poynt = vsurf * ip
    wmag = 2.0e-7 * np.pi * li * ip2**2 * volume / np.maximum(len2, 1.0e-16)
    dwmag = np.gradient(wmag, t)
    return np.abs(poynt - dwmag)


def _build_regular_efit_grids(t, rmid, rout, ip, poh, qpsi, volp):
    """Map EFIT fields to regular time/radius grids following the IDL setup."""
    nt = max(2, int(1000.0 * (np.max(t) - np.min(t))))
    tgrid = np.min(t) + 0.001 * np.arange(nt)
    nr = max(2, int(500.0 * (0.91 - np.min(rmid))))
    rgrid = np.min(rmid) + 0.002 * np.arange(nr)

    # Interpolate 1D traces to regular time grid.
    rout_i = _safe_interp1(t, rout, tgrid)
    ip_i = _safe_interp1(t, ip, tgrid)
    poh_i = _safe_interp1(t, poh, tgrid)

    # Interpolate 2D EFIT fields to regular (tgrid, rgrid).
    t2d = np.repeat(t[:, None], rmid.shape[1], axis=1)
    rr, tt = np.meshgrid(rgrid, tgrid)
    qpsi_i = _tri_interp(rmid, t2d, qpsi, rr, tt)
    volp_i = _tri_interp(rmid, t2d, volp, rr, tt)

    return {
        "nt": nt,
        "nr": nr,
        "tgrid": tgrid,
        "rgrid": rgrid,
        "rout_i": rout_i,
        "ip_i": ip_i,
        "poh_i": poh_i,
        "qpsi_i": qpsi_i,
        "volp_i": volp_i,
    }


def _get_analysis_data(shot, verbose=False) -> tuple:
    if MDSplus is None:
        raise ModuleNotFoundError("MDSplus is required for zeff_neo Python implementation")

    tree = MDSplus.Tree("analysis", int(shot))

    # Time base from EFIT current trace.
    pcurr_node = tree.getNode("\\efit_geqdsk:pcurrt").getData()
    try:
        # IDL uses dim_of(...,2) for time in this EFIT signal.
        t = np.asarray(pcurr_node.dim_of(2).data(), dtype=float)
    except Exception:
        try:
            t = np.asarray(pcurr_node.dim_of(1).data(), dtype=float)
        except Exception:
            t = np.asarray(pcurr_node.dim_of(0).data(), dtype=float)

    data = {
        "t": t,
        "rout": np.asarray(tree.getNode("\\efit_aeqdsk:rout").data(), dtype=float)
        / 100.0,
        "rmid": np.asarray(tree.getNode("\\EFIT_RMID").data(), dtype=float),
        "ip": np.abs(np.asarray(tree.getNode("\\efit_aeqdsk:pasmat").data(), dtype=float)),
        "ip2": np.asarray(tree.getNode("\\efit_aeqdsk:cpasma").data(), dtype=float),
        "li": np.asarray(tree.getNode("\\efit_aeqdsk:ali").data(), dtype=float),
        "wmhd": np.asarray(tree.getNode("\\efit_aeqdsk:wplasm").data(), dtype=float),
        "psurf": np.asarray(tree.getNode("\\efit_aeqdsk:sibdry").data(), dtype=float)
        * 2.0
        * np.pi,
        "nbbbs": np.asarray(tree.getNode("\\EFIT_GEQDSK:NBBBS").data(), dtype=int),
        "rbbbs": np.asarray(tree.getNode("\\efit_geqdsk:rbbbs").data(), dtype=float),
        "zbbbs": np.asarray(tree.getNode("\\efit_geqdsk:zbbbs").data(), dtype=float),
        "volume": np.asarray(tree.getNode("\\efit_aeqdsk:vout").data(), dtype=float)
        / 1.0e6,
        "volp": np.asarray(tree.getNode("\\efit_fitout:volp").data(), dtype=float),
        "qpsi": np.asarray(tree.getNode("\\efit_fitout:qpsi").data(), dtype=float),
    }

    nt = data["t"].size
    data["rmid"] = _time_first(data["rmid"], nt)
    data["qpsi"] = _time_first(data["qpsi"], nt)
    data["volp"] = _time_first(data["volp"], nt)
    data["rbbbs"] = _time_first(data["rbbbs"], nt)
    data["zbbbs"] = _time_first(data["zbbbs"], nt)

    if verbose:
        print("EFIT data loaded")

    # Explicitly convert all data to numpy arrays with consistent dtypes.
    t = np.asarray(data["t"], dtype=float)
    rout = np.asarray(data["rout"], dtype=float)
    rmid = np.asarray(data["rmid"], dtype=float)
    ip = np.asarray(data["ip"], dtype=float)
    ip2 = np.asarray(data["ip2"], dtype=float)
    li = np.asarray(data["li"], dtype=float)
    wmhd = np.asarray(data["wmhd"], dtype=float)
    psurf = np.asarray(data["psurf"], dtype=float)
    nbbbs = np.asarray(data["nbbbs"], dtype=int)
    rbbbs = np.asarray(data["rbbbs"], dtype=float)
    zbbbs = np.asarray(data["zbbbs"], dtype=float)
    volume = np.asarray(data["volume"], dtype=float)
    volp = np.asarray(data["volp"], dtype=float)
    qpsi = np.asarray(data["qpsi"], dtype=float)

    return t, rout, rmid, ip, ip2, li, wmhd, psurf, nbbbs, rbbbs, zbbbs, volume, volp, qpsi

def _get_ts_local(shot, efit, verbose=False):
    """
    Port of IDL get_ts_local:
    Build Te/Ne fields on the regular EFIT (tgrid, rgrid) mesh.
    """
    if MDSplus is None:
        raise ModuleNotFoundError("MDSplus is required for zeff_neo Python implementation")

    tree = MDSplus.Tree("electrons", int(shot))

    te_core = np.asarray(tree.getNode("\\yag_new.results.profiles:te_rz").data(), dtype=float)
    ne_core = np.asarray(tree.getNode("\\yag_new.results.profiles:ne_rz").data(), dtype=float)
    r_core = np.asarray(tree.getNode("yag_new.results.profiles:r_mid_t").data(), dtype=float)
    t_core = np.asarray(
        tree.getNode("\\yag_new.results.profiles:te_rz").getData().dim_of(0).data(),
        dtype=float,
    )

    # Original IDL scales Te to eV before transport calculations.
    te_core = 1000.0 * te_core

    nt_core = t_core.size
    te_core = _time_first(te_core, nt_core)
    ne_core = _time_first(ne_core, nt_core)
    r_core = _time_first(r_core, nt_core)

    # IDL median(..., 3, dim=2): remove channel dropouts before interpolation.
    te_core = _median_filter_axis1(te_core, width=3)
    ne_core = _median_filter_axis1(ne_core, width=3)

    try:
        te_edge = np.asarray(tree.getNode("\\ts_te").data(), dtype=float)
        ne_edge = np.asarray(tree.getNode("\\ts_ne").data(), dtype=float)
        r_edge = np.asarray(tree.getNode("\\ts_rmid").data(), dtype=float)
        t_edge = np.asarray(tree.getNode("\\ts_te").getData().dim_of(0).data(), dtype=float)

        nt_edge = t_edge.size
        te_edge = _time_first(te_edge, nt_edge)
        ne_edge = _time_first(ne_edge, nt_edge)
        r_edge = _time_first(r_edge, nt_edge)

        if verbose:
            print("Got Edge TS")
    except Exception:
        t_edge = t_core
        te_edge = np.zeros((t_edge.size, 1), dtype=float)
        ne_edge = np.zeros((t_edge.size, 1), dtype=float)
        r_edge = np.full((t_edge.size, 1), 0.91, dtype=float)
        if verbose:
            print("Error getting edge TS")

    # Bring core channels onto edge timebase before concatenation.
    te_core_i = np.empty((t_edge.size, te_core.shape[1]), dtype=float)
    ne_core_i = np.empty((t_edge.size, ne_core.shape[1]), dtype=float)
    r_core_i = np.empty((t_edge.size, r_core.shape[1]), dtype=float)
    for j in range(te_core.shape[1]):
        te_core_i[:, j] = _safe_interp1(t_core, te_core[:, j], t_edge)
        ne_core_i[:, j] = _safe_interp1(t_core, ne_core[:, j], t_edge)
        r_core_i[:, j] = _safe_interp1(t_core, r_core[:, j], t_edge)

    r_temp = np.maximum(np.concatenate([r_core_i, r_edge], axis=1), 0.0)
    te_temp = np.concatenate([te_core_i, te_edge], axis=1)
    ne_temp = np.concatenate([ne_core_i, ne_edge], axis=1)

    # Sort each time slice by radius.
    order = np.argsort(r_temp, axis=1)
    rows = np.arange(r_temp.shape[0])[:, None]
    r_raw = r_temp[rows, order]
    te_raw = te_temp[rows, order]
    ne_raw = ne_temp[rows, order]

    # Pad at small radius to stabilize triangulation, mirroring IDL behavior.
    r_pad = np.full((r_raw.shape[0], 1), 0.5, dtype=float)
    te_pad = te_raw[:, :1]
    ne_pad = ne_raw[:, :1]
    r_raw = np.concatenate([r_pad, r_raw], axis=1)
    te_raw = np.concatenate([te_pad, te_raw], axis=1)
    ne_raw = np.concatenate([ne_pad, ne_raw], axis=1)

    valid_t = np.where(np.nansum(r_raw, axis=1) > 0.0)[0]
    r_raw = r_raw[valid_t]
    te_raw = te_raw[valid_t]
    ne_raw = ne_raw[valid_t]
    t_raw = t_edge[valid_t]

    # Optional smoothing in time, matching the original smooth(..., [1, 5]).
    te_smooth = _moving_average_time(te_raw, width=5)
    ne_smooth = _moving_average_time(ne_raw, width=5)

    t2d = np.repeat(t_raw[:, None], r_raw.shape[1], axis=1)
    rr, tt = np.meshgrid(efit["rgrid"], efit["tgrid"])

    te_fine = _tri_interp(r_raw, t2d, te_smooth, rr, tt)
    ne_fine = _tri_interp(r_raw, t2d, ne_smooth, rr, tt)

    ts = {
        "te": te_fine,
        "ner": ne_fine,
        "r_raw": r_raw,
        "t_raw": t_raw,
        "te_raw": te_raw,
        "ne_raw": ne_raw,
    }

    if verbose:
        print("Got Core TS")

    return ts

def make_plots(tgrid, ip_neo_ma, ip_ma, times, zeff, zeff_grid, shot, save_plots=""):
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))

    ax = axs[0]
    for i, z_val in enumerate(zeff_grid):
        ax.plot(tgrid, ip_neo_ma[:, i], lw=1.0, label=f"Zeff={z_val:.2f}")
    ax.plot(tgrid, ip_ma, c="k", lw=2.0, label="Ip measured")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Plasma Current [MA]")
    ax.legend(fontsize=8, ncol=2)

    ax = axs[1]
    ax.plot(times, zeff, "-o", ms=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Zeff")
    ax.set_title(f"Shot {shot}")
    for i in range(2): axs[i].grid(True)
    

    fig.tight_layout()

    if save_plots:
        fig.savefig(f"{save_plots}/zeff_neo_{shot}.pdf", transparent=True)
        print('Saved plot to:', f"{save_plots}/zeff_neo_{shot}.pdf")

    plt.show(block=True)

#################################################################################
#################################################################################

def zeff_neo(
    shot,
    zion=1.0,
    zeff_max=4.0,
    n_zeff=8,
    dt=0.1,
    trange=(0.5, 1.5),
    plot=False,
    verbose=False,
    strict_diagnostics=False,
    save_plots="",
):
    """
    Python port of the legacy IDL `zeff_neo` routine.

    Notes:
    - This implementation keeps the same physical model and data flow.
    - MDS calls are currently direct via MDSplus Tree.getNode(...).data().
      In a follow-up step these accesses can be routed through `conn.get(...)`
      from mdsthin without changing the numerical pipeline.
    """

    # Load EFIT data from MDS+ tree
    t, rout, rmid, ip, ip2, li, wmhd, psurf, nbbbs, rbbbs, zbbbs, volume, volp, qpsi = \
        _get_analysis_data(shot, verbose=verbose)

    # Effective ohmic heating power proxy used to reconstruct E_phi.
    poh = _compute_ohmic_power_proxy(t, psurf, ip, li, ip2, volume, nbbbs, rbbbs, zbbbs)

    # Build regular grids and interpolate EFIT fields, matching IDL resolution choices.
    grid_data = _build_regular_efit_grids(t, rmid, rout, ip, poh, qpsi, volp)
    nt = grid_data["nt"]
    nr = grid_data["nr"]
    tgrid = grid_data["tgrid"]
    rgrid = grid_data["rgrid"]
    rout_i = grid_data["rout_i"]
    ip_i = grid_data["ip_i"]
    poh_i = grid_data["poh_i"]
    qpsi_i = grid_data["qpsi_i"]
    volp_i = grid_data["volp_i"]

    # Differential shell volume between adjacent radial grid surfaces.
    dv = np.fmax(volp_i - np.roll(volp_i, 1, axis=1), 0.0)
    # Effective loop voltage per current, then toroidal electric field E_phi.
    vres = np.divide(
        poh_i,
        ip_i,
        out=np.zeros_like(poh_i, dtype=float),
        where=np.abs(ip_i) > 0.0,
    )

    efit = {
        "nt": nt,
        "nr": nr,
        "rout": rout_i,
        "rgrid": rgrid,
        "tgrid": tgrid,
        "qpsi": qpsi_i,
        "vres": vres,
        "dv": dv,
    }

    if verbose:
        print("EFIT data interpolated")

    ts = _get_ts_local(shot, efit, verbose=verbose)
    ner = np.asarray(ts["ner"], dtype=float)
    te = np.asarray(ts["te"], dtype=float)

    zeff_grid = np.linspace(zion, zeff_max, int(n_zeff))

    # ip_neo[t, z] stores neoclassical plasma current predicted at each time
    # for each assumed Zeff value in zeff_grid.
    ip_neo = np.zeros((nt, zeff_grid.size), dtype=float)
    majr = rout_i[:, None] * np.ones((1, nr), dtype=float)
    minr = np.ones((nt, 1), dtype=float) * rgrid[None, :] - majr

    # eps = r_minor / R_major geometric inverse aspect-ratio proxy.
    eps = minr / np.maximum(majr, 1.0e-16)
    eps_pos = np.where(eps > 0.0, eps, np.nan)
    ft = np.full_like(eps_pos, np.nan)
    pos = eps_pos > 0.0
    ft[pos] = np.sqrt(eps_pos[pos])

    ner_safe = np.where(np.isfinite(ner) & (ner > 0.0), ner, 1.0e-30)
    te_safe = np.where(np.isfinite(te) & (te > 0.0), te, 1.0e-6)

    # Lambda_e: electron Coulomb logarithm approximation.
    Lambda_e = 31.3 - np.log(np.sqrt(ner_safe) / te_safe)
    # ephi: toroidal electric field E_phi from effective loop voltage.
    ephi = (vres / (2.0 * np.pi * np.maximum(rout_i, 1.0e-16)))[:, None] * np.ones(
        (1, nr), dtype=float
    )

    if verbose:
        print("Calculating neoclassical Ip for Zeff")

    for i, z_val in enumerate(zeff_grid):
        if verbose:
            print(f"Zeff = {z_val}")

        zneo = z_val + np.zeros((nt, nr), dtype=float)
        # nz: effective ion-charge correction in Spitzer conductivity model.
        nz = 0.58 + 0.74 / (0.76 + zneo)
        # sig_spitz: Spitzer conductivity estimate.
        sig_spitz = 1.9e4 * te_safe**1.5 / (zneo * nz * Lambda_e)
        # nu_star_e: normalized electron collisionality.
        nu_star_e = (
            6.92e-18
            * qpsi_i
            * majr
            * ner_safe
            * zneo
            * Lambda_e
            / (te_safe**2 * eps_pos**1.5)
        )

        # ft_eff_33: effective trapped fraction correction (Sauter model form).
        ft_eff_33 = ft / (
            1.0
            + (0.55 - 0.1 * ft) * np.sqrt(np.maximum(nu_star_e, 0.0))
            + 0.45 * (1.0 - ft) * nu_star_e / np.maximum(zneo**1.5, 1.0e-16)
        )
        # sig_ratio: neoclassical correction factor to Spitzer conductivity.
        sig_ratio = (
            1.0
            - (1.0 + 0.36 / zneo) * ft_eff_33
            + (0.59 / zneo) * ft_eff_33**2
            - (0.23 / zneo) * ft_eff_33**3
        )
        # sig_neo: neoclassical conductivity.
        sig_neo = sig_spitz * sig_ratio

        # j: toroidal current density magnitude from sigma * E_phi.
        j = np.abs(sig_neo * ephi)
        j[~np.isfinite(j)] = 0.0

        ip_neo[:, i] = np.nansum(j * dv / (2.0 * np.pi * np.maximum(majr, 1.0e-16)), axis=1)

    # Convert A to MA for consistency with IDL plotting and interpolation.
    ip_ma = ip_i * 1.0e-6
    ip_neo_ma = ip_neo * 1.0e-6

    n_out = int(np.floor((trange[1] - trange[0]) / dt))
    if n_out <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    times = np.zeros(n_out, dtype=float)
    zeff = np.zeros(n_out, dtype=float)
    diag_bins = []

    for i in range(n_out):
        times[i] = trange[0] + (i + 0.5) * dt
        mask = (tgrid >= (times[i] - dt / 2.0)) & (tgrid < (times[i] + dt / 2.0))

        if not np.any(mask):
            continue

        # ip_curve is modeled Ip_neo(Zeff) averaged over the dt window.
        ip_curve = np.mean(ip_neo_ma[mask, :], axis=0)
        # target_ip is measured Ip over the same window.
        target_ip = float(np.mean(ip_ma[mask]))

        # Final Zeff estimate: invert Ip_neo(Zeff) at measured Ip.
        if strict_diagnostics:
            zeff[i], diag = _interp_zeff_from_ip_curve_diag(
                zeff_grid, ip_curve, target_ip
            )
            diag["bin"] = int(i)
            diag["time_center"] = float(times[i])
            diag_bins.append(diag)
        else:
            zeff[i] = _interp_zeff_from_ip_curve(zeff_grid, ip_curve, target_ip)

    if plot:
         make_plots(tgrid, ip_neo_ma, ip_ma, times, zeff, zeff_grid, shot, save_plots)

    if strict_diagnostics:
        diagnostics = {
            "times": times,
            "zeff": zeff,
            "ip_ma": ip_ma,
            "ip_neo_ma": ip_neo_ma,
            "zeff_grid": zeff_grid,
            "diag_bins": diag_bins,
            "grid_shapes": {
                "qpsi_i": qpsi_i.shape,
                "volp_i": volp_i.shape,
                "te": te.shape,
                "ner": ner.shape,
            },
        }
        diagnostics["diag_summary"] = _summarize_diag_bins(diag_bins)
        diagnostics["field_summary"] = {
            "qpsi_i": _array_stats(qpsi_i),
            "volp_i": _array_stats(volp_i),
            "dv": _array_stats(dv),
            "te": _array_stats(te),
            "ner": _array_stats(ner),
            "ip_ma": _array_stats(ip_ma),
            "ip_neo_ma": _array_stats(ip_neo_ma),
        }
        return zeff, times, diagnostics


    # Consistency check:
    doConsistencyCheck(zeff, times)

    return zeff, times


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Python port of zeff_neo IDL routine")
    parser.add_argument("--shot", type=int, default = 1110316020)#1120815026, help="C-Mod shot number")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=1.5)
    parser.add_argument("--n-zeff", type=int, default=8)
    parser.add_argument("--zeff-max", type=float, default=4.0)
    parser.add_argument("--zion", type=float, default=1.0)
    parser.add_argument("--plot", default=True, action="store_true")
    parser.add_argument("--verbose", default=True,  action="store_true")
    parser.add_argument("--strict-diagnostics", action="store_true")
    parser.add_argument("--save-plots", type=str, default='../output_plots')
    args = parser.parse_args()

    out = zeff_neo(
        args.shot,
        zion=args.zion,
        zeff_max=args.zeff_max,
        n_zeff=args.n_zeff,
        dt=args.dt,
        trange=(args.tmin, args.tmax),
        plot=args.plot,
        verbose=args.verbose,
        strict_diagnostics=args.strict_diagnostics,
        save_plots=args.save_plots,
    )
    if args.strict_diagnostics:
        zeff, times, diagnostics = out
    else:
        zeff, times = out

    print("times")
    print(times)
    print("zeff")
    print(zeff)

    if args.strict_diagnostics:
        print("diagnostics")
        print(
            {
                "n_bins": len(diagnostics["diag_bins"]),
                "grid_shapes": diagnostics["grid_shapes"],
                "diag_summary": diagnostics["diag_summary"],
                "field_summary": diagnostics["field_summary"],
            }
        )


if __name__ == "__main__":
    main()
    print('Done')