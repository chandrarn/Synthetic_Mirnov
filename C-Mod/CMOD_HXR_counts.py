"""
CMOD HXR Counts Analysis - Python Port
Analyzes hard X-ray counts from C-Mod tokamak with pulse height analysis
 
Ported from Matlab: /home/wallaceg/matlab/densitylimit/CMOD_HXR_counts_no_plot.m



Example usage:
    res, t, e, channels, f, M = CMOD_HXR_counts(1080219008, 
                                                 e=np.linspace(0, 300, 100), 
                                                 t=np.linspace(0, 2, 100), 
                                                 c_shield=1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata
from scipy import stats
import warnings
from get_Cmod_Data import openTree

warnings.filterwarnings('ignore')


def CMOD_HXR_counts(shot, ch=np.arange(9, 24), t=np.arange(.5, 1.5, .1), e=[40, 200], data_dict=None, plot_Min=None, 
                   plot_Max=None, plot_ET=True, movie_ET=False, c_shield=0, 
                   c_detector=0, analysis='keV', minDt=0.0, debug=True, debug_bins=False,
                   savePlot='', force_two_rows=False):
    """
    Extract and process HXR count data from C-Mod tokamak.
    
    Parameters
    ----------
    shot : int
        Shot number
    ch : list or array, optional
        Channel numbers (default 1-32)
    t : float or array
        Time axis: single number for binning count, or array of time points (default 200)
    e : float or array
        Energy axis: single number for binning count, or array of energy points (default 100)
    data_dict : dict, optional
        Pre-loaded data dictionary with keys as channel numbers and values containing 't' and 'V'
    plot_Min : float, optional
        Minimum value for plotting
    plot_Max : float, optional
        Maximum value for plotting
    plot_ET : bool
        Whether to create E-T plots (default True)
    movie_ET : bool
        Whether to create movie frames (default False)
    c_shield : int
        Shield compensation: 0=none, 1=standard, 2=advanced (default 0)
    c_detector : bool
        Apply detector response compensation (default False)
    analysis : str
        Analysis type: 'keV', 'new', or 'old' (default 'keV')
    minDt : float
        Minimum time difference threshold (default 0.0)
    debug : bool
        Print debug messages (default True)
    debug_bins : bool
        Print per-channel gridding diagnostics (default False)
    force_two_rows : bool
        If True, arrange E-T plots in two rows with as many columns as needed (default False)
    
    Returns
    -------
    res : ndarray
        Binned 3D array (time x energy x channel)
    t : ndarray
        Time axis
    e : ndarray
        Energy axis
    channels : list
        Channel numbers processed
    f : ndarray
        Compensation factors
    M : list
        Movie frames (if movie_ET=True, else empty list)
    """
    
    if ch is None:
        ch = np.arange(1, 33)  # channels 1-32
    else:
        ch = np.atleast_1d(ch)
    
    channels = ch.copy()
    
    # Energy calibration coefficients for all 32 channels
    V60H = np.array([0.377, 0.422, 0.382, 0.402, 0.392, 0.372, 0.372, 0.362, 
                     0.382, 0.377, 0.377, 0.392, 0.407, 0.414, 0.432, 0.397,
                     0.372, 0.377, 0.392, 0.387, 0.387, 0.392, 0.387, 0.387, 
                     0.412, 0.422, 0.384, 0.404, 0.389, 0.392, 0.407, 0.372])
    
    V122H = np.array([0.844, 0.925, 0.859, 0.884, 0.869, 0.829, 0.824, 0.809, 
                      0.829, 0.829, 0.824, 0.869, 0.889, 0.925, 0.950, 0.864,
                      0.829, 0.829, 0.889, 0.859, 0.849, 0.869, 0.854, 0.854, 
                      0.889, 0.929, 0.854, 0.869, 0.864, 0.869, 0.905, 0.834])
    
    Ecal = np.zeros((32, 2))
    Ecal[:, 0] = (122.0 - 60.0) / (V122H - V60H)
    Ecal[:, 1] = 60.0 - Ecal[:, 0] * V60H
    
    # Initialize data structure
    data = {}
    for ch_num in channels:
        data[ch_num] = {'time': np.array([]), 'energy': np.array([]), 'fs': np.array([])}
    
    # Load pre-existing data if provided
    if data_dict is not None:
        for ch_num in channels:
            if ch_num in data_dict and 'V' in data_dict[ch_num] and 't' in data_dict[ch_num]:
                data[ch_num]['time'] = np.asarray(data_dict[ch_num]['t']).flatten()
                # Apply Ecal calibration coefficients to convert voltage to energy
                data[ch_num]['energy'] = Ecal[ch_num - 1, 0] * np.asarray(data_dict[ch_num]['V']).flatten() + Ecal[ch_num - 1, 1]
    
    # Open MDSplus connection
    # try:
    conn = openTree(shot)
    
    # Get shielding parameters
    try:
        thick = conn.get(r'\LH::TOP.HXR.PARAMS:SSTHICK').data()
        if thick is None:
            thick = 0.0
    except:
        thick = 0.0
        if debug:
            print("Could not retrieve HXR shielding thickness")
    
    aluminum_thickness = 0.001
    detector_thickness = 0.002
    iron_thickness = thick * 2.54 / 100.0
    
    tlim = [np.inf, -np.inf]
    elim = [np.inf, -np.inf]
    
    # Load channel data
    for ch_num in channels:
        if len(data[ch_num]['time']) == 0:
            # try:
            if analysis == 'new':
                if debug:
                    print(f'Loading channel {ch_num} from MDSplus with new PHA analysis...')
                energy_data = conn.get(rf'\LH::TOP.HXR.RESULTS.MAXIMA2:CH{ch_num:02d}').data()
                time_data = conn.get(rf'dim_of(\LH::TOP.HXR.RESULTS.MAXIMA2:CH{ch_num:02d})').data()
                if energy_data is not None and time_data is not None:
                    # Apply Ecal scaling for energy calibration
                    data[ch_num]['energy'] = Ecal[ch_num - 1, 0] * energy_data + Ecal[ch_num - 1, 1]
                    data[ch_num]['time'] = time_data
            elif analysis == 'keV':
                if debug:
                    print(f'Loading channel {ch_num} from MDSplus with new PHA analysis (keV)...')
                energy_data = conn.get(rf'\LH::TOP.HXR.RESULTS.MAXIMA2:CH{ch_num:02d}_keV').data()
                time_data = conn.get(rf'dim_of(\LH::TOP.HXR.RESULTS.MAXIMA2:CH{ch_num:02d})').data()
                if energy_data is not None and time_data is not None:
                    # Already in keV, no scaling needed
                    data[ch_num]['energy'] = energy_data
                    data[ch_num]['time'] = time_data
            else:  # 'old' analysis
                if debug:
                    print(f'Loading channel {ch_num} from MDSplus with old PHA analysis...')
                value = conn.get(rf'\LH::TOP.HXR.RESULTS.MAXIMA:CH{ch_num:02d}').data()
                time_data = conn.get(rf'dim_of(\LH::TOP.HXR.RESULTS.MAXIMA:CH{ch_num:02d})').data()
                try:
                    baseline = conn.get(rf'\LH::TOP.HXR.RESULTS.BASELINEOLD:CH{ch_num:02d}').data()
                except:
                    try:
                        baseline = conn.get(rf'\LH::TOP.HXR.RESULTS.BASELINE').data()
                    except:
                        baseline = 0
                        print('WARNING: No Baseline Found')
                    
                if value is not None and baseline is not None:
                    data[ch_num]['time'] = time_data
                    data[ch_num]['energy'] = Ecal[ch_num - 1, 0] * (value[:] - baseline) + Ecal[ch_num - 1, 1]
            
            # except Exception as ex:
            #     if debug:
            #         print(f'Could not load channel {ch_num}: {str(ex)}')
        else:
            if debug:
                print(f'Using pre-loaded data for channel {ch_num}')
        
        # Handle time remapping if needed
        if isinstance(t, np.ndarray) and t.shape[0] == 2:
            f_interp = interp1d(t[0, :], t[1, :], kind='linear', bounds_error=False, fill_value='extrapolate')
            time_remapped = f_interp(data[ch_num]['time'])
            sort_idx = np.argsort(time_remapped)
            data[ch_num]['time'] = time_remapped[sort_idx]
            data[ch_num]['energy'] = data[ch_num]['energy'][sort_idx]
        
        # Apply minimum time difference filter
        if len(data[ch_num]['time']) > 0:
            dt = np.concatenate(([100], np.diff(data[ch_num]['time'])))
            mask = dt > minDt
            data[ch_num]['energy'] = data[ch_num]['energy'][mask]
            data[ch_num]['time'] = data[ch_num]['time'][mask]
            data[ch_num]['fs'] = np.ones_like(data[ch_num]['time'])
        
        # Track time and energy limits
        if len(data[ch_num]['time']) > 0:
            tlim[0] = min(tlim[0], np.min(data[ch_num]['time']))
            tlim[1] = max(tlim[1], np.max(data[ch_num]['time']))
            elim[0] = min(elim[0], np.min(data[ch_num]['energy']))
            elim[1] = max(elim[1], np.max(data[ch_num]['energy']))

    # except Exception as ex:
    #     if debug:
    #         print(f"Error accessing MDSplus data: {str(ex)}")
    
    # Create time and energy axes
    if np.isscalar(t):
        t_axis = np.linspace(tlim[0], tlim[1], int(t))
    else:
        t_axis = np.asarray(t).flatten()
    
    if np.isscalar(e):
        e_axis = np.linspace(elim[0], elim[1], int(e))
    else:
        e_axis = np.asarray(e).flatten()
    
    t = t_axis
    e = e_axis
    
    # Shield compensation
    if c_shield == 1:
        if debug:
            print('Compensating shielding...')
        
        # Try to load attenuation data
        try:
            g_304 = np.loadtxt('g_304.txt')
            rho_304 = 8.0
            att_aluminum = np.ones_like(e_axis)  # Placeholder if attlinyp not available
            att_iron = np.interp(e_axis, g_304[:, 0] * 1e3, g_304[:, 6] * rho_304 * 100, left=0, right=0)
        except:
            if debug:
                print("Could not load g_304.txt for shielding compensation")
            att_aluminum = np.ones_like(e_axis)
            att_iron = np.ones_like(e_axis)
        
        fs = 1.0 / np.exp(-att_aluminum * aluminum_thickness - att_iron * iron_thickness)
    else:
        fs = np.ones_like(e_axis)
    
    # Advanced shield compensation
    if c_shield == 2:
        if debug:
            print('Compensating shielding (advanced)...')
        try:
            g_304 = np.loadtxt('g_304.txt')
            rho_304 = 8.0
            for ch_num in channels:
                if len(data[ch_num]['energy']) > 0:
                    att_aluminum = np.ones_like(data[ch_num]['energy'])
                    att_iron = np.interp(data[ch_num]['energy'], g_304[:, 0] * 1e3, 
                                        g_304[:, 6] * rho_304 * 100, left=0, right=0)
                    data[ch_num]['fs'] = 1.0 / np.exp(-att_aluminum * aluminum_thickness - 
                                                      att_iron * iron_thickness)
                else:
                    data[ch_num]['fs'] = np.ones_like(data[ch_num]['time'])
        except:
            if debug:
                print("Could not apply advanced shielding compensation")
            for ch_num in channels:
                data[ch_num]['fs'] = np.ones_like(data[ch_num]['time'])
    else:
        for ch_num in channels:
            data[ch_num]['fs'] = np.ones_like(data[ch_num]['time'])
    
    # Detector compensation
    if c_detector:
        if debug:
            print('Compensating detector response...')
        try:
            g_CZT = np.loadtxt('g_CZT.txt')
            rho_CZT = 5.78
            abs_dec = np.interp(e_axis, g_CZT[:, 0] * 1e3, g_CZT[:, 6] * rho_CZT * 100, left=0, right=0)
            fd = 1.0 / (1.0 - np.exp(-abs_dec * detector_thickness))
        except:
            if debug:
                print("Could not load g_CZT.txt for detector compensation")
            fd = np.ones_like(e_axis)
    else:
        fd = np.ones_like(e_axis)
    
    f = fs * fd
    F = np.tile(f.reshape(1, -1), (len(t_axis), 1))
    
    # Bin data into 2D histograms
    res = np.zeros((len(t_axis), len(e_axis), len(channels)))
    
    for idx, ch_num in enumerate(channels):
        if debug:
            print(f'Binning channel {ch_num}...')
        
        if len(data[ch_num]['time']) > 0:
            grid_data = qgriddata(
                data[ch_num]['time'],
                data[ch_num]['energy'],
                data[ch_num]['fs'],
                t_axis,
                e_axis,
                operation='sum',
            )
            res[:, :, idx] = grid_data * F

            if debug_bins:
                print_channel_bin_debug(
                    ch_num=ch_num,
                    raw_t=data[ch_num]['time'],
                    raw_e=data[ch_num]['energy'],
                    grid_data=grid_data,
                    t_axis=t_axis,
                    e_axis=e_axis,
                )
    
    # Plotting
    M = []
    if plot_ET:
        cmax = np.nanmax(res)
        if plot_Min is None:
            plot_Min = 0
        if plot_Max is None:
            plot_Max = np.log10(cmax + 1)
        
        n_channels = len(channels)
        if force_two_rows and n_channels > 1:
            n_rows = 2
            n_cols = int(np.ceil(n_channels / n_rows))
            figsize = (max(12, 2.4 * n_cols), 8)
        else:
            n_cols = int(np.ceil(np.sqrt(n_channels)))
            n_rows = int(np.ceil(n_channels / n_cols))
            figsize = (12, 10)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True, layout='constrained')
        if n_channels == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ch_num in enumerate(channels):
            ax = axes[idx]
            img_data = np.log10(res[:, :, idx].T + 1)
            im = ax.contourf(t_axis, e_axis, img_data, levels=20, vmin=plot_Min, vmax=plot_Max, zorder=-5)
            ax.set_title((f'{shot} ' if idx == 0 else '')+f'Ch: {ch_num:02d}')
            if idx % n_cols == 0:
                ax.set_ylabel('Energy [keV]')
            if idx >= n_channels - n_cols:
                ax.set_xlabel('Time [s]')
            if (idx + 1) % n_cols == 0:
                plt.colorbar(im, ax=ax, label='log10(Counts)')
            ax.set_rasterization_zorder(-1)  # Ensure contourf is below gridlines
        # Hide unused subplots
        for idx in range(n_channels, len(axes)):
            axes[idx].axis('off')
        

        # plt.tight_layout()
        if savePlot:
            plt.savefig(savePlot+f'CMOD_HXR_counts_{shot}.pdf', transparent=True)
            if debug: 
                print(f"Saved plot to {savePlot}CMOD_HXR_counts_{shot}.pdf")
        plt.show(block=False)
    
    if movie_ET:
        cmax = np.nanmax(res)
        if plot_Min is None:
            plot_Min = 0
        if plot_Max is None:
            plot_Max = np.log10(cmax + 1)
        
        fig = plt.figure()
        for idx, ch_num in enumerate(channels):
            if debug:
                print(f'Movie frame for channel {ch_num}...')
            
            plt.clf()
            img_data = np.log10(res[:, :, idx].T + 1)
            plt.contourf(t_axis, e_axis, img_data, levels=20, vmin=plot_Min, vmax=plot_Max)
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (keV)')
            plt.title(f'Ch: {ch_num:02d}')
            plt.colorbar()
            plt.draw()
            plt.pause(0.1)
            M.append(fig.canvas.copy_from_bbox(fig.bbox))
        
        plt.close(fig)
    
    return res, t, e, channels, f, M


def _centers_to_edges(centers):
    centers = np.asarray(centers).ravel()
    if centers.size == 1:
        half_width = 0.5
        return np.array([centers[0] - half_width, centers[0] + half_width], dtype=float)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate(([first], mids, [last]))


def print_channel_bin_debug(ch_num, raw_t, raw_e, grid_data, t_axis, e_axis):
    raw_t = np.asarray(raw_t).ravel()
    raw_e = np.asarray(raw_e).ravel()
    grid_data = np.asarray(grid_data)
    t_axis = np.asarray(t_axis).ravel()
    e_axis = np.asarray(e_axis).ravel()

    t_edges = _centers_to_edges(t_axis)
    e_edges = _centers_to_edges(e_axis)

    finite = np.isfinite(raw_t) & np.isfinite(raw_e)
    in_t = (raw_t >= t_edges[0]) & (raw_t <= t_edges[-1])
    in_e = (raw_e >= e_edges[0]) & (raw_e <= e_edges[-1])
    in_bounds = finite & in_t & in_e

    nz = np.argwhere(grid_data > 0)
    nz_count = nz.shape[0]
    total_bins = grid_data.size

    msg = [
        f"[ch {ch_num:02d}] raw={raw_t.size}, finite={np.count_nonzero(finite)}, in_bounds={np.count_nonzero(in_bounds)}",
        f"[ch {ch_num:02d}] nonzero_bins={nz_count}/{total_bins}",
    ]

    if nz_count > 0:
        t_idx = nz[:, 0]
        e_idx = nz[:, 1]
        edge_hits = np.count_nonzero(
            (t_idx == 0) | (t_idx == (t_axis.size - 1)) |
            (e_idx == 0) | (e_idx == (e_axis.size - 1))
        )
        msg.append(
            f"[ch {ch_num:02d}] t_bin_span={t_idx.min()}..{t_idx.max()} ({t_axis[t_idx.min()]:.4g}..{t_axis[t_idx.max()]:.4g})"
        )
        msg.append(
            f"[ch {ch_num:02d}] e_bin_span={e_idx.min()}..{e_idx.max()} ({e_axis[e_idx.min()]:.4g}..{e_axis[e_idx.max()]:.4g})"
        )
        msg.append(f"[ch {ch_num:02d}] edge_bin_fraction={edge_hits / nz_count:.3f}")

    print("\n".join(msg))


def qgriddata(px, py, data, x_axis, y_axis, operation='sum'):
    """
    Grid data similar to MATLAB's accumarray-based gridding.
    
    Parameters
    ----------
    px : array
        X coordinates of data points
    py : array
        Y coordinates of data points
    data : array
        Data values at (px, py)
    x_axis : ndarray (1D)
        X bin centers (time axis)
    y_axis : ndarray (1D)
        Y bin centers (energy axis)
    operation : str
        Operation to apply: 'sum', 'mean', 'count' (default 'sum')
    
    Returns
    -------
    res : ndarray
        Gridded data
    """
    
    px = np.asarray(px).ravel()
    py = np.asarray(py).ravel()
    data = np.asarray(data).ravel()
    x_axis = np.asarray(x_axis).ravel()
    y_axis = np.asarray(y_axis).ravel()

    x_edges = _centers_to_edges(x_axis)
    y_edges = _centers_to_edges(y_axis)

    valid = np.isfinite(px) & np.isfinite(py)
    if operation != 'count':
        valid &= np.isfinite(data)
    if not np.any(valid):
        return np.zeros((x_axis.size, y_axis.size), dtype=float)

    pxv = px[valid]
    pyv = py[valid]
    wv = data[valid] if operation != 'count' else None

    if operation == 'sum':
        hist, _, _ = np.histogram2d(pxv, pyv, bins=[x_edges, y_edges], weights=wv)
        return hist

    if operation == 'count':
        hist, _, _ = np.histogram2d(pxv, pyv, bins=[x_edges, y_edges])
        return hist

    if operation == 'mean':
        sums, _, _ = np.histogram2d(pxv, pyv, bins=[x_edges, y_edges], weights=wv)
        counts, _, _ = np.histogram2d(pxv, pyv, bins=[x_edges, y_edges])
        with np.errstate(divide='ignore', invalid='ignore'):
            means = sums / counts
        means[~np.isfinite(means)] = 0.0
        return means

    raise ValueError(f"Unsupported operation '{operation}'. Use 'sum', 'count', or 'mean'.")


if __name__ == '__main__':
    # Example usage
    shot = 1110201006#1080219008 #1140613008#
    ch=np.arange(1, 32)
    t=np.arange(.5, 2, .05)
    e=np.arange(50,400, 10)
    analysis='old'
    plot_ET=True
    c_shield=0

    ch = np.arange(1,17,dtype=int)

    # Simple call with defaults
    res, t, e, channels, f, M = CMOD_HXR_counts(shot=shot, analysis=analysis, plot_ET=plot_ET,\
                                                 c_shield=c_shield, ch=ch, t=t, e=e, debug=True, debug_bins=True,\
                                                    savePlot='../output_plots/', force_two_rows=True)
    
    print(f"Processed shot {shot}")
    print(f"Result shape: {res.shape}")
    print(f"Channels: {channels}")
    print(f"Time range: {t[0]:.3f} - {t[-1]:.3f} s")
    print(f"Energy range: {e[0]:.1f} - {e[-1]:.1f} keV")
    print('Finished processing.')
