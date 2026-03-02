from sys import path
path.append('/home/rianc/Documents/Synthetic_Mirnov/C-Mod/')
from get_Cmod_Data import __loadData
from header_Cmod import plt, np, correct_Bode, __doFilter

from scipy.interpolate import griddata
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

def visualize_bpk_contours(bp_k, time_range=None, freq_range=[50e3, 150e3], 
                          n_components=3, save_path=None, plot_timeseries=True):
    """
    Visualize spatial structure of bp_k data using SVD decomposition.
    
    Extracts dominant spatial modes from bandpass-filtered magnetic coil data
    and plots them as contours over theta-phi coordinates.
    
    Args:
        bp_k: BP_K object containing sensor data
        time_range: [t_start, t_end] in seconds (None = use all)
        freq_range: [f_low, f_high] bandpass filter frequencies in Hz
        n_components: Number of SVD components to plot
        save_path: Optional path to save figure
        plot_timeseries: If True, create diagnostic plot of filtered timeseries
    """
    
    # Extract time indices
    if time_range is None:
        time_indices = np.arange(len(bp_k.time))
    else:
        time_indices = np.where((bp_k.time >= time_range[0]) & 
                               (bp_k.time <= time_range[1]))[0]
    
    if len(time_indices) == 0:
        raise ValueError(f"No data found in time range {time_range}")
    
    # Calculate theta and phi for each sensor
    R = np.array(bp_k.R)
    Z = np.array(bp_k.Z)
    Phi = np.array(bp_k.Phi)
    
    r_mag = 0.68  # C-Mod magnetic axis (approximate)
    theta = np.arctan2(Z, R - r_mag)
    phi = Phi * np.pi / 180.0  # Convert to radians
    
    # Extract data for time range: (n_sensors, n_time)
    data = bp_k.data[:, time_indices]
    n_sensors, n_time = data.shape
    
    print(f"Processing {n_sensors} sensors over {n_time} time points")
    print(f"Time range: {bp_k.time[time_indices[0]]:.4f} - {bp_k.time[time_indices[-1]]:.4f} s")
    
    # Design bandpass filter
    fs = 1.0 / (bp_k.time[1] - bp_k.time[0])  # Sampling frequency
    nyq = 0.5 * fs
    low = freq_range[0] / nyq
    high = freq_range[1] / nyq
    
    # # Ensure filter frequencies are valid
    # low = np.clip(low, 1e-5, 0.99)
    # high = np.clip(high, 1e-5 + 0.01, 0.99)
    
    # b, a = butter(4, [low, high], btype='band')
    
    # Apply bandpass filter to each sensor
    data_filtered = np.zeros_like(data)
    for i in range(n_sensors):
        try:
            # data_filtered[i, :] = filtfilt(b, a, data[i, :])
            data_filtered[i, :] = __doFilter(data[i, :], bp_k.time[time_indices],
                                             HP_Freq=freq_range[0],
                                             LP_Freq=freq_range[1])[0]
        except Exception as e:
            print(f"Warning: filtering failed for sensor {i}: {e}")
            data_filtered[i, :] = data[i, :]
    
    print(f"Applied bandpass filter: {freq_range[0]/1e3:.1f} - {freq_range[1]/1e3:.1f} kHz")
    
    # Diagnostic plot: show filtered timeseries
    if plot_timeseries:
        ts_save_path = save_path.replace('.pdf', '_timeseries.pdf') if save_path else None
        plot_filtered_timeseries(bp_k, time_indices, data_filtered, theta, phi, 
                                n_sensors_plot=2, save_path=ts_save_path)
    
    # Perform SVD: data_filtered = U @ S @ Vh
    # U: (n_sensors, n_sensors) - spatial modes
    # S: (min(n_sensors, n_time),) - singular values
    # Vh: (n_time, n_time) - temporal modes
    U, S, Vh = np.linalg.svd(data_filtered, full_matrices=False)
    
    # Normalize singular values for display
    S_normalized = S / S[0] * 100  # As percentage of dominant mode
    
    print(f"SVD decomposition complete")
    print(f"Top 5 singular values (% of max): {S_normalized[:5]}")
    
    # Create subplot grid
    n_plots = min(n_components, len(S))
    n_rows, n_cols = __optimal_subplot_grid(n_plots)
    
    plt.close('BP_K_SVD_Contours')
    fig, axes = plt.subplots(n_rows, n_cols, num='BP_K_SVD_Contours',
                            figsize=(5*n_cols, 4*n_rows), 
                            tight_layout=True, sharex=True, sharey=True)
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Create interpolation grid
    phi_grid = np.linspace(phi.min(), phi.max(), 100)
    theta_grid = np.linspace(theta.min(), theta.max(), 100)
    Phi_grid, Theta_grid = np.meshgrid(phi_grid, theta_grid)
    
    # Pre-compute all mode grids to determine symmetric color limits
    mode_grids = []
    for comp_idx in range(n_plots):
        spatial_mode = U[:, comp_idx]
        mode_grid = griddata((phi, theta), spatial_mode, 
                            (Phi_grid, Theta_grid), method='linear')
        mode_grids.append(mode_grid)
    
    # Compute symmetric color limits based on all modes
    all_values = np.concatenate([grid[~np.isnan(grid)].flatten() for grid in mode_grids])
    vmax = np.max(np.abs(all_values)) * 0.5
    vmin = -vmax
    
    # Plot each component with unified color scale
    for comp_idx in range(n_plots):
        ax = axes[comp_idx]
        
        # Use pre-computed mode grid
        mode_grid = mode_grids[comp_idx]
        
        # Plot filled contours with symmetric limits
        levels = 20
        c = ax.contourf(Phi_grid * 180/np.pi, Theta_grid * 180/np.pi, 
                       mode_grid, levels=levels, cmap='RdBu_r', 
                       vmin=vmin, vmax=vmax)
        
        # Overlay sensor locations
        ax.scatter(Phi, theta * 180/np.pi, 
                  c='black', s=30, marker='x', linewidths=1.5, 
                  alpha=0.7, zorder=10)
        
        # Add colorbar
        cbar = fig.colorbar(c, ax=ax, label='Mode Amplitude [arb]')
        
        # Title with singular value info
        ax.set_title(f'SVD Mode {comp_idx+1}\n' + 
                    f'($\sigma$ = {S_normalized[comp_idx]:.1f}\% of max)',
                    fontsize=10, fontweight='bold')
        
        ax.set_rasterization_zorder(0)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    # Set axis labels
    for ax in axes[::n_cols]:  # Left column
        if ax.get_visible():
            ax.set_ylabel(r'$\theta$ [deg]', fontsize=11, fontweight='bold')
    
    for ax in axes[-n_cols:]:  # Bottom row
        if ax.get_visible():
            ax.set_xlabel(r'$\phi$ [deg]', fontsize=11, fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'BP_K Spatial Modes (Shot {bp_k.shotno})\n' + 
                f'Time: {bp_k.time[time_indices[0]]:.3f}-{bp_k.time[time_indices[-1]]:.3f} s, ' +
                f'Freq: {freq_range[0]/1e3:.0f}-{freq_range[1]/1e3:.0f} kHz',
                fontsize=12, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved figure to {save_path}")
    
    plt.show()
    
    return U, S, Vh, theta, phi


def plot_svd_spectrum(S, n_display=20, save_path=None):
    """
    Plot the singular value spectrum to see mode importance.
    
    Args:
        S: Singular values from SVD
        n_display: Number of singular values to display
        save_path: Optional path to save figure
    """
    plt.close('SVD_Spectrum')
    fig, (ax1, ax2) = plt.subplots(1, 2, num='SVD_Spectrum', 
                                   figsize=(10, 4), tight_layout=True)
    
    n_plot = min(n_display, len(S))
    indices = np.arange(1, n_plot + 1)
    S_norm = S[:n_plot] / S[0] * 100
    
    # Linear scale
    ax1.bar(indices, S_norm, color='steelblue', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Mode Number', fontweight='bold')
    ax1.set_ylabel('Relative Strength [\% of max]', fontweight='bold')
    ax1.set_title('SVD Singular Value Spectrum', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(indices, S_norm, 'o-', color='steelblue', 
                linewidth=2, markersize=6, markeredgecolor='black', markeredgewidth=1)
    ax2.set_xlabel('Mode Number', fontweight='bold')
    ax2.set_ylabel('Relative Strength [\% of max]', fontweight='bold')
    ax2.set_title('SVD Spectrum (Log Scale)', fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved spectrum plot to {save_path}")
    
    plt.show()


def plot_filtered_timeseries(bp_k, time_indices, data_filtered, theta, phi, 
                             n_sensors_plot=6, save_path=None):
    """
    Plot filtered time-series data for a subset of sensors as a sanity check.
    
    Args:
        bp_k: BP_K object containing sensor data and metadata
        time_indices: Indices of time window used
        data_filtered: Filtered data array (n_sensors, n_time)
        theta: Theta coordinates for each sensor
        phi: Phi coordinates for each sensor
        n_sensors_plot: Number of sensors to plot (evenly spaced)
        save_path: Optional path to save figure
    """
    n_sensors, n_time = data_filtered.shape
    time = bp_k.time[time_indices]
    
    # Select evenly spaced sensors for visualization
    if n_sensors_plot != 2:
        sensor_indices = np.linspace(0, n_sensors-1, n_sensors_plot, dtype=int)
    else: sensor_indices = [26,27]  # specifically plot these two sensors for now (bp2t_abk, bp2t_ghk )
    # Create subplot grid
    n_rows, n_cols = __optimal_subplot_grid(n_sensors_plot)
    
    plt.close('BP_K_Filtered_Timeseries')
    fig, axes = plt.subplots(n_rows, n_cols, num='BP_K_Filtered_Timeseries',
                            figsize=(4*n_cols, 3*n_rows), 
                            tight_layout=True, sharex=True)
    
    # Handle single subplot case
    if n_sensors_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each selected sensor
    for plot_idx, sensor_idx in enumerate(sensor_indices):
        ax = axes[plot_idx]
        
        # Get sensor data
        signal = data_filtered[sensor_idx, :]
        
        # Plot time series
        ax.plot(time * 1e3, signal * 1e3, 'b-', linewidth=1.5, alpha=0.8)
        ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.3)
        
        # Add sensor info to title
        theta_deg = theta[sensor_idx] * 180 / np.pi
        phi_deg = phi[sensor_idx] * 180 / np.pi
        ax.set_title(f'Sensor {bp_k.names[sensor_idx]}\n' + 
                    r'$\theta$=' + f'{theta_deg:.1f}°, ' + 
                    r'$\phi$=' + f'{phi_deg:.1f}°',
                    fontsize=9)
        
        # Compute RMS for annotation
        rms = np.sqrt(np.mean(signal**2))
        ax.text(0.02, 0.98, f'RMS: {rms*1e3:.2f} mT/s', 
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Signal [mT/s]', fontsize=9)
    
    # Hide unused subplots
    for j in range(n_sensors_plot, len(axes)):
        axes[j].set_visible(False)
    
    # Set x-axis labels for bottom row
    for ax in axes[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel('Time [ms]', fontsize=9)
    
    # Overall title
    fig.suptitle(f'Filtered Time-Series (Shot {bp_k.shotno})\n' + 
                f'Time: {time[0]:.3f}-{time[-1]:.3f} s',
                fontsize=11, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', transparent=True)
        print(f"Saved timeseries plot to {save_path}")
    
    plt.show()


def __optimal_subplot_grid(num_datasets: int):
    """Find optimal subplot grid dimensions."""
    if num_datasets <= 0:
        return (1, 1)
    
    sqrt_j = int(np.sqrt(num_datasets))
    
    best_waste = float('inf')
    best_m, best_n = 1, num_datasets
    
    for m in range(max(1, sqrt_j - 2), sqrt_j + 3):
        n = int(np.ceil(num_datasets / m))
        waste = m * n - num_datasets
        aspect_ratio_penalty = abs(m - n) * 0.01
        total_cost = waste + aspect_ratio_penalty
        
        if total_cost < best_waste:
            best_waste = total_cost
            best_m, best_n = m, n
    
    return (best_m, best_n)


if __name__ == '__main__':
    shotno = 1160930034#1160714026
    # Load bp_k data
    # bp_k = __loadData(1160930034,pullData='bp_k')['bp_k']
    
    bp_k = __loadData(shotno,pullData='bp_k')['bp_k']
    # Example: visualize modes during a specific time window
    U, S, Vh, theta, phi = visualize_bpk_contours(
        bp_k,
        time_range=[0.974,0.9795],  # 200 ms window
        # time_range=[1.374,1.3793],
        freq_range=[1e3, 20e3],  # 50-200 kHz
        # freq_range=[15e3, 30e3],  # 50-200 kHz  
        n_components=6,  # Plot first 6 modes
        save_path='output_plots/bpk_svd_contours_%d.pdf'%shotno
    )
    
    # Plot singular value spectrum
    plot_svd_spectrum(S, n_display=20, 
                     save_path='output_plots/bpk_svd_spectrum_%d.pdf'%shotno)
    
    # # Plot filtered timeseries for sanity check
    # plot_filtered_timeseries(
    #     bp_k, 
    #     time_indices=np.arange(100, 200),  # Example: middle of the time range
    #     data_filtered=data_filtered,  # Use the last filtered data
    #     theta=theta, 
    #     phi=phi, 
    #     n_sensors_plot=6, 
    #     save_path='output_plots/bpk_filtered_timeseries.pdf'
    # )
    
    print('\nSVD Analysis Complete')
    print(f'U shape (spatial modes): {U.shape}')
    print(f'S shape (singular values): {S.shape}')
    print(f'Vh shape (temporal modes): {Vh.shape}')