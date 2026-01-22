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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rc,cm
import matplotlib
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
try: matplotlib.use('TkAgg')
except: matplotlib.use('pdf')
plt.ion()

# Allow importing C-Mod utilities for geometry (optional)
import sys
sys.path.append('../C-Mod/')
try:
    from get_Cmod_Data import __loadData  # for bp_k geometry
    from mirnov_Probe_Geometry import Mirnov_Geometry as Mirnov_Geometry_C_Mod
except Exception:
    __loadData = None


##################################################################################################


# -------------------------
# Sensor ordering utilities
# -------------------------
def gen_sensor_ordering(sensor_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Return (sorted_names, original_names_array, order_indices) for a consistent ordering:
      1) BPXX_ABK (XX numeric)
      2) BPXX_GHK (XX numeric)
      3) BPXT_ABK (XX numeric)
      4) BPXT_GHK (XX numeric)
      5) BP_AA_BOT (AA letters) [all BOT first]
      6) BP_AA_TOP (AA letters)
    Case-insensitive matching; unknowns go to the end, lexicographic.
    """
    def _sensor_sort_key(name: str):
        u = str(name).upper()
        m = re.match(r'^BP(\d+)_([A-Z]+)$', u)  # BPXX_ABK/GHK
        if m:
            num = int(m.group(1)); suf = m.group(2)
            # Group by suffix first: all ABK (cat=0), then all GHK (cat=1)
            cat = 0 if suf == 'ABK' else 1 if suf == 'GHK' else 2
            return (0, cat, num, u)
        m = re.match(r'^BP(\d+)T_([A-Z]+)$', u)  # BPXT_ABK/GHK
        if m:
            num = int(m.group(1)); suf = m.group(2)
            # Group by suffix first: all ABK (cat=0), then all GHK (cat=1)
            cat = 0 if suf == 'ABK' else 1 if suf == 'GHK' else 2
            return (1, cat, num, u)
        m = re.match(r'^BP_([A-Z]{2})_((?:TOP|BOT))$', u)  # BP_AA_TOP/BOT
        if m:
            aa = m.group(1); pos = m.group(2)
            # Treat BOT and TOP as separate categories: all BOT first, then all TOP
            cat = 0 if pos == 'BOT' else 1
            return (2, cat, aa, u)
        # Fallback: put at end, keep lexicographic
        return (9, 9, u, u, u)

    _names = np.array(sensor_names).astype(str)
    order = [i for i, _ in sorted(enumerate(_names), key=lambda t: _sensor_sort_key(t[1]))]

    return _names[order], _names, order

##################################################################################################
##################################################################################################

# -------------------------
# Visualization utilities
# -------------------------
def visualize_dataset_regions(ds: xr.Dataset, time_indices: np.ndarray, 
                              F_mode_vars: List[str], regions: List[List[Tuple[np.ndarray, np.ndarray]]],
                              t_window_width: int = 4, 
                              sensor_to_plot: str = 'BP01_ABK', 
                              save_path: Optional[str] = None,
                              mode_region_fLim: Optional[List[float]] = None) -> None:
    """
    Visualize the frequency-time regions used for data extraction from a dataset.
    
    Creates a plot showing the real spectrogram for one sensor with colored rectangles
    indicating the mode regions (frequency bands at each time point) that are used
    for feature extraction.
    
    Args:
        ds: xarray Dataset containing spectrogram data
        time_indices: Array of time indices used for sampling
        F_mode_vars: List of mode frequency variable names (e.g., ['F_Mode_0', 'F_Mode_1'])
        Width in timepoints to use for rectangle width (default: 4)
        regions: List[mode][time_idx] of (freq_inds, sensor_inds) tuples
        sensor_to_plot: Name of sensor to visualize (default: 'BP01_ABK')
        save_path: Optional path to save the figure (e.g., '../output_plots/')
        mode_region_fLim: Optional [f_min, f_max] frequency limits in kHz for y-axis
    """
    # Check if sensor exists in dataset
    sensor_var = f"{sensor_to_plot}_real"
    if sensor_var not in ds.data_vars:
        print(f"Warning: Sensor '{sensor_to_plot}' not found in dataset. Available sensors:")
        available = [str(v).replace('_real', '') for v in ds.data_vars if str(v).endswith('_real') and 'Mode' not in str(v)]
        print(f"  {available[:5]}... ({len(available)} total)")
        if available:
            sensor_to_plot = available[0]
            sensor_var = f"{sensor_to_plot}_real"
            print(f"Using '{sensor_to_plot}' instead.")
        else:
            print("No sensors available for visualization.")
            return
    
    data = ds[sensor_var].values
    
    plt.close(f'Debug_{sensor_to_plot}_Real_Spectrogram')
    fig, ax = plt.subplots(1, 1, num=f'Debug_{sensor_to_plot}_Real_Spectrogram', 
                          tight_layout=True, figsize=(5, 3))
    
    # Plot spectrogram
    im = ax.pcolormesh(ds['time'].values * 1e3, ds['frequency'].values * 1e-3, 
                       data, cmap='viridis', zorder=-5)
    fig.colorbar(im, ax=ax, label=r'Synth Sig. [T/s]')
    
    # Collect valid regions for plotting
    # Stores valid regions as (time_idx, mode_idx, freq_inds, sensor_inds)
    store_indices = []
    for time_idx, ti in enumerate(time_indices):
        for mode_idx in range(len(F_mode_vars)):
            freq_inds, sensor_inds = regions[mode_idx][time_idx]
            if len(freq_inds) > 0:
                store_indices.append((time_idx, mode_idx, freq_inds, sensor_inds))
    
    # Loop over valid mode analysis regions and plot rectangular regions
    for (time_idx, mode_idx, freq_inds, sensor_inds) in store_indices:
        if len(freq_inds) > 0:  # Only plot if there are frequency indices
            # Define the rectangular region: time is single point, so use a small width around it
            time_val = float(ds['time'].values[time_indices[time_idx]] * 1e3)
            freq_min = ds['frequency'].values[freq_inds].min() * 1e-3
            freq_max = ds['frequency'].values[freq_inds].max() * 1e-3
            
            # Assume time bin width for rectangle width (or use a fixed small value)
            time_bin_width = t_window_width*(ds['time'].values[1] - ds['time'].values[0]) * 1e3 if len(ds['time']) > 1 else 0.1
            
            # Extract mode numbers for label
            mode_m = ds.attrs.get('mode_m', 0)
            mode_n = ds.attrs.get('mode_n', 0)
            m_ = mode_m[mode_idx] if np.ndim(mode_m) > 0 else mode_m
            n_ = mode_n[mode_idx] if np.ndim(mode_n) > 0 else mode_n
            
            # Draw rectangle
            rect = Rectangle((time_val - time_bin_width / 2, freq_min), 
                           time_bin_width, freq_max - freq_min,
                           linewidth=2, edgecolor=plt.get_cmap('tab10')(mode_idx), 
                           facecolor='none',
                           label=f'{F_mode_vars[mode_idx]}: {m_}/{n_}' if time_idx == 0 else None)
            ax.add_patch(rect)
    
    ax.legend(fontsize=8, handlelength=1.5)
    ax.set_xlim(ds['time'].values.min() * 1e3, ds['time'].values.max() * 1e3)
    ax.set_ylim(ds['frequency'].values.min() * 1e-3, ds['frequency'].values.max() * 1e-3)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_rasterization_zorder(-1)
    
    if mode_region_fLim is not None:
        ax.set_ylim(mode_region_fLim[0], mode_region_fLim[1])
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f'Debug_{sensor_to_plot}_Real_Spectrogram.pdf'),
                   transparent=True)
        print(f"Saved visualization to {os.path.join(save_path, f'Debug_{sensor_to_plot}_Real_Spectrogram.pdf')}")
    

    ##########################################################33
    # Plot all real, imaginary values for all sensors, for one timepoint in store_indices

    available = [str(v).replace('_real', '') for v in ds.data_vars if str(v).endswith('_real') and 'Mode' not in str(v)]
    real_region = np.array( [ds[f"{base}_real"].values[store_indices[-1][2], store_indices[-1][0]] for base in available] )
    imag_region = np.array( [ds[f"{base}_imag"].values[store_indices[-1][2], store_indices[-1][0]] for base in available] )

    plt.close('Debug_Real_Imag_Matrix')
    fig,ax=plt.subplots(1,2,num='Debug_Real_Imag_Matrix',tight_layout=True,figsize=(8,4),sharex=True,sharey=True)
    # Storeindices: valid regions as (time_idx, mode_idx, freq_inds, sensor_inds)
    im0 = ax[0].pcolormesh(store_indices[-1][3],ds['frequency'][store_indices[-1][2]]*1e-3,real_region.T,cmap='viridis')
    fig.colorbar(im0,ax=ax[0],label=r'Synth Sig. [T/s]')
    fig.suptitle('Mag/Arg Signal (Mode %s, Time %1.2e s)'%(F_mode_vars[store_indices[-1][1]],ds['time'][time_indices[store_indices[-1][0]]]))
    
    im1 = ax[1].pcolormesh(store_indices[-1][3],ds['frequency'][store_indices[-1][2]]*1e-3,imag_region.T,cmap='viridis')
    fig.colorbar(im1,ax=ax[1],label=r'$\phi_{sig}$ [rad]')
    # ax[1].set_title('Imag part of region')
    ax[0].set_ylabel('Frequency [kHz]')
    ax[0].set_xlabel('Sensor index')
    if save_path: fig.savefig(save_path+'Debug_Real_Imag_Matrix.pdf',transparent=True)





    plt.show()
############################################################

def visualize_cached_data(cfg: CacheConfig, y_m, y_n, X_ri,doDelta=False):
    """
    # Visualize cached data
    # Plot y_m, y_n traces vs sample
    # For two different y_n, y_m values, plot the distribution of real, imaginary values
    """

    # Plot y_m, y_n traces vs sample
    plt.close('Debug_Cached_y_m_n_Traces')
    fig, ax = plt.subplots(1, 1, num='Debug_Cached_y_m_n_Traces', tight_layout=True, figsize=(6, 4))
    ax.plot(np.arange(len(y_m)), y_m, label='Mode m')
    ax.plot(np.arange(len(y_n)), y_n, label='Mode n', alpha=.7)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Mode number')
    ax.legend(fontsize=8)
    ax.grid()
    if cfg.viz_save_path:
        fig.savefig(os.path.join(cfg.viz_save_path, 'Debug_Cached_y_m_n_Traces.pdf'),
                    transparent=True)
        print(f"Saved visualization to {os.path.join(cfg.viz_save_path, 'Debug_Cached_y_m_n_Traces.pdf')}")
    plt.show()

    ##############################################

    # for (1/1), (2/1), and then (2/1), (2/2) overplot the real, imaginary signals on two axes
    fig,ax=plt.subplots(2,2,num='Debug_Cached_Real_Imag_Distributions'+('_Delta' if doDelta else ''),\
                        tight_layout=True,figsize=(6,5),sharex=True,sharey=False)
    target_modes = [(1,1),(2,1),(2,1),(2,2)]

    for i,(m_target,n_target) in enumerate(target_modes):
        # Find indices matching target modes
        indices = np.where((y_m == m_target) & (y_n == n_target))[0]
        if len(indices) == 0:
            print(f"Warning: No samples found for mode {m_target}/{n_target}. Skipping.")
            continue
        
        # if len(indices) > 5: indices = indices[:5]  # limit to first 5 for clarity

        real_vals = X_ri[indices, :, 0]
        imag_vals = X_ri[indices, :, 1] 


        for ind,imag in enumerate(imag_vals):
            # Do delta, check for negative real part, trim imag to [0, 2pi]
            real_tmp = real_vals[ind] - real_vals[ind][0]
            imag -= imag[0] * (1 if doDelta else 0)

            imag[real_tmp < 0] += np.pi
            real_tmp[real_tmp < 0 ] *= -1
            imag %= 2*np.pi 

            ax[0, i//2].plot(imag,'-*', alpha=0.01, ms=5,
                                color=plt.get_cmap('tab10')(i),
                              label=f'Mode {m_target}/{n_target} (N={len(indices)})' if ind==0 else None)
            ax[1, i//2].plot(real_tmp,'-*', alpha=0.01, ms=5,
                                color=plt.get_cmap('tab10')(i),)
        
    for i in range(4):ax[np.unravel_index(i,(2,2))].grid()

    ax[1,0].set_xlabel('Sensor Index')
    ax[1,1].set_xlabel('Sensor Index')
    ax[0,0].set_ylabel('Angle Component ['+(r'$\Delta$' if doDelta else '')+'rad]')
    ax[1,0].set_ylabel('Magnitude Component ['+(r'$\Delta$' if doDelta else '')+'T]')
    ax[0,0].legend(fontsize=8)
    ax[0,1].legend(fontsize=8)
    
        
    
    if cfg.viz_save_path:
        fig.savefig(os.path.join(cfg.viz_save_path, fig.canvas.manager.get_window_title()+'.pdf'),
                    transparent=True)
        print(f"Saved visualization to {os.path.join(cfg.viz_save_path,  fig.canvas.manager.get_window_title()+ '.pdf')}")
    plt.show()
    print("Visualization of cached data complete.")
##############################################################################################
def visualize_contours(cfg: CacheConfig, y_m, y_n, X_ri,theta,phi,doDelta=False):
    # Plot reconstructed signal for a given mode (Real * cos(Imag)) over contours of theta, phi
    from scipy.interpolate import griddata
    
    plt.close('Debug_Cached_Contour_Plots'+('_Delta' if doDelta else ''))
    m,n = __optimal_subplot_grid(len(y_m))

    fig,ax=plt.subplots(m,n,num='Debug_Cached_Contour_Plots'+('_Delta' if doDelta else ''),\
                        tight_layout=True,figsize=(5,4),sharex=True,sharey=True)
    ax = ax.flatten() if m*n > 1 else [ax]
    theta *= 180.0/np.pi  # convert to degrees

    for i in range(len(y_m)):
        # Reconstruct signal
        real_part = X_ri[i,:,0] - (X_ri[i,0,0] if doDelta else 0)
        imag_part = X_ri[i,:,1] - (X_ri[i,0,1] if doDelta else 0)
        signal_reconstructed = real_part * np.cos(imag_part)

        # Create regular grid for interpolation
        phi_grid = np.linspace(phi.min(), phi.max(), 50)
        theta_grid = np.linspace(theta.min(), theta.max(), 50) 
        Phi_grid, Theta_grid = np.meshgrid(phi_grid, theta_grid)
        
        # Interpolate irregular data to regular grid
        signal_grid = griddata((phi, theta), signal_reconstructed, 
                              (Phi_grid, Theta_grid), method='linear')
        
        # Plot filled contours on regular grid
        c = ax[i].contourf(Phi_grid, Theta_grid, signal_grid, levels=20, cmap='viridis')
        # Overlay original sensor locations
        ax[i].scatter(phi, theta, c='red', s=20, marker='x', linewidths=1, alpha=0.5)
        fig.colorbar(c, ax=ax[i], label='Signal [T]')
        ax[i].set_title(f'Sample {i}: Mode {y_m[i]}/{y_n[i]}', fontsize=8)

        ax[i].set_rasterization_zorder(-1)
        # ax[i].set_aspect('equal', adjustable='box')
        ax[i].set_xlim(-360,0)
    # Hide unused subplots
    for j in range(len(y_m), len(ax)):
        ax[j].set_visible(False)

    for axi in ax[::n]:
        if axi.get_visible():
            axi.set_ylabel(r'$\theta$ [rad]')
    for axi in ax[-n:]:
        if axi.get_visible():
            axi.set_xlabel(r'$\phi$ [rad]')

    if cfg.viz_save_path:
        fig.savefig(os.path.join(cfg.viz_save_path, fig.canvas.manager.get_window_title()+'.pdf'),
                    transparent=True)
        print(f"Saved visualization to {os.path.join(cfg.viz_save_path,  fig.canvas.manager.get_window_title()+ '.pdf')}")  
    
    plt.show()
    print("Visualization of cached contour data complete.")
##############################################################################################
##############################################################################################

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
    visualize_first: bool = False  # Whether to visualize first dataset regions
    viz_save_path: Optional[str] = None  # Path to save visualization plots
    viz_sensor: str = 'BP01_ABK'  # Sensor to visualize
    viz_freq_lim: Optional[List[float]] = None  # Frequency limits [f_min, f_max] in kHz
    load_saved_data: bool = False  # Whether to load existing cached data if available
    t_window_width: int = 0  # Width in timepoints for visualization rectangles
    saveDataset: bool = True  # Whether to save the cached dataset to disk


def _get_dataset_filepaths(data_dir: str, n_datasets: int = -1) -> List[str]:
    """Get sorted list of NetCDF dataset file paths from directory."""
    files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')])
    if n_datasets > 0:
        files = files[:n_datasets]
    return files


def _open_dataset(filepath: str) -> xr.Dataset:
    """Load a single NetCDF dataset from filepath."""
    try:
        ds = xr.open_dataset(filepath)
        ds.load()  # bring into memory for speed
        return ds
    except Exception as e:
        raise RuntimeError(f"Failed to open {filepath}: {e}")


def _extract_sensor_names(ds: xr.Dataset) -> List[str]:
    """Extract base sensor names from dataset variables (strip _real/_imag suffixes)."""
    # Cast names to str to avoid type issues from xarray's Hashable keys
    sn = [str(v) for v in ds.data_vars if 'Mode' not in str(v)]
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
                              freq_tolerance: float, debug: bool = False) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Return regions per mode per selected time index: regions[mode][time_idx] = (freq_inds, sensor_inds)
    If frequency at timepoint is zero, freq_inds will be empty. sensor_inds = all sensors by default.
    Expands tolerance until at least one frequency bin falls in the band.

    The returned structure is a list of modes, each containing a list of time indices,
    where each entry is a tuple of (freq_inds, sensor_inds).
    """
    n_sensors = len(_extract_sensor_names(ds))
    freq_vals = ds['frequency'].values
    df = float(freq_vals[2] - freq_vals[0]) if len(freq_vals) > 1 else 0.0

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
            while (fmax - fmin) < df:
                fmin = f0 * (1 - tol) if f0 * (1 - tol) > 0 else 0.0
                fmax = f0 * (1 + tol)
                tol *= 1.5
                # if tol > 1.0:  # safety
                #     break
            freq_inds = np.where((freq_vals >= fmin) & (freq_vals <= fmax))[0]
            if freq_inds.size == 0:
                mode_regions.append((np.array([], dtype=int), np.arange(n_sensors)))
            else:
                mode_regions.append((freq_inds, np.arange(n_sensors)))
        regions.append(mode_regions)
    
    if debug:
        plt.close('Debug_Mode_Regions')
        fig,ax = plt.subplots(1,1,num='Debug_Mode_Regions',tight_layout=True,figsize=(5,3))
        for mi, mode_var in enumerate(F_mode_vars):
            ax.plot(ds['time'].values*1e3, ds[mode_var].values*1e-3, label=mode_var)
            for ti_i, ti in enumerate(time_indices):ax.plot(ds['time'].values[ti]*1e3,
                                                            ds[mode_var].values[ti]*1e-3,'o',
                                                            color=plt.get_cmap('tab10')(mi))
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Frequency [kHz]')
        ax.legend()
        ax.grid()
        plt.show()
    
    return regions

#####################################################################################################

def _avg_band_per_sensor(ds: xr.Dataset, time_index: int, freq_inds: np.ndarray,
                         sensor_bases: List[str], t_window_width: int) -> Tuple[np.ndarray, np.ndarray]:
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
    # Only pulling one timepoint
    try:
        real_stack = np.stack([ds[f"{base}_real"].values[freq_inds,time_index] for base in sensor_bases], axis=0)
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
    # catch for zero begin the best frequency
    if chosen_freq_idx <= 0: chosen_freq_idx = 1


    # Average real & imag over time at chosen frequency index per sensor
    if t_window_width > 0:
        time_indicies = np.arange(time_index - t_window_width/2 , time_index + t_window_width/2)
    else: 
        time_indicies = np.array([time_index])
    if np.any(time_indicies < 0) or np.any(time_indicies >= ds.dims['time']):
        time_indicies = np.clip(time_indicies, 0, ds.dims['time'] - 1).astype(int)
    time_idx_arr = time_indicies.astype(int)

    for si, base in enumerate(sensor_bases):
        try:
            real_vals = ds[f"{base}_real"].values[chosen_freq_idx, time_idx_arr]
            imag_vals = ds[f"{base}_imag"].values[chosen_freq_idx, time_idx_arr]
        except KeyError:
            real_vals = np.array([])
            imag_vals = np.array([])
        real_out[si] = float(real_vals.mean()) if real_vals.size else 0.0
        imag_out[si] = float(imag_vals.mean()) if imag_vals.size else 0.0
        if real_out[si] < 0: # Check for negative amplitude [This needs to be repeated after first-sensor subtraction]
            real_out[si] = np.abs(real_out[si])
            imag_out[si] = (imag_out[si] + np.pi) % (2 * np.pi)
    return real_out, imag_out


def _get_mode_labels(ds: xr.Dataset, F_mode_vars: List[str]) -> List[Tuple[int, int]]:
    """Return per-mode label pairs (m, n) from dataset attrs.

    Ensures lengths match number of modes, broadcasting single values when needed.
    If attrs are missing, falls back to zeros.
    """
    def _normalize(vals_obj, n_modes: int) -> List[int]:
        if vals_obj is None:
            return [0] * n_modes
        try:
            vals_list = list(vals_obj) if np.ndim(vals_obj) > 0 else [int(vals_obj)]
        except Exception:
            vals_list = [int(vals_obj)]
        if len(vals_list) == 1 and n_modes > 1:
            vals_list = vals_list * n_modes
        if len(vals_list) != n_modes:
            raise ValueError(
                f"Length of labels in attrs ({len(vals_list)}) != number of modes ({n_modes})"
            )
        return [int(v) for v in vals_list]

    n_modes = len(F_mode_vars)
    m_vals = _normalize(ds.attrs.get('mode_m', None), n_modes)
    n_vals = _normalize(ds.attrs.get('mode_n', None), n_modes)
    return list(zip(m_vals, n_vals))


def _maybe_geometry_from_bp_k(geometry_shot: Optional[int], names_sorted: np.ndarray)\
      -> Tuple[np.ndarray, np.ndarray, List[int]]:
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
    matching_sensor_inds = [] # For use extracting correct sensors from X_ri later
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
                ph[i] = Phi * np.pi / 180.0 # convert to radians
                matching_sensor_inds.append(i)
    except Exception as e:
        print(f"Warning: could not load geometry from bp_k: {e}")
    return th, ph, matching_sensor_inds 

def _geometry_from_gen_MAGX(geometry_shot: int, names_sorted: np.ndarray, r_magx: float = 0.68, z_magx : float = 0) \
     -> Tuple[np.ndarray, np.ndarray]:

    phi, theta_pol, R, Z = Mirnov_Geometry_C_Mod(geometry_shot)

    th = np.zeros(len(names_sorted), dtype=np.float32)
    ph = np.zeros(len(names_sorted), dtype=np.float32)      
    for i, nm in enumerate(names_sorted):
        name = str(nm).upper()
        if name not in phi:
            raise ValueError(f"Sensor name '{name}' not found in MAGX geometry for shot {geometry_shot}")
        
        R_s = R[nm]
        Z_s = Z[nm]
        th[i] = np.atan2(Z_s - z_magx, R_s - r_magx)
        ph[i] = phi[nm]*np.pi/180.0  # convert to radians   

    return th, ph


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
    # Try to load existing cached data, if desired
    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    if os.path.exists(cfg.out_path) and cfg.load_saved_data: 
        data = np.load(cfg.out_path, allow_pickle=True)
        print(f"Loaded cached dataset from {cfg.out_path} :: X_ri={data['X_ri'].shape}, y={data['y'].shape}")
        return {k: data[k] for k in data.files}

    # Get list of dataset file paths
    dataset_files = _get_dataset_filepaths(cfg.data_dir, n_datasets=cfg.n_datasets)
    if not dataset_files:
        raise FileNotFoundError(f"No NetCDF datasets found in {cfg.data_dir}")

    print(f"Found {len(dataset_files)} dataset files to process")

    # Open the first dataset to determine sensor ordering, then apply globally
    print(f"Loading first dataset to determine sensor ordering: {dataset_files[0]}")
    first_ds = _open_dataset(dataset_files[0])
    base_sensor_names = _extract_sensor_names(first_ds)
    names_sorted, names_orig, order_idx = gen_sensor_ordering(base_sensor_names)
    first_ds.close()  # Close the first dataset
    del first_ds

    X_ri_list: List[np.ndarray] = []
    y_list: List[int] = []  # chosen training labels based on cfg.use_mode
    y_m_list: List[int] = []  # raw m labels per sample
    y_n_list: List[int] = []  # raw n labels per sample
    used_files: List[str] = []
    used_counts: List[int] = []

    # Flag to visualize only the first dataset
    visualized_first = False

    # Loop over dataset files, loading one at a time
    for ds_idx, ds_filepath in enumerate(dataset_files):
        print(f"Processing dataset {ds_idx+1}/{len(dataset_files)}: {os.path.basename(ds_filepath)}")
        
        try:
            ds = _open_dataset(ds_filepath)
        except Exception as e:
            print(f"Warning: failed to open {ds_filepath}: {e}")
            continue
        sensor_bases = _extract_sensor_names(ds)
        
        # Reorder indices to match global sorted names; if any name missing, fill zeros later
        # Build mapping from sorted names to indices in this dataset
        # This ensures consistent ordering across datasets, matching the sorting scheme
        idx_map = []
        name_to_idx = {b: i for i, b in enumerate(sensor_bases)}
        for nm in names_sorted:
            idx_map.append(name_to_idx.get(nm, -1))

        # Determine time indices to use
        times = np.arange(len(ds['time']))
        if cfg.num_timepoints is not None and cfg.num_timepoints > 0 and cfg.num_timepoints < len(times):
            # random subset for caching speed
            rng = np.random.default_rng(42)
            times = np.sort(rng.choice(times, size=cfg.num_timepoints, replace=False))

        F_mode_vars = [str(v) for v in ds.data_vars if str(v).startswith('F_Mode_')]
        mode_label_pairs = _get_mode_labels(ds, F_mode_vars)
        regions = _define_mode_regions_time(ds, F_mode_vars, times, cfg.freq_tolerance)

        # Visualize first dataset if requested
        if cfg.visualize_first and not visualized_first:
            print(f"Visualizing regions for first dataset...")
            visualize_dataset_regions(ds, times, F_mode_vars, regions,
                                    sensor_to_plot=cfg.viz_sensor,
                                    save_path=cfg.viz_save_path,
                                    mode_region_fLim=cfg.viz_freq_lim,
                                    t_window_width=cfg.t_window_width)
            visualized_first = True

        n_added = 0
        for ti_i, ti in enumerate(times):
            for m_i, mode_var in enumerate(F_mode_vars):
                freq_inds, _sensor_inds = regions[m_i][ti_i]
                if freq_inds.size == 0:
                    continue  # skip zero-frequency times
                real_band, imag_band = _avg_band_per_sensor(ds, int(ti), freq_inds, sensor_bases, t_window_width=cfg.t_window_width)
                # Reorder to global sorted names; missing names -> 0
                S = len(names_sorted)
                real_sorted = np.zeros(S, dtype=np.float32)
                imag_sorted = np.zeros(S, dtype=np.float32)
                # Loop over ordered sensor name indicies to maintain consistant ordering
                for s_idx, src in enumerate(idx_map): 
                    if src >= 0:
                        real_sorted[s_idx] = real_band[src]
                        imag_sorted[s_idx] = imag_band[src]
                X_ri_list.append(np.stack([real_sorted, imag_sorted], axis=-1))  # (S, 2)
                # Check for negative magnitude

                # Choose which label to append based on cfg.use_mode while retaining both
                m_label, n_label = mode_label_pairs[m_i]
                chosen_label = n_label if cfg.use_mode.lower() == 'n' else m_label
                y_list.append(chosen_label)
                y_m_list.append(m_label)
                y_n_list.append(n_label)

                n_added += 1

        used_files.append(ds_filepath)
        used_counts.append(n_added)
        print(f"Accumulated {n_added} samples from dataset")
        
        # Close the dataset to free memory
        ds.close()
        del ds

    if not X_ri_list:
        raise RuntimeError("No samples accumulated. Check frequency bands and tolerance.")

    X_ri = np.stack(X_ri_list, axis=0)  # (N, S, 2)
    y = np.array(y_list, dtype=np.int64)
    y_m = np.array(y_m_list, dtype=np.int64)
    y_n = np.array(y_n_list, dtype=np.int64)

    theta = phi = None
    if cfg.include_geometry:
        # theta, phi = _maybe_geometry_from_bp_k(cfg.geometry_shot, names_sorted)
        theta, phi = _geometry_from_gen_MAGX(cfg.geometry_shot, names_sorted) 

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
        'y': y,  # training labels chosen by cfg.use_mode
        'y_m': y_m,  # raw m labels
        'y_n': y_n,  # raw n labels
        'sensor_names': names_sorted,
        'meta': np.array(json.dumps(meta)),
    }
    if theta is not None and phi is not None:
        save_dict['theta'] = theta
        save_dict['phi'] = phi

    if cfg.saveDataset:
        np.savez(cfg.out_path, **save_dict)
        print(f"Saved cached dataset to {cfg.out_path} :: X_ri={X_ri.shape}, y={y.shape}")

    return save_dict
###############################################################################################
def sensor_reduction(X_ri: np.ndarray, sensor_names: np.ndarray, theta: Optional[np.ndarray],
                     phi: Optional[np.ndarray], geometry_shot: int)\
                          -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  
    
    th, ph, matching_sensor_inds = _maybe_geometry_from_bp_k(geometry_shot, sensor_names)

    theta = theta[matching_sensor_inds]; phi = phi[matching_sensor_inds]
    X_ri_reduced = X_ri[:, matching_sensor_inds, :]
    sensor_names_reduced = sensor_names[matching_sensor_inds]

    return X_ri_reduced, sensor_names_reduced, theta, phi

# -------------------------
# High-level interface
# -------------------------
def build_or_load_cached_dataset(data_dir: str, out_path: str, use_mode: str = 'n',
                                 include_geometry: bool = True, geometry_shot: Optional[int] = None,
                                 num_timepoints: int = -1, freq_tolerance: float = 0.1,
                                 n_datasets: int = -1, visualize_first: bool = False,
                                 viz_save_path: Optional[str] = None,
                                 viz_sensor: str = 'BP01_ABK',
                                 viz_freq_lim: Optional[List[float]] = None,
                                 load_saved_data : bool = True,
                                 doVisualize : bool = False,
                                 saveDataset: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
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
        visualize_first: Whether to visualize frequency-time regions for first dataset
        viz_save_path: Path to save visualization plots (e.g., '../output_plots/')
        viz_sensor: Sensor name to visualize (default: 'BP01_ABK')
        viz_freq_lim: Frequency limits [f_min, f_max] in kHz for visualization y-axis
        
    Returns:
        X_ri: (N, S, 2) array of [real, imag] features
        y: (N,) array of labels
        sensor_names: (S,) array of sensor names in sorted order
        theta: (S,) array of theta coordinates (or None)
        phi: (S,) array of phi coordinates (or None)
    """
    cfg = CacheConfig(data_dir=data_dir, out_path=out_path, num_timepoints=num_timepoints,
                      freq_tolerance=freq_tolerance, include_geometry=include_geometry,
                      geometry_shot=geometry_shot, use_mode=use_mode, n_datasets=n_datasets,
                      visualize_first=visualize_first, viz_save_path=viz_save_path,
                      viz_sensor=viz_sensor, viz_freq_lim=viz_freq_lim,\
                        load_saved_data=load_saved_data, saveDataset=saveDataset)
    dat = cache_training_dataset(cfg)
    X_ri = dat['X_ri'] if isinstance(dat, dict) else dat.get('X_ri')
    y = dat['y'] if isinstance(dat, dict) else dat.get('y')
    y_m = dat['y_m'] if isinstance(dat, dict) else dat.get('y_m')
    y_n = dat['y_n'] if isinstance(dat, dict) else dat.get('y_n')
    sensor_names = dat['sensor_names'] if isinstance(dat, dict) else dat.get('sensor_names')
    theta = dat.get('theta', None) if isinstance(dat, dict) else None
    phi = dat.get('phi', None) if isinstance(dat, dict) else None
    
    # Process sensor reduction if needed
    if geometry_shot is not None and include_geometry:
        X_ri, sensor_names, theta, phi = \
            sensor_reduction(X_ri, sensor_names, theta, phi, geometry_shot) 

    if doVisualize: 
        visualize_cached_data(cfg, y_m, y_n, X_ri)
        # visualize_contours(cfg, y_m, y_n, X_ri,theta,phi)


    return X_ri, (y_n if use_mode == 'n' else y_m), y_m, y_n, sensor_names, theta, phi


if __name__ == "__main__":
    # Quick test/example
    print("Data caching module for synthetic spectrogram datasets.")
    print("Import this module and use build_or_load_cached_dataset() to prepare training data.")
    print("\nExample usage with visualization:")
    print("X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(")
    print("    data_dir='../data_output/synthetic_spectrograms/',")
    print("    out_path='cached_data.npz',")
    print("    visualize_first=True,  # Enable visualization")
    print("    viz_save_path='../output_plots/',  # Save plots here")
    print("    viz_sensor='BP01_ABK',  # Sensor to visualize")
    print("    viz_freq_lim=[0, 300]  # Frequency limits in kHz")
    print(")")


    X_ri, y, y_m, y_n, sensor_names, theta, phi = build_or_load_cached_dataset(
        data_dir='../data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/',
        out_path='../data_output/synthetic_spectrograms/low_m-n_testing/'+\
            'new_Mirnov_set/cached_data_-1.npz',
        visualize_first=True,  # Enable visualization
        viz_save_path='../output_plots/',  # Save plots here
        viz_sensor='BP01_ABK',  # Sensor to visualize
        viz_freq_lim=[0, 300],  # Frequency limits in kHz
        n_datasets=-1,  # Limit to first 2 datasets for testing
        num_timepoints=10,  # Limit to 10 timepoints per dataset for testing
        load_saved_data=False, # Force load existing cached data if available
        include_geometry=True,
        geometry_shot=1160714026,
        doVisualize=False,
    )

    print('Cached dataset shapes:   X_ri=', X_ri.shape, ', y_m=', y_m.shape, ', y_n=', y_n.shape)

############################################################################################    
############################################################################################
def __optimal_subplot_grid(num_datasets: int) -> Tuple[int, int]:
    """
    Find optimal subplot grid dimensions (rows, cols) for num_datasets.
    
    Minimizes wasted space by finding m, n such that:
    - m * n >= num_datasets
    - m * n is minimized
    - m and n are as close as possible (prefer m <= n for landscape orientation)
    
    Args:
        num_datasets: Number of datasets to plot
        
    Returns:
        (rows, cols) tuple for optimal subplot grid
    """
    if num_datasets <= 0:
        return (1, 1)
    
    # Start with square root and search nearby values
    sqrt_j = int(np.sqrt(num_datasets))
    
    best_waste = float('inf')
    best_m, best_n = 1, num_datasets
    
    # Search from sqrt down to 1 for number of rows
    for m in range(max(1, sqrt_j - 2), sqrt_j + 3):
        n = int(np.ceil(num_datasets / m))
        waste = m * n - num_datasets
        
        # Prefer solutions closer to square (m closer to n)
        aspect_ratio_penalty = abs(m - n) * 0.01
        total_cost = waste + aspect_ratio_penalty
        
        if total_cost < best_waste:
            best_waste = total_cost
            best_m, best_n = m, n
    
    return (best_m, best_n)
