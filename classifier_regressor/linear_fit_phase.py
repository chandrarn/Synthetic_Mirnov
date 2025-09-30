#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:06:47 2025
    Pull in datasets
    extract real, imag components for fixed timepoints, along with target labels
    pull in sensor locations
    for n: use rows in Z for calcualtion, with phase-normalized imag components from dataset
    
    Initial test with one file
@author: rianc
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
import xarray as xr

#from regression_phase import extract_spectrogram_matrices, load_xarray_datasets, \
#            define_mode_regions


def linear_phase_fit():
    
    # Load in one dataset
    data_directory = "../data_output/synthetic_spectrograms/" 
    ds = load_xarray_datasets(data_directory)
    
    # Load sensor locations (necessary for n# determination)
    R_sensors, Z_sensors, phi_sensors = get_sensors()
    Z_level_sensor_names, Z_level_sensor_inds = gen_sensor_groupings_by_Z(Z_sensors)
    
    Z_phi_map(Z_sensors,phi_sensors)
    
    # Get component matrix 
    X, y_m, y_n = prepare_training_data(ds,Z_level_sensor_inds,10)
    
################################################################################
def get_sensors():
    with open('../C-Mod/C_Mod_Mirnov_Geometry_R.json','r',\
              encoding='utf-8') as f: R=json.load(f)
    with open('../C-Mod/C_Mod_Mirnov_Geometry_Z.json','r',\
              encoding='utf-8') as f:  Z=json.load(f)
    with open('../C-Mod/C_Mod_Mirnov_Geometry_Phi.json','r',\
              encoding='utf-8') as f:  phi=json.load(f)        
    
    return R, Z, phi
###############################################################################
def gen_sensor_groupings_by_Z(Z,Z_levels=[10,13,20],tol=1):
    # Generate the correct level groupings by sensor name, for later extraction
    Z_level_sensor_names = []
    Z_level_sensor_inds = []
    
    Z_list = np.array([Z[z_] for z_ in Z]) * 1e2
    names_list = np.array(list(Z.keys()))
    for z in Z_levels:
        for s in [-1,1]:
            z_inds = np.argwhere((z*s-tol < Z_list) & (Z_list < z*s+tol) )
            Z_level_sensor_names.append(names_list[z_inds])
            Z_level_sensor_inds.append(z_inds)
    
    return Z_level_sensor_names, Z_level_sensor_inds
###############################################################################
    
def prepare_training_data(datasets,Z_level_sensor_inds, num_timepoints=22):
    """
    Prepare training data from all datasets.
    Instead of padding, loop across time_indices and extract relevant elements only if frequency != 0.
    Collect features and targets in lists, then convert to arrays.
    """
    all_features = []
    all_targets_m = []
    all_targets_n = []

    for ds in datasets:
        # Select random timepoints
        time_indices = np.linspace(0.1*len(ds['time']),.9*len(ds['time']),num_timepoints)
        
        # Extract matrices for each timepoint, real/imag for each sensor
        real_mats, imag_mats, sensor_names = extract_spectrogram_matrices(ds, time_indices)
        
        # Get F_Mode variables
        F_mode_vars = [var for var in ds.data_vars if var.startswith('F_Mode_')]
        
        # Define regions (now time-dependent)
        regions = define_mode_regions(ds, F_mode_vars, time_indices)
        
        # Get targets
        mode_m = ds.attrs['mode_m']
        mode_n = ds.attrs['mode_n']
        if np.size(mode_m) == 1: mode_m = [mode_m] # Catch for single mode
        if np.size(mode_n) == 1: mode_n = [mode_n]
        if len(mode_m) != len(F_mode_vars) or len(mode_n) != len(F_mode_vars):
            raise ValueError("Length of mode_m/n does not match number of F_Mode_ variables")


        # Loop across time_indices
        for time_idx in range(len(time_indices)):
            for mode_idx in range(len(F_mode_vars)):

                freq_idx, sensor_idx = regions[mode_idx][time_idx]

                
                if len(freq_idx) > 0:  # Only if frequency != 0 (i.e., valid region)
                    real_region = real_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    imag_region = imag_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    
                    # Extract features : take maximim peak and pull phase from that?
                    peak_in_f = np.argmax(np.mean(real_region, axis=0))  
                    avg_imag_per_sensor = imag_region[peak_in_f]  # Extrax all sensor phases at peak amplitude-f
                    
                    
                    all_z_levels_mode_phase_phase = []
                    # Loop through toroidal rings of sensors, by z-level
                    for Z_level_names_list in Z_level_sensor_inds:
                        diffs_imag = avg_imag_per_sensor[Z_level_names_list[1:]] - \
                            avg_imag_per_sensor[Z_level_names_list0]]
                        
                        all_z_levels_mode_phase_phase = np.concatenate([\
                                all_z_levels_mode_phase_phase,diff_imag])
                    
                    all_features.append(all_z_levels_mode_phase_phase)
                    # all_targets_m.append(mode_m[mode_idx])
                   
                  
                   all_targets_n.append(mode_n[mode_idx])
    
    # Convert lists to numpy arrays
    X = np.array(all_features)
    y_m = np.array(all_targets_m)
    y_n = np.array(all_targets_n)
    
    return X, y_m, y_n

###############################################################################
def __Coord_Map(R,Z,phi,R_map,Z_map,phi_map,doSave,shotno,z_levels,z_inds_out,save_Ext):
    plt.close('Z-Phi_Map_%d%s'%(shotno,save_Ext))
    fig,ax=plt.subplots(1,1,num='Z-Phi_Map_%d%s'%(shotno,save_Ext),tight_layout=True,figsize=(3.4,2))
    fn_norm = lambda r: (r-np.min(R))/(np.max(R)-np.min(R)) *.5 +.5
    if np.mean(phi) < 0: phi += 360; phi_map += 360
    
    ax.plot(phi_map,Z_map*1e2,'k*',label='Unused')
    z_inds_out=np.array(z_inds_out,dtype=object)
    for ind,z_inds in enumerate(z_inds_out[:,0]):
        flag=True
        for z_ind in z_inds:
            plt.plot(phi[z_ind], Z[z_ind]*1e2,'*', c=plt.get_cmap('tab10')(ind),\
                 alpha=fn_norm(R[z_ind]), label=('%1.1f cm'%z_inds_out[ind,1])*flag,ms=5)
            flag=False
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel(r'Z [cm]')
    ax.legend(fontsize=9,handlelength=1.5,title_fontsize=8,)
     #         title=r'R$in${%1.1f-%1.1f}'%(np.min(R),np.max(R)))
    ax.grid()
    
    if doSave:
        fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.png',transparent=True)
    plt.show() 


def Z_phi_map(Z,phi):
    plt.close('Z_Map_Phi')
    fig,ax=plt.subplots(1,1,num='Z_Map_Phi',tight_layout=True)
    Z_ = np.array([Z[z_] for z_ in Z])
    phi_ = [phi[p_] for p_ in phi]
    
    ax.plot(phi_,Z_*1e3,'*')
    ax.set_xlabel(r'$\Phi$ [deg]')
    ax.set_ylabel(r'Z [cm]')
    ax.grid()
    
    plt.show()
    
    
    
################################################################################
def load_xarray_datasets(data_directory, n_files=1):
    """
    Load all xarray datasets from NetCDF files in the specified directory.
    """
    file_pattern = os.path.join(data_directory, "*.nc")
    files = glob.glob(file_pattern)
    datasets = []
    for file in files[:n_files]:
        ds = xr.open_dataset(file)
        datasets.append(ds)
    return datasets

def extract_spectrogram_matrices(ds, timepoint_indices):
    """
    Extract real and imaginary spectrogram matrices for given timepoints.
    Similar to convert_spectrogram_to_training_data function.

    returns a list of matricies, one per timepoint, for the real and imaginary parts of each sensor
    """
    sensor_names = [var for var in ds.data_vars if "Mode" not in var]
    num_sensors = len(sensor_names) // 2  # Assuming _real and _imag for each sensor

    all_real_matrices = []
    all_imag_matrices = []

    for timepoint_index in timepoint_indices:
        spect_real = np.zeros((len(ds['frequency']), num_sensors))
        spect_imag = np.zeros((len(ds['frequency']), num_sensors))

        for i in range(num_sensors):
            sensor_name = sensor_names[i*2][:-5]  # Remove _real or _imag
            spect_real[:, i] = ds[sensor_name + '_real'].isel(time=timepoint_index).values
            spect_imag[:, i] = ds[sensor_name + '_imag'].isel(time=timepoint_index).values

        all_real_matrices.append(spect_real)
        all_imag_matrices.append(spect_imag)

    return all_real_matrices, all_imag_matrices, sensor_names[:num_sensors]


###############################################################################

def define_mode_regions(ds, F_mode_vars, time_indices, freq_tolerance=0.1):
    """
    Define regions in frequency-sensor space for each mode based on F_Mode_# vectors.
    Now accounts for freq_values being functions of time.
    Returns a list of lists: regions[mode][timepoint] = (freq_indices, sensor_indices)
    # Will return an entry with empty indices if frequency is zero at that timepoint
    # freq_tolerance defines the fractional range around the mode frequency to include
    """
    number_sensors = len([var for var in ds.data_vars if "Mode" not in var]) // 2
    regions = []
    for mode_var in F_mode_vars:
        mode_regions = []
        freq_values_all = ds[mode_var].values  # Full time-dependent frequency vector for this mode
        for time_idx in time_indices:
            # Get the frequency at this specific timepoint
            freq_at_time = freq_values_all[time_idx]
            # Define frequency range around this value
            if freq_at_time == 0: # Skip modes with zero frequency
                mode_regions.append(([], []))
                print(f"Skipping mode {mode_var} at time index {time_idx} due to zero frequency")
                continue
            freq_min=freq_max=0
            while freq_max - freq_min < ds['frequency'].values[1]-ds['frequency'].values[0]:
                freq_min = freq_at_time * (1 - freq_tolerance)
                freq_max = freq_at_time * (1 + freq_tolerance)
                freq_tolerance *= 1.5  # Expand range if too narrow

            freq_indices = np.where((ds['frequency'].values >= freq_min) &
                                    (ds['frequency'].values <= freq_max))[0]
            if len(freq_indices) == 0:
                print(f"Warning: No frequency indices found for mode {mode_var} at time index {time_idx}")
            sensor_indices = np.arange(number_sensors)  # All sensors
            mode_regions.append((freq_indices, sensor_indices))
        regions.append(mode_regions)
    return regions

