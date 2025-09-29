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
from regression_phase import extract_spectrogram_matrices, load_xarray_datasets, \
            define_mode_regions


def linear_phase_fit():
    
    # Load in one dataset
    data_directory = "../data_output/synthetic_spectrograms/" 
    ds = load_xarray_datasets(data_directory)
    
    # Load sensor locations (necessary for n# determination)
    R_sensors, Z_sensors, phi_sensors = get_sensors()
    Z_level_sensor_names, Z_level_sensor_inds = gen_sensor_groupings_by_Z(Z_sensors)
    
    Z_phi_map(Z_sensors,phi_sensors)
    return Z_grouping_names, Z_sensors
    # Get component matrix 
    X, y_m, y_n = prepare_training_data(ds,Z_level_sensor_names,10)
    
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
    names_list = list(Z.keys())
    for z in Z_levels:
        for s in [-1,1]:
            z_inds = np.argwhere((z*s-tol < Z_list) & (Z_list < z*s+tol) )
            Z_level_sensor_names.append(names_list[z_inds])
            Z_level_sensor_inds.append(z_inds)
    
    return Z_level_sensor_names, Z_level_sensor_inds
###############################################################################
    
def prepare_training_data(datasets, num_timepoints=22):
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
                    
                    # Add pairwise differences (for simplicity, difference with the first sensor)
                    if len(avg_real_per_sensor) > 1:
                        diffs_real = avg_real_per_sensor[1:] - avg_real_per_sensor[0]
                        diffs_imag = avg_imag_per_sensor[1:] - avg_imag_per_sensor[0]
                        feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor, diffs_real, diffs_imag])
                    else:
                        feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor])

                  
                    all_features.append(feat)
                    all_targets_m.append(mode_m[mode_idx])
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
    
    
    
    
