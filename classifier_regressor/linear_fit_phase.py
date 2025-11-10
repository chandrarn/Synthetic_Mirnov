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
import json
import os
import glob
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib import rc,cm
import matplotlib
from matplotlib.colors import Normalize
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
matplotlib.use('TkAgg')
plt.ion()

#from regression_phase import extract_spectrogram_matrices, load_xarray_datasets, \
#            define_mode_regions


def linear_phase_fit(doPlot_Z_phi_map=True):
    
    # Load in one dataset
    data_directory = "../data_output/synthetic_spectrograms/low_m-n_testing/" 
    ds = load_xarray_datasets(data_directory,n_files=1)
    
    # Load sensor locations (necessary for n# determination)
    R_sensors, Z_sensors, phi_sensors = get_sensors()
    
    # Generate groupings of sensors by Z level
    Z_levels=[10,13,20]
    Z_level_sensor_names, Z_level_sensor_inds = gen_sensor_groupings_by_Z(Z_sensors,Z_levels)
    
    # Generate delta phi values for each Z level grouping
    dPhi, ordered_phi = generate_delta_phi_by_Z_level(phi_sensors,Z_level_sensor_names)

    # Get the indicies of the sensors in the dataset ordering, for each Z level grouping
    Z_level_inds_ds_ordering = find_sensor_by_index(Z_level_sensor_names, ds[0])
    
    if doPlot_Z_phi_map:
        Z_phi_map(Z_sensors,phi_sensors,Z_level_sensor_names)
    
    
    # Get component matrix
    # Phase measured is size [total number of fittable modes in all datasets] x [total number of sensors in all Z levels = 35]
    phase_measured, y_m, y_n, all_F_mode_vars, all_time_indices = \
        prepare_training_data(ds,Z_level_inds_ds_ordering,10)

   

    # Fit linear model
    print('Fitting linear model')
    n_opt, error =  run_optimization(phase_measured,y_n,dPhi,ordered_phi,showMode=1)

    plot_phase_by_group(phase_measured,ordered_phi,dPhi,y_n,Z_levels,n_opt,X_ind=1)
    # Plot results
    build_plot(ds,n_opt,all_F_mode_vars,all_time_indices,'',y_n,ds_ind=1)

    # Build Confusion Matrix
    create_confusion_matrix(n_opt, y_n, error, np.arange(1,16))  # Assuming x_range is accessible or pass it
    
    print('Done')
###########################################################################
def run_optimization(phase_measured,y_n,dPhi,ordered_phi,showMode=0,doPlot=True):

    flatPhase = np.array([p for p_ in dPhi for p in p_])
    fn = lambda n: __fn_opt(n,flatPhase,phase_measured)

    #res = minimize(fn,[n[-1]],bounds=[[-15,15]])
    #n_opt = res.x
    x_range = np.arange(1,16)
    x_range = x_range[x_range != 0] # Exclude n=0
    error = np.array([fn(-n_) for n_ in x_range])
    objective = np.argmin(error,axis=0)
    n_opt = x_range[objective]

    if doPlot:
        plt.close('Debug_n_optimization')
        fig,ax=plt.subplots(1,1,num='Debug_n_optimization',tight_layout=True,figsize=(5,3))
        ax.plot(x_range,error[:,showMode],'o-',ms=4,lw=1)

    return n_opt, error
################################################################################
def build_plot(ds,n_opt,all_F_mode_vars,all_time_indices,save_Ext,y_n,ds_ind = 0):
    # Build plot of frequency vs time, with mode n overlaid
    plt.close('Frequency_vs_time_with_n')
    fig,ax=plt.subplots(1,1,num='Frequency_vs_time_with_n',tight_layout=True,figsize=(5,3))
    

    # Probably only useful for a single dataset
    time = ds[ds_ind]['time'].values[all_time_indices[ds_ind]]*1e3
    F_mode_vars = np.array([ds[ds_ind][mode_name].values[all_time_indices[ds_ind]] \
                   for mode_name in all_F_mode_vars[ds_ind] ])
    
    ind = 0
    norm = Normalize(vmin=np.min(n_opt), vmax=np.max(n_opt))
    for ind_t,t in enumerate(time):
        for _,f in enumerate(F_mode_vars[:,ind_t]):
            if f > 0:
                ax.plot(t,f*1e-3,'.',alpha=0.9,ms=7,color=plt.get_cmap('tab10')(norm(n_opt[ind])))
                ind+=1  

    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap('tab10'), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm,ax=ax,pad=0.01)
    cbar.set_label(r'$n_{opt}$',rotation=270,labelpad=15)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Frequency [kHz]')



######################################################################################
def plot_phase_by_group(phase_measured,ordered_phi,dPhi,y_n,Z_level_vals,n_opt,X_ind=0,):
    # Plot raw phase by Z level grouping
    # X_ind is the index of the mode to plot in terms of all possible modes
    plt.close('Raw_Phase_by_Z-level')
    fig,ax = plt.subplots(1,1,num='Raw_Phase_by_Z-level',tight_layout=True,figsize=(5,3))

    Z_levels_full = [val for num in Z_level_vals for val in (-num,num)]
    ind=0
    for level,phi_list in enumerate(dPhi):
        phase = phase_measured[X_ind,ind:ind+len(phi_list)]
        # dPhi_local  = dPhi[level,ind:ind+len(phi_list)]
        ax.plot(phi_list,phase,'o',ms=4,lw=1,label='%1.1f cm'%(Z_levels_full[level]) )
        ind+=len(phi_list)
        x_vals_n = np.linspace(np.min(phi_list),np.max(phi_list),100)
    pred_phase = (-n_opt[X_ind] * np.linspace(0,2*np.pi,50)) % (2*np.pi)
    # pred_phase[pred_phase > np.pi] -= 2*np.pi
    ax.plot(np.linspace(0,2*np.pi,50), pred_phase,'--',color=plt.get_cmap('tab10')(0),lw=1)
    ax.grid()
    ax.set_xlabel(r'$\Delta\Phi_{mir}$ [rad]')
    ax.set_ylabel(r'$\Delta$ Phase [rad]')

    ax.legend(handlelength=1.5,fontsize=8,
              title=r'$n_{synth.}$ = %d, $n_{opt}$ = %d'%(y_n[X_ind],n_opt[X_ind]),title_fontsize=10,ncols=2)

    fig.savefig('../output_plots/Raw_Phase_by_Z-level.pdf',transparent=True)

################################################################################
def __fn_opt(n,angles,phase_measured):
    # at every angle, calculate the distance to the corresponding predicted angle
    return np.mean(np.sqrt( (np.cos(phase_measured)-np.cos(n * angles))**2 +\
                   (np.sin(phase_measured)-np.sin(n * angles))**2 ),axis=1)


###############################################################################
def find_sensor_by_index(Z_level_sensor_names, ds):
    out_inds = []
    # Get the sensor names in the ordering of the dataset, map to their indicies
    # Only need to look at half the variables, since real/imag pairs 
    # share the same ordering
    sensor_names_ds = [var.replace('_real','') for var in ds.data_vars if "Mode" not in var][::2]
    sensor_names_index_map = {name: idx for idx, name in enumerate(sensor_names_ds)}

    for i, name_list in enumerate(Z_level_sensor_names):
        out_inds.append([sensor_names_index_map[name] for name in name_list if name in sensor_names_index_map])
    
    return out_inds
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
def gen_sensor_groupings_by_Z(Z,Z_levels,tol=1):
    # Generate the correct level groupings by sensor name, for later extraction
    Z_level_sensor_names = []
    Z_level_sensor_inds = []
    
    Z_list = np.array([Z[z_] for z_ in Z]) * 1e2
    names_list = np.array(list(Z.keys()))
    for z in Z_levels:
        for s in [-1,1]:
            z_inds = np.argwhere((z*s-tol < Z_list) & (Z_list < z*s+tol) ).squeeze()
            Z_level_sensor_names.append(names_list[z_inds])
            Z_level_sensor_inds.append(z_inds)
    
    return Z_level_sensor_names, Z_level_sensor_inds
#################################################################################
def generate_delta_phi_by_Z_level(phi,Z_level_sensor_names):
    # Generate the (spatial) delta phi values for each Z level grouping
    all_dphi = []
    all_phi = []
    for name_list in Z_level_sensor_names:
        phi_list = np.array([phi[n_] for n_ in name_list])
        dphi = phi_list[:] - phi_list[0] # Normalize to first sensor in group (include 0)
        dphi *= np.pi/180 # Convert to radians
        dphi = dphi % (2*np.pi) # Wrap to 2pi
        all_dphi.append(dphi)
        all_phi.append(phi_list)
    
    return all_dphi,all_phi
###############################################################################
    
def prepare_training_data(datasets,Z_level_inds_ds_ordering, num_timepoints=22,debug_Plot=True,\
                          saveDebugPlots='../output_plots/',mode_region_fLim=[0,300]):
    """
    Prepare training data from all datasets.
    Instead of padding, loop across time_indices and extract relevant elements only if frequency != 0.
    Collect features and targets in lists, then convert to arrays.
    """
    all_features = []
    all_targets_m = []
    all_targets_n = []
    all_F_mode_vars = []
    all_time_indices = []
    for ds in datasets:
        # Select random timepoints
        time_indices = np.linspace(0.1*len(ds['time']),.9*len(ds['time']),num_timepoints,dtype=int)
        all_time_indices.append(time_indices)
        # Extract matrices for each timepoint, real/imag for each sensor
        real_mats, imag_mats, sensor_names = extract_spectrogram_matrices(ds, time_indices)
        
        # Get F_Mode variables
        F_mode_vars = [var for var in ds.data_vars if var.startswith('F_Mode_')]
        all_F_mode_vars.append(F_mode_vars)

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
        store_indices = []
        for time_idx in range(len(time_indices)):
            for mode_idx in range(len(F_mode_vars)):

                freq_idx, sensor_idx = regions[mode_idx][time_idx]
                
                if len(freq_idx) > 0:  # Only if frequency != 0 (i.e., valid region)

                    # Store valid mode region coordinates
                    store_indices.append((time_idx, mode_idx,freq_idx, sensor_idx))

                    # Extract relevant region from real/imag matrices
                    real_region = real_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    imag_region = imag_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    
                    
                    # Extract features : take maximim peak and pull phase from that?
                    peak_in_f = np.argmax(np.mean(real_region, axis=1))  
                    avg_imag_per_sensor = imag_region[peak_in_f]  # Extrax all sensor phases at peak amplitude-f
                    
                    # Vector of phase differences for all Z_levels, for a given mode, time difference
                    all_z_levels_mode_phase_phase = []

                    # Loop through toroidal rings of sensors, by z-level
                    for ind,Z_level_inds in enumerate(Z_level_inds_ds_ordering):
                        diffs_imag = (avg_imag_per_sensor[Z_level_inds[:]] - \
                            avg_imag_per_sensor[Z_level_inds[0]]) % (2*np.pi)
                        
                        # Normalize by dPhi for that Z level
                        # diffs_imag = diffs_imag / dPhi[ind]
                        
                        all_z_levels_mode_phase_phase = np.concatenate([\
                                all_z_levels_mode_phase_phase,diffs_imag])
                    
                    all_features.append(all_z_levels_mode_phase_phase)
                    all_targets_m.append(mode_m[mode_idx])
                   
                   
                    all_targets_n.append(mode_n[mode_idx])
    

    # Convert lists to numpy arrays
    X = np.array(all_features)
    y_m = np.array(all_targets_m)
    y_n = np.array(all_targets_n)
    
    # Debug plot: plot last real/imag matrix
    if debug_Plot:
        plt.close('Debug_Real_Imag_Matrix')
        fig,ax=plt.subplots(1,2,num='Debug_Real_Imag_Matrix',tight_layout=True,figsize=(8,4),sharex=True,sharey=True)
        im0 = ax[0].pcolormesh(store_indices[-1][3],ds['frequency'][store_indices[-1][2]]*1e-3,real_region,cmap='viridis')
        fig.colorbar(im0,ax=ax[0],label=r'Synth Sig. [T/s]')
        fig.suptitle('Mag/Arg Signal (Mode %s, Time %1.2e s)'%(F_mode_vars[store_indices[-1][1]],ds['time'][time_indices[store_indices[-1][0]]]))
        
        im1 = ax[1].pcolormesh(store_indices[-1][3],ds['frequency'][store_indices[-1][2]]*1e-3,imag_region,cmap='viridis')
        fig.colorbar(im1,ax=ax[1],label=r'$\phi_{sig}$ [rad]')
        # ax[1].set_title('Imag part of region')
        ax[0].set_ylabel('Frequency [kHz]')
        ax[0].set_xlabel('Sensor index')
        if saveDebugPlots: fig.savefig(saveDebugPlots+'Debug_Real_Imag_Matrix.pdf',transparent=True)
    
    # For one sensor and the final dataset, plot the real magnitude and freq_indx, time_idx regions
    if debug_Plot:
        sensor_to_plot = 'BP01_ABK'
        data = ds[sensor_to_plot+'_real'].values
        plt.close(f'Debug_{sensor_to_plot}_Real_Spectrogram')
        fig,ax=plt.subplots(1,1,num=f'Debug_{sensor_to_plot}_Real_Spectrogram',tight_layout=True,figsize=(5,3))
        im = ax.pcolormesh(ds['time'].values*1e3,ds['frequency'].values*1e-3,data,cmap='viridis',zorder=-5)
        fig.colorbar(im,ax=ax,label=r'Synth Sig. [T/s]')

        # Loop over valid mode analysis regions and plot rectangular regions
        from matplotlib.patches import Rectangle
        for (time_idx, mode_idx, freq_idx, sensor_idx) in store_indices:
            if len(freq_idx) > 0:  # Only plot if there are frequency indices
                # Define the rectangular region: time is single point, so use a small width around it
                time_val = ds['time'].values[time_indices[time_idx]] * 1e3
                freq_min = ds['frequency'].values[freq_idx].min() * 1e-3
                freq_max = ds['frequency'].values[freq_idx].max() * 1e-3
                # Assume time bin width for rectangle width (or use a fixed small value)
                time_bin_width = (ds['time'].values[1] - ds['time'].values[0]) * 1e3 if len(ds['time']) > 1 else 0.1  # Fallback small width
                m_=ds.mode_m[mode_idx] if np.size(ds.mode_m)>1 else ds.mode_m
                n_=ds.mode_n[mode_idx] if np.size(ds.mode_n)>1 else ds.mode_n
                rect = Rectangle((time_val - time_bin_width/2, freq_min), time_bin_width, freq_max - freq_min,
                                 linewidth=2, edgecolor=plt.get_cmap('tab10')(mode_idx), facecolor='none',
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

        if saveDebugPlots: fig.savefig(saveDebugPlots+f'Debug_{sensor_to_plot}_Real_Spectrogram.pdf',\
                                       transparent=True)
    return X, y_m, y_n, all_F_mode_vars, all_time_indices

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


def Z_phi_map(Z,phi,Z_names):
    plt.close('Z_Map_Phi')
    fig,ax=plt.subplots(1,1,num='Z_Map_Phi',tight_layout=True,figsize=(5,2))
    Z_ = np.array([Z[z_] for z_ in Z])
    phi_ = [phi[p_] for p_ in phi]
    
    ax.plot(phi_,Z_,'*k',label='All Mirnov Sensors')
    for i,name in enumerate(Z_names):
        phi_ = [phi[n] for n in name]
        Z_ = np.array([Z[n] for n in name])
        ax.plot(phi_,Z_,'*',ms=10)
    ax.set_xlabel(r'$\Phi$ [deg]')
    ax.set_ylabel(r'Z [cm]')
    ax.legend(fontsize=8,handlelength=1.5)
    ax.grid()
    fig.savefig('../output_plots/Z_Map_Phi.pdf',transparent=True)
    
    plt.show()
    
    
    
################################################################################
def load_xarray_datasets(data_directory, n_files=2):
    """
    Load all xarray datasets from NetCDF files in the specified directory.
    """
    file_pattern = os.path.join(data_directory, "*.nc")
    files = glob.glob(file_pattern)
    datasets = []
    if n_files == -1:
        n_files = len(files)
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

def define_mode_regions(ds, F_mode_vars, time_indices, freq_tolerance_init=0.1):
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

            # At minimum, include at least three frequency bins per region
            freq_min=freq_max=0
            freq_tolerance = freq_tolerance_init
            while freq_max - freq_min < ds['frequency'].values[4]-ds['frequency'].values[0]:
                freq_min = freq_at_time * (1 - freq_tolerance)
                freq_max = freq_at_time * (1 + freq_tolerance)
                freq_tolerance *= 1.5  # Expand range if too narrow
            # Check for too large range
            if freq_max-freq_min > 60e3: # 60 kHz max range
                freq_min = freq_at_time - 30e3
                freq_max = freq_at_time + 30e3

            freq_indices = np.where((ds['frequency'].values >= freq_min) &
                                    (ds['frequency'].values <= freq_max))[0]
            if len(freq_indices) == 0:
                print(f"Warning: No frequency indices found for mode {mode_var} at time index {time_idx}")
            sensor_indices = np.arange(number_sensors)  # All sensors
            mode_regions.append((freq_indices, sensor_indices))
        regions.append(mode_regions)
    return regions

################################################################################
def create_confusion_matrix(n_opt, y_n, error, x_range):
    """
    Create a confusion matrix plot, compute accuracy, and compute AUC for multiclass classification.
    """
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, ConfusionMatrixDisplay
    import numpy as np

    # Compute confusion matrix
    cm = confusion_matrix(y_n, n_opt)
    
    # Plot confusion matrix
    plt.close('Confusion_Matrix')
    fig, ax = plt.subplots(1, 1, num='Confusion_Matrix', tight_layout=True, figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_n, n_opt, ax=ax, cmap='viridis')
   
    
    # Compute accuracy
    accuracy = accuracy_score(y_n, n_opt)
    print(f'Accuracy: {accuracy:.4f}')
    
    # Compute AUC using error as scores (higher score for lower error)
    # Use softmax-like probabilities from error
    probs = np.exp(-error) / np.sum(np.exp(-error), axis=0)  # probs is (n_classes, n_samples)
    probs = probs.T  # Now (n_samples, n_classes)
    
    # Select only the classes present in y_n
    unique_n = np.unique(y_n)
    indices = [np.where(x_range == n)[0][0] for n in unique_n]
    probs_selected = probs[:, indices]
    
    # Normalize each row to ensure they sum to 1 (fix numerical issues)
    probs_selected = probs_selected / probs_selected.sum(axis=1, keepdims=True)
    
    # For multiclass AUC, use one-vs-rest
    try:
        auc = roc_auc_score(y_n, probs_selected, multi_class='ovr')
        print(f'AUC (OVR): {auc:.4f}')
    except ValueError as e:
        print(f'AUC could not be computed: {e}')
        auc = None

    ax.set_title('Confusion Matrix for n Prediction (Accuracy: {:.2f}, AUC: {:.2f})'.\
                 format(accuracy, auc if auc is not None else 0))
    plt.show()
    fig.savefig(f'../output_plots/Confusion_Matrix_Linear_acc_{accuracy:.3f}_auc_{auc:.2f}.pdf',transparent=True)
    
    return accuracy, auc
###########################################################################
if __name__ == "__main__":
    linear_phase_fit()
    print('Done')