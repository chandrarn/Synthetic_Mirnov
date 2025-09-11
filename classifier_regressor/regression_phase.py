#!/usr/bin/env python3

# Regression phase for ML mode id classifier
# Assumes defined region in sensor-frequency space, for a single mode, with amplitude and phase described separately

import numpy as np
import xarray as xr
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('font',**{'size':11})
matplotlib.rc('text', usetex=True)
matplotlib.use('TkAgg')
plt.ion()

def load_xarray_datasets(data_directory):
    """
    Load all xarray datasets from NetCDF files in the specified directory.
    """
    file_pattern = os.path.join(data_directory, "*.nc")
    files = glob.glob(file_pattern)
    datasets = []
    for file in files:
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

def extract_features_from_regions(real_matrices, imag_matrices, regions,normalize_real=True):
    """
    Extract features from the defined regions for each timepoint and mode.
    Updated to handle time-dependent regions: regions[mode][timepoint]
    Features: average value across region indices for all sensors, concatenated for real and imaginary.
    """
    features = []
    for time_idx, (real_mat, imag_mat) in enumerate(zip(real_matrices, imag_matrices)):
        mode_features = []
        for mode_idx, mode_regions in enumerate(regions):
            freq_idx, sensor_idx = mode_regions[time_idx]  # Get region for this mode and timepoint
            real_region = real_mat[np.ix_(freq_idx, sensor_idx)]
            imag_region = imag_mat[np.ix_(freq_idx, sensor_idx)]

            # Normalize if needed
            if normalize_real: real_region /= np.max(np.abs(real_region)) if np.max(np.abs(real_region)) != 0 else 1

            # Extract features: average across frequency for each sensor, then concatenate real and imag
            avg_real_per_sensor = np.mean(real_region, axis=0)  # Average across frequency (axis=0) for each sensor
            avg_imag_per_sensor = np.mean(imag_region, axis=0)  # Same for imag

            # Add pairwise differences (for simplicity, difference with the first sensor)
            if len(avg_real_per_sensor) > 1:
                diffs_real = avg_real_per_sensor[1:] - avg_real_per_sensor[0]
                diffs_imag = avg_imag_per_sensor[1:] - avg_imag_per_sensor[0]
                feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor, diffs_real, diffs_imag])
            else:
                feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor])

            mode_features.extend(feat)
        features.append(mode_features)
    return np.array(features)

def prepare_training_data(datasets, num_timepoints=10):
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
        time_indices = np.random.choice(len(ds['time']), num_timepoints, replace=False)
        
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
                    # Extract features
                    # Extract features: average across frequency for each sensor, then concatenate real and imag
                    avg_real_per_sensor = np.mean(real_region, axis=0)  # Average across frequency (axis=0) for each sensor
                    avg_imag_per_sensor = np.mean(imag_region, axis=0)  # Same for imag
                    
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

def train_regression_model(X, y_m, y_n, datasets, doPlot=True):
    """
    Train regression models for m and n.
    """
    # Split data
    X_train, X_test, y_m_train, y_m_test, y_n_train, y_n_test = train_test_split(
        X, y_m, y_n, test_size=0.2, random_state=42)
    
    # Train models
    model_m = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=-1)
    model_n = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=-1)
    
    model_m.fit(X_train, y_m_train)
    model_n.fit(X_train, y_n_train)
    
    # Evaluate
    pred_m = model_m.predict(X_test)
    pred_n = model_n.predict(X_test)
    
    rms_m = np.sqrt(mean_squared_error(y_m_test, pred_m))
    rms_n = np.sqrt(mean_squared_error(y_n_test, pred_n))
    
    print(f"MSE for m: {rms_m}")
    print(f"MSE for n: {rms_n}")

    if doPlot:
        f_name = f'Regression_results_N_{len(datasets)}_TP_{X.shape[0]}_FT_{X.shape[1]}_'+\
            f'Sensor_Set_{datasets[0].attrs['sensor_set']}_Mesh_file_{datasets[0].attrs['mesh_file']}'
        plt.close('regression_results')
        fig, ax = plt.subplots(2,1,num='regression_results',figsize=(5, 4),tight_layout=True)
        ax[0].scatter(y_m_test, pred_m, alpha=0.5,label=f'RMS={rms_m:.2f}')
        ax[0].scatter(y_m_train, model_m.predict(X_train), alpha=0.1,color='gray',\
                      label=f'Train data, RMS = {np.sqrt(mean_squared_error(y_m_train, model_m.predict(X_train))):.2f}')
        ax[0].plot([min(y_m_test), max(y_m_test)], [min(y_m_test), max(y_m_test)], 'r--')
        ax[0].set_xlabel('True m')
        ax[0].set_ylabel('Predicted m')
        ax[0].legend(fontsize=8)
        #ax[0].title('Mode Number m Prediction')

        ax[1].scatter(y_n_test, pred_n, alpha=0.5,label=f'RMS={rms_n:.2f}')
        ax[1].scatter(y_n_train, model_n.predict(X_train), alpha=0.1,color='gray', \
                      label=f'Train data, RMS = {np.sqrt(mean_squared_error(y_n_train, model_n.predict(X_train))):.2f}')
        ax[1].plot([min(y_n_test), max(y_n_test)], [min(y_n_test), max(y_n_test)], 'r--')
        
        ax[1].set_xlabel('True n')
        ax[1].set_ylabel('Predicted n')
        ax[1].legend(fontsize=8)
        #ax[1].title('Toroidal Mode Number n Prediction')

        for i in range(2):
            ax[i].grid()

        fig.savefig('../output_plots/'+f_name+'.pdf',transparent=True)

        plt.show()
    
    return model_m, model_n

if __name__ == "__main__":
    # Directory containing the saved xarray files
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/test/"  # Replace with actual path
    
    # Load datasets
    datasets = load_xarray_datasets(data_dir)
    
    # Prepare training data
    X, y_m, y_n = prepare_training_data(datasets, num_timepoints=10)
    
    # Train models
    model_m, model_n = train_regression_model(X, y_m, y_n, datasets)
    
    # Save models if needed
    # import joblib
    # joblib.dump(model_m, 'model_m.pkl')
    # joblib.dump(model_n, 'model_n.pkl')


