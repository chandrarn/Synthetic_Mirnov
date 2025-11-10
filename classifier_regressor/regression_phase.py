#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Regression phase for ML mode id classifier
# Assumes defined region in sensor-frequency space, for a single mode, with amplitude and phase described separately

import numpy as np
import xarray as xr
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler  # Add this import

plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
matplotlib.rc('font',**{'size':11})
matplotlib.rc('text', usetex=True)
# matplotlib.use('TkAgg')
plt.ion()

def load_xarray_datasets(data_directory, n_files=1):
    """
    Load all xarray datasets from NetCDF files in the specified directory.
    """
    file_pattern = os.path.join(data_directory, "*.nc")
    files = glob.glob(file_pattern)
    datasets = []
    if n_files == -1: n_files = len(files)
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

    # strip _ real and _ imag from sensor names
    sensor_names = [name[:-5] for name in sensor_names[::2]]

    return all_real_matrices, all_imag_matrices, sensor_names # Return only sensor names without _real/_imag

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

def prepare_training_data(datasets, num_timepoints=22, theta=None, phi=None):
    """
    Prepare training data from all datasets.
    Instead of padding, loop across time_indices and extract relevant elements only if frequency != 0.
    Collect features and targets in lists, then convert to arrays.
    """
    all_features = []
    all_targets_m = []
    all_targets_n = []

    diffs_real_all = []
    diffs_imag_all = []
    theta_diffs_all = []
    phi_diffs_all = []
    sensor_names_all = []
    for ds in datasets:
        # Select random timepoints
        time_indices = np.random.choice(len(ds['time']), num_timepoints, replace=False)
        
        # Extract matrices for each timepoint, real/imag for each sensor
        real_mats, imag_mats, sensor_names = extract_spectrogram_matrices(ds, time_indices)
        
        # Remove duplicates by converting to set and back to list
        sensor_names = list(set(sensor_names))
        
        # Filter sensor_names to only include those present in theta and phi
        sensor_names_filtered = [name for name in sensor_names if name in theta and name in phi]
        
        # Get indices of filtered sensors in the original sensor_names list
        sensor_indices = [sensor_names.index(name) for name in sensor_names_filtered]
        
        # # Slice the matrices to only include the filtered sensors
        # for time_idx in range(len(real_mats)):
        #     real_mats[time_idx] = real_mats[time_idx][:, sensor_indices]
        #     imag_mats[time_idx] = imag_mats[time_idx][:, sensor_indices]
        
        # Update sensor_names to the filtered list
        # Implicitly, this takes on the orderings that will go into the feature vector
        sensor_names = sensor_names_filtered
        sensor_names_all.extend(sensor_names)
        
        # Get theta and phi values for the filtered sensors
        theta_vals = np.array([theta[name] for name in sensor_names])
        phi_vals = np.array([phi[name] for name in sensor_names])
        theta_diff = theta_vals[1:] - theta_vals[0]
        phi_diff = phi_vals[1:] - phi_vals[0]

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


        # Loop across time_indices and modes, extract features only if frequency != 0
        for time_idx in range(len(time_indices)):
            for mode_idx in range(len(F_mode_vars)):

                freq_idx, sensor_idx = regions[mode_idx][time_idx]

               

                if len(freq_idx) > 0:  # Only if frequency != 0 (i.e., valid region)
                     # Sensor indices per region must only refer to sensors being used
                    sensor_idx = sensor_idx[sensor_indices]

                    real_region = real_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    imag_region = imag_mats[time_idx][np.ix_(freq_idx, sensor_idx)]
                    # Extract features
                    # Extract features: average across frequency for each sensor, then concatenate real and imag
                    
                    # Find the on average largest frequency channel
                    f_chan = np.argmax( np.mean( np.abs(real_region), axis=1) )
                    # Normalize real part of sensors based on this channel
                    real_region /= np.max( np.abs(real_region[f_chan, :]) ) 

                

                    avg_real_per_sensor = np.mean(real_region, axis=0)  # Average across frequency (axis=0) for each sensor
                    # Normalize real part of sensors
                    avg_real_per_sensor /= np.max(np.abs(avg_real_per_sensor)) 

                    avg_imag_per_sensor = np.mean(imag_region, axis=0)  # Same for imag
                    
                    # Add pairwise differences (for simplicity, difference with the first sensor)
                    diffs_real = avg_real_per_sensor[1:] - avg_real_per_sensor[0]
                    diffs_imag = avg_imag_per_sensor[1:] - avg_imag_per_sensor[0]
                    
                    if theta is None: 
                        feat = np.concatenate([ diffs_real, diffs_imag])
                    else:
                        diffs_real_all.append(diffs_real)  # Collect raw diffs (no per-sample norm yet)
                        diffs_imag_all.append(diffs_imag)
                        theta_diffs_all.append(theta_diff)
                        phi_diffs_all.append(phi_diff)
                        # feat = np.stack([diffs_real, diffs_imag, theta_vals[1:]-theta_vals[0], phi_vals[1:]-phi_vals[0]])
                    # if len(avg_real_per_sensor) > 1:
                    #     feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor, diffs_real, diffs_imag])
                    # else:
                    #     feat = np.concatenate([avg_real_per_sensor, avg_imag_per_sensor])
                    # Concat features separately, stack at the end
                  
                    if theta is None: all_features.append(feat)
                    all_targets_m.append(mode_m[mode_idx])
                    all_targets_n.append(mode_n[mode_idx])
    
    # Convert lists to numpy arrays
    # X should look like (num_samples, num sensors, num_features) where num_samples = total valid timepoints * modes across all datasets
    if theta is not None:
        # Apply scalar to saved feature data, to allow saving of fitted scalar
        # # After collecting all data, apply StandardScaler globally
        # scaler = StandardScaler()
        # # Stack into (N, 4) for scaling (N = total samples, 4 = features per sample)
        # all_diffs = np.stack([np.array(diffs_real_all), np.array(diffs_imag_all),
        #                       np.array(theta_diffs_all), np.array(phi_diffs_all)], axis=-1)  # (N, S-1, 4) assuming S-1 diffs
        # # Flatten to (N*(S-1), 4) for scaler, then reshape back
        # N, S_diff, C = all_diffs.shape
        # all_diffs_flat = all_diffs.reshape(-1, C)
        # scaled_diffs_flat = scaler.fit_transform(all_diffs_flat)
        # scaled_diffs = scaled_diffs_flat.reshape(N, S_diff, C)
        # # Unpack back to lists or arrays
        # diffs_real_all = scaled_diffs[:, :, 0].tolist()
        # diffs_imag_all = scaled_diffs[:, :, 1].tolist()
        # theta_diffs_all = scaled_diffs[:, :, 2].tolist()
        # phi_diffs_all = scaled_diffs[:, :, 3].tolist()

        X = np.stack([np.array(diffs_real_all), np.array(diffs_imag_all), \
                        np.array(theta_diffs_all), np.array(phi_diffs_all)], axis=-1 )
    else:
        X = np.array(all_features)
    y_m = np.array(all_targets_m)
    y_n = np.array(all_targets_n)
    
    # Only pull unique sensor names
    unique_ordered = list(dict.fromkeys(sensor_names_all))
    return X, y_m, y_n, unique_ordered

def train_regression_model(X, y_m, y_n):
    """
    Train regression models for m and n.
    """
    # Split data
    X_train, X_test, y_m_train, y_m_test, y_n_train, y_n_test = train_test_split(
        X, y_m, y_n, test_size=0.2, random_state=42)
    
    # # Train models
    # model_m = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=-1)
    # model_n = RandomForestRegressor(n_estimators=10000, random_state=42, n_jobs=-1)
    
    # Train models with XGBoost
    model_m = XGBRegressor(n_estimators=10000, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=10)
    model_n = XGBRegressor(n_estimators=10000, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=10)
    

    model_m.fit(X_train, y_m_train)
    model_n.fit(X_train, y_n_train)
    
    # Evaluate
    pred_m = model_m.predict(X_test)
    pred_n = model_n.predict(X_test)

    pred_m_train = model_m.predict(X_train)
    pred_n_train = model_n.predict(X_train)
    
    rms_m = np.sqrt(mean_squared_error(y_m_test, pred_m))
    rms_n = np.sqrt(mean_squared_error(y_n_test, pred_n))
    
    print(f"MSE for m: {rms_m}")
    print(f"MSE for n: {rms_n}")

    rms_m = f'RMS = {rms_m:.2f}'
    rms_n = f'RMS = {rms_n:.2f}'
    
    return model_m, model_n, pred_m, pred_n, \
        pred_m_train, pred_n_train, rms_m, rms_n,  y_m_train, y_m_test, y_n_train, y_n_test




##################################################################3
##########################################################################
# Classification attempt, instead of regression
def train_classification_model(X, y_m, y_n, doPlot=True):
    """
    Train classification models for m and n using XGBoost.
    Assumes y_m and y_n are categorical (will be encoded if not).
    """
    # Encode targets if they are not integers (e.g., strings or floats)
    le_m = LabelEncoder()
    le_n = LabelEncoder()
    y_m_encoded = le_m.fit_transform(y_m)
    y_n_encoded = le_n.fit_transform(y_n)
    
    # Split data
    X_train, X_test, y_m_train, y_m_test, y_n_train, y_n_test = train_test_split(
        X, y_m_encoded, y_n_encoded, test_size=0.2, random_state=42, stratify=y_n_encoded)  # Stratify for balance
    
    # Train models with XGBoost Classifier
    model_m = XGBClassifier(n_estimators=1000, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=10,
                            objective='multi:softprob', num_class=len(le_m.classes_))
    model_n = XGBClassifier(n_estimators=1000, random_state=42, n_jobs=-1, learning_rate=0.1, max_depth=10,
                            objective='multi:softprob', num_class=len(le_n.classes_))
    
    model_m.fit(X_train, y_m_train)
    model_n.fit(X_train, y_n_train)
    
    # Predict probabilities and classes
    pred_m_prob = model_m.predict_proba(X_test)
    pred_n_prob = model_n.predict_proba(X_test)
    pred_m_class = model_m.predict(X_test)
    pred_n_class = model_n.predict(X_test)
    train_m_class = model_m.predict(X_train)
    train_n_class = model_n.predict(X_train)
    
    # Evaluate
    acc_m = accuracy_score(y_m_test, pred_m_class)
    acc_n = accuracy_score(y_n_test, pred_n_class)
    auc_m = roc_auc_score(y_m_test, pred_m_prob, multi_class='ovr', average='macro')
    auc_n = roc_auc_score(y_n_test, pred_n_prob, multi_class='ovr', average='macro')
    
    print(f"Accuracy for m: {acc_m:.3f}")
    print(f"Accuracy for n: {acc_n:.3f}")
    print(f"AUC for m: {auc_m:.3f}")
    print(f"AUC for n: {auc_n:.3f}")
    
    if doPlot:
        # Plot confusion matrices or AUC curves (example for n)
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay.from_predictions(y_n_test, pred_n_class, ax=ax[0])
        ConfusionMatrixDisplay.from_predictions(y_m_test, pred_m_class, ax=ax[1])
        ax[0].set_title(f'$n$ Classification (Acc={acc_n:.3f}, AUC={auc_n:.3f})')
        ax[1].set_title(f'$m$ Classification (Acc={acc_m:.3f}, AUC={auc_m:.3f})')
        
        # For AUC, you could plot ROC curves per class
        # (Omitted for brevity; use sklearn's plot_roc_curve or manual plotting)
        
        plt.show()
    
    predicted_n = le_n.inverse_transform(pred_n_class)
    predicted_m = le_n.inverse_transform(pred_m_class)
    predicted_n_train = le_n.inverse_transform(train_n_class)
    predicted_m_train = le_n.inverse_transform(train_m_class)

    return model_m, model_n, predicted_m, predicted_n, \
        predicted_m_train, predicted_n_train, f'AUC = {auc_m:0.2f}', f'AUC = {auc_n:0.2f}',\
                y_m_train, y_m_test,  y_n_train, y_n_test

############################################################################
############################################################################
def plot_regression_results(y_m_train, y_m_test, y_n_train, y_n_test,X,pred_m,pred_m_train,\
                             pred_n,pred_n_train,error_m,error_n, fName_params,saveExt=''):
    f_name = f'Regression_results_XGB_N_{fName_params[N]}_TP_{X.shape[0]}_FT_{X.shape[1]}_'+\
        f'Sensor_Set_{fName_params[sensor_set]}_Mesh_file_{fName_params[mesh_file]}{saveExt}'
    
    plt.close('regression_results')
    fig, ax = plt.subplots(2,1,num='regression_results',figsize=(5, 4),tight_layout=True)

    ax[0].scatter(y_m_test, pred_m, alpha=0.5,label=f'Prediction, {error_m}')
    ax[0].scatter(y_m_train, pred_m_train, alpha=0.1,color='gray',\
                    label=f'Train data, RMS = {np.sqrt(mean_squared_error(y_m_train, pred_m_train )):.2f}')
    ax[0].plot([min(y_m_test), max(y_m_test)], [min(y_m_test), max(y_m_test)], 'r--')
    ax[0].set_xlabel('True m')
    ax[0].set_ylabel('Predicted m')
    ax[0].legend(fontsize=8)
    #ax[0].title('Mode Number m Prediction')

    ax[1].scatter(y_n_test, pred_n, alpha=0.5,label=f'Prediction, {error_n}')
    ax[1].scatter(y_n_train,pred_n_train, alpha=0.1,color='gray', \
                    label=f'Train data, RMS = {np.sqrt(mean_squared_error(y_n_train, pred_n_train )):.2f}')
    ax[1].plot([min(y_n_test), max(y_n_test)], [min(y_n_test), max(y_n_test)], 'r--')
    
    ax[1].set_xlabel('True n')
    ax[1].set_ylabel('Predicted n')
    ax[1].legend(fontsize=8)
    #ax[1].title('Toroidal Mode Number n Prediction')

    for i in range(2):
        ax[i].grid()

    fig.savefig('../output_plots/'+f_name+'.pdf',transparent=True)

    plt.show()

#############################################################################
def save_training_data(X, y_m, y_n, sensors_names, filename=None):
    """
    Save the training data to a .npz file.
    """
    np.savez(filename, X=X, y_m=y_m, y_n=y_n, sensor_names =sensors_names)
    print(f"Training data saved to {filename}")
############################################################################
def apply_Scalar_to_training_data(X, scaler):
    """
    Apply a fitted StandardScaler to the training data X.
    Assumes X shape is (num_samples, num_sensors, num_features).
    """
    num_samples, num_sensors, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)  # Flatten to (num_samples * num_sensors, num_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, num_sensors, num_features)  # Reshape back
    return X_scaled, scaler
############################################################################
def load_training_data(sensor_set, mesh_file, data_dir, theta=None, phi=None,n_files=1,\
                       forceDataReload=False):
    try: 
        if forceDataReload: raise SyntaxError
        dat = np.load(f'training_data_Sensor_Set_{sensor_set}_Mesh_file_{mesh_file}.npz')
        X, y_m, y_n, sensor_names  = dat['X'], dat['y_m'], dat['y_n'], dat['sensor_names']

        print(f"Loaded existing training data with {X.shape[0]} samples and {X.shape[1]} features")
    except:
        print("No existing training data found, generating new data...")
        
        
        # Load datasets
        datasets = load_xarray_datasets(data_dir,n_files=n_files)
        
        # Prepare training data
        X, y_m, y_n, sensor_names = prepare_training_data(datasets, num_timepoints=10, theta=theta, phi=phi)
        
        save_training_data(X, y_m, y_n, sensor_names, \
                           filename=f'training_data_Sensor_Set_{sensor_set}_Mesh_file_{mesh_file}.npz')

        print(f"Generated training data with {X.shape[0]} samples and {X.shape[1]} features")
    
    # Apply StandardScaler
    X, scaler = apply_Scalar_to_training_data(X, StandardScaler())

    return X, y_m, y_n, sensor_names, scaler
####################################################################
if __name__ == "__main__":
    # Directory containing the saved xarray files
    
    sensor_set = 'C_MOD_LIM'
    mesh_file = 'C_Mod_ThinCurr_Combined-homology.h5'
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/"  

    X, y_m, y_n,sensor_names,scaler = load_training_data(sensor_set, mesh_file, data_dir)

    # Train models
    fName_params = {'N': len(X), 'sensor_set': sensor_set, 'mesh_file': mesh_file}

    # model_m, model_n, pred_m, pred_n, \
    #     pred_m_train, pred_n_train,err_m, err_n,  y_m_train, y_m_test, y_n_train, y_n_tepth=6,est = \
    #         train_regression_model(X, y_m, y_n   )

    model_m, model_n, pred_m, pred_n,  pred_m_train, pred_n_train,\
          err_m, err_n,  y_m_train, y_m_test, y_n_train, y_n_test = \
              train_classification_model(X, y_m, y_n)


    plot_regression_results(y_m_train, y_m_test, y_n_train, y_n_test,X,pred_m,pred_m_train,\
                             pred_n,pred_n_train,err_m,err_n, fName_params,'_Classifier_low_mn')
    # Save models if needed
    # import joblib
    # joblib.dump(model_m, 'model_m.pkl')
    # joblib.dump(model_n, 'model_n.pkl')

    print('Done')


    """
    Write a new version of the regression_phase script, using a neural network instead of random forrest regression. Let the input have four channels: for each sensor, it will be given the real and imaginary components of a signal (as is done in prepare_training_data), but to that add the theta and phi direction coordinate of the sensor. For now, let the target vector be the n or m number, but not both at the same time. The network is being trained to recognize spatial patterns in the data, which are labeled by n and m interger numbers. As such, relationships between feature values are more important than their individual absolute values. We may wish to try a recurrent or reservior neural network for this reason. We may wish to use a Fourier Neutral Network kernal as a first test. The code to pull and prepare the real, imaginary data vectors and target vectors can remain as is, if needed
    """
