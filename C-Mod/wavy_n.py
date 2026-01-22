#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:27:44 2025
    Load in WavyStar filtered spectrogram
    option one: contour detect, get single n number for each blob
    option two: step through in time, allowing for time varying n
    
    ToDo: get tLim, fLim automatically
@author: rian
"""

from header_Cmod import plt, np, cv2, iio, sys, xr, os
from get_Cmod_Data import __loadData
sys.path.append('../signal_analysis/')

from estimate_n_improved import run_n
import matplotlib as mpl
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
sys.path.append('../classifier_regressor/')
from cnn_predictor import ConvSpatialNet, build_input_tensor,plot_feature_stats  # Import necessary classes/functions from cnn_predictor.py
from sklearn.preprocessing import LabelEncoder
from mirnov_Probe_Geometry import Mirnov_Geometry as Mirnov_Geometry_C_Mod  # Assuming this is available for sensor coords

from scipy.fft import fft

sys.path.append('../C-Mod/')
from header_Cmod import __doFilter, remove_freq_band


def mode_id_n(wavy_file='Spectrogram_C_Mod_Data_BP_1051202011_Wavy.png',
              tLim_orig=[.75,1.1],fLim_orig=[0,625],shotno=1051202011,
              f_band=[],doSave='',saveExt='',cLim=[], use_nn=False,\
                 min_contour_w=5, min_contour_h=5,doPlot=False,CNN=False,\
                     t_band=[],shotno_model=None, use_lr=False, fno_model_path=None, lr_model_path=None, **kwargs):  # Add use_lr parameter
    # Identify n# from segmented spectrogram associated with a shot
    
    # Load in png
    wavy, contour, hierarchy, c_out, tLim, fLim = \
        gen_contours(wavy_file,tLim_orig,fLim_orig, f_band,  t_band, \
                     min_contour_w,min_contour_h, )
    
    # step through contours, set filtering range
    filter_limits = gen_filter_ranges(c_out, tLim, fLim, wavy.shape[:2])
    
    # Get raw signal data (only if using NN, to avoid redundant loading)
    bp_k = __loadData(shotno, pullData=['bp_k'], forceReload=['bp_k'] * False)['bp_k']

    # Get model sensor names if using NN or LR
    if use_nn or use_lr:
        sensor_names_model_order = __loadData(shotno_model, pullData=['bp_k'])['bp_k'].names

    # Calculate n# in the filterbands identified
    n_opt_out = []
    features_out = []
    for ind, lims in enumerate(filter_limits):
        print('Time: %3.3f-%3.3f, Freq: %3.3f-%3.3f' % (*lims['Time'], *lims['Freq']))
        
        if use_nn:
            # Use NN predictor instead of run_n
            if CNN:
                n_opt = predict_n_with_nn(shotno, lims['Time'], lims['Freq'][0] * 1e3,\
                                       lims['Freq'][1] * 1e3, bp_k, **kwargs)
            else: 
                n_opt, feats = predict_n_with_fno(shotno, lims['Time'], lims['Freq'][0] * 1e3,\
                                       lims['Freq'][1] * 1e3, bp_k, sensor_names_model_order,
                                   fno_model_path=fno_model_path or '../output_models/fno_classifier_n.pth',
                                   **kwargs)
                features_out.append(feats)
        elif use_lr:
            # Use logistic regression predictor
            n_opt, feats = predict_n_with_lr(shotno, lims['Time'], lims['Freq'][0] * 1e3,\
                                       lims['Freq'][1] * 1e3, bp_k, sensor_names_model_order,
                                   lr_model_path=lr_model_path or '../output_models/logistic_regression_n.pkl',
                                   **kwargs)
            features_out.append(feats)
        else:
            # Original run_n call
            n_opt = run_n(shotno=shotno, tLim=lims['Time'], fLim=None, 
                          HP_Freq=lims['Freq'][0] * 1e3, LP_Freq=lims['Freq'][1] * 1e3,
                          n=[lims['Freq'][0]**(1/3)], doSave='', save_Ext='',
                          directLoad=True, tLim_plot=[], z_levels=[8,10,14], doBode=True,
                          bp_k=bp_k, doPlot=doPlot)[-1]
        filter_limits[ind]['n_opt'] = n_opt
        n_opt_out.append(n_opt)
        
        # if ind > 10: break
    
    # color contours (fill in? step through contour horizontally, fill vertically?)
    overplot_n(filter_limits,wavy, tLim, fLim,n_opt_out,doSave,
               saveExt+('_fno' if use_nn else ('_lr' if use_lr else '_linear')),shotno,cLim)
    
    # diagnose feature distributions
    plot_data_distributions(features_out,n_opt_out, saveExt='_%d'%shotno)

    return wavy, contour, hierarchy, c_out,filter_limits,n_opt_out
############################################################
def overplot_n(filter_limits,wavy,tRange,fRange,n_opt,doSave,saveExt,
               shotno,cLim):
    title = 'WavyStar_n_Estimator_%d%s'%(shotno,saveExt)
    plt.close(title)
    fig,ax = plt.subplots(1,1,num=title,tight_layout=True)
    

    
    ax.contourf(tRange,fRange,wavy,cmap=plt.get_cmap('Greys'),zorder=-5)
    
    cmap = mpl.cm.viridis
    bounds = np.arange( *(cLim if cLim else \
        [np.nanmin(n_opt),np.nanmax(n_opt)+(2 if np.nanmin(n_opt) == np.nanmax(n_opt) else 1)]) )

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    for ind,lims in enumerate(filter_limits):
        if np.isnan(n_opt[ind]):continue
        contours = lims['Contours']
        contours = np.vstack((contours,contours[0]))
        
        ax.plot(tRange[contours[:,0]],fRange[contours[:,1]],c=cmap(norm(n_opt[ind])) )
    
    ax.text(.04,.97,'%d'%shotno,transform=ax.transAxes,fontsize=8,
            verticalalignment='top',bbox={'boxstyle':'round','alpha':.7,
                                          'facecolor':'white'})
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             label=r'n\#',ax=ax)
    ax.set_rasterization_zorder(-1)
    
    
    if doSave: 
        fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',transparent=True)
        print('Saved: '+ doSave+fig.canvas.manager.get_window_title()+'.pdf')
    plt.show()
        
############################################################
def plot_data_distributions(features_out,n_opt, saveData=True,  saveExt=''):
    # Plot feature distributions for each predicted n#
    features_out = np.array(features_out)
    if features_out.size == 0:
        print('plot_data_distributions: no features to plot')
        return
    n_opt_arr = np.array(n_opt)
    if len(n_opt_arr) != features_out.shape[0]:
        print('plot_data_distributions: length mismatch between n_opt and features_out; skipping plot')
        return

    n_features = features_out.shape[2]
    
    plt.close('Scaled_Feature_Distributions'+saveExt)
    n_rows = 3
    n_cols = 2
    fig,ax = plt.subplots(n_rows,n_cols,num='Scaled_Feature_Distributions'+saveExt,
                         tight_layout=True,figsize=(7,6),sharex=True)
    ax = ax.flatten()

    # Feature labels
    feature_labels = [r'$\Delta$Mag FFT', r'$\sin(\Delta\angle\mathrm{FFT})$', r'$\cos(\Delta\angle\mathrm{FFT})$', r'$\Delta\theta$', r'$\Delta\phi$']
    # Unique n groups and color map
    # Filter out NaNs but keep index mapping
    unique_ns = [int(v) if np.isfinite(v) else np.nan for v in np.unique(n_opt_arr)]
    # Build color map for groups (skip NaN group coloring if present)
    cmap = plt.get_cmap('tab10')
    colors = {}
    color_i = 0
    for n_val in unique_ns:
        if np.isnan(n_val):
            continue
        colors[n_val] = cmap(color_i % 10)
        color_i += 1

    for i in range(n_features):
        # Plot each group in its own color
        handles = []
        labels = []
        for n_val in unique_ns:
            if np.isnan(n_val):
                # Optional: gray for NaNs
                idx = np.where(~np.isfinite(n_opt_arr))[0]
                color = (0.6,0.6,0.6,0.6)
                label = 'n=NaN'
            else:
                idx = np.where(n_opt_arr == n_val)[0]
                color = colors.get(n_val, cmap(0))
                label = f'n={int(n_val)}'
            if idx.size == 0:
                continue
            ax[i].plot(features_out[idx, :, i].T, color=color, alpha=0.7)
            # Add a proxy handle for legend once per group
            h, = ax[i].plot([], [], color=color, label=label)
            handles.append(h); labels.append(label)

        ax[i].set_xlabel(f'Feature {feature_labels[i]}')
        ax[i].set_ylabel('Value')
        ax[i].grid()
        if handles:
            ax[i].legend(handles=handles, labels=labels, fontsize=8, frameon=True, handlelength=1.5)
    # Hide unused axes
    for j in range(n_features, len(ax)):
        ax[j].set_visible(False)
    
    if saveExt:
        fig.savefig('../output_plots/Feature_Distributions'+saveExt+'.pdf',transparent=True)
        print('Saving: '+'../output_plots/Feature_Distributions'+saveExt+'.pdf')
    
    if saveData: np.savez('../data_output/wavy_n_features.npz',features_scaled=features_out,n_opt=n_opt)
    plt.show()
############################################################
def gen_filter_ranges(c_out,tLim,fLim,dimImage):
    # Convert coordinates from pixels to time, freq
    tRange = tLim # Already have the new limits #np.linspace(*tLim_orig,dimImage[0])
    fRange = fLim # png and .xr files will generate freq in kHz
    
    filter_limits = [] # structured as time, spatial limits, contour#
    for c in c_out:
        try:
            filter_limits.append(\
             {'Time':[tRange[np.min(c[:,0])], tRange[np.max(c[:,0])]],\
              'Freq':[fRange[np.min(c[:,1])], fRange[np.max(c[:,1])]],
              'Contours':c})
        except: 
            print(c)
            raise SyntaxError
    return filter_limits
############################################################
def gen_contours(wavy_file,tLim_orig=[],fLim_orig=[],f_band = [],
                 t_band = [], min_contour_w=5,min_contour_h=5,
                 doPlot=True,):
    # Load in the wavystar segmented data (in .png or .netcdf format)
    if '.png' in wavy_file: wavy, tLim, fLim = open_wavy_png(wavy_file, tLim_orig,fLim_orig)
    else:  wavy, tLim, fLim = open_wavy_xarray(wavy_file)

    if doPlot:
        plt.close('Input')
        fig,ax = plt.subplots(1,1,tight_layout=True,sharex=True,sharey=True,num='Input')
        ax.contourf(tLim,fLim,wavy[:,:])
    
    # Contour detect a la geqdsk cv2.findContours
    contour,hierarchy=cv2.findContours(wavy,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    c_out=[]
    for c in contour: 
        c=c.squeeze()
        # minimum size checks for valid contour
        if len(c)<=2:continue
        if np.max(c[:,0])-np.min(c[:,0])<min_contour_w: continue
        if np.max(c[:,1])-np.min(c[:,1])<min_contour_h: continue
        
        # frequency limit
        if f_band:
            if fLim[np.min(c[:,1])] < f_band[0]: continue
            if fLim[np.max(c[:,1])] > f_band[1]: continue
        if t_band:
            if tLim[np.min(c[:,0])] < t_band[0]: continue
            if tLim[np.max(c[:,0])] > t_band[1]: continue
    
        if doPlot: 
            ax.plot(tLim[c[:,0]],fLim[c[:,1]],'r-',lw=1)
            
        c_out.append(c)
    if doPlot:
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Frequency [kHz]')
        plt.show()
    return wavy, contour, hierarchy, c_out, tLim, fLim
################################################
def open_wavy_png(wavy_file, tLim_orig,fLim_orig):
    wavy = iio.imread('../output_plots/'+wavy_file)
    wavy=wavy[::-1,:,:]
    
    tLim = np.linspace(*tLim_orig,wavy.shape[0])
    fLim = np.linspace(*fLim_orig,wavy.shape[1])
    wavy = np.array(wavy[:,:,0]==253,dtype=np.uint8)

    return wavy, tLim, fLim 
def open_wavy_xarray(wave_file_nc):
    dataset = xr.open_dataset(wave_file_nc)
    tLim = dataset.coords['time'].values
    fLim = dataset.coords['frequency'].values*1e-3
    wavy = dataset.binary.values
    return wavy, tLim, fLim

####################################################
def load_and_filter_raw_signal(shotno, tLim, r_maj, HP_Freq, LP_Freq, bp_k=None,doDebug=False):
        # Load and filter data (similar to estimate_n_improved.py)
    bp_k = __loadData(shotno, pullData='bp_k', forceReload=['bp_k'] * False)['bp_k'] if bp_k is None else bp_k
    data = bp_k.data
    f_samp = bp_k.f_samp
    inds = np.arange(*[np.argmin(np.abs(bp_k.time - t)) for t in tLim])
    data = data[:, inds]
    time = bp_k.time[inds]
    phi = {sensor_name: bp_k.Phi[ind] * np.pi/180 for ind, sensor_name in enumerate(bp_k.names)}
    theta_dict = {sensor_name: np.arctan2(bp_k.Z[ind] - 0, bp_k.R[ind] - r_maj) for ind,sensor_name in enumerate(bp_k.names)}    # 
    R = {sensor_name: bp_k.R[ind] for ind, sensor_name in enumerate(bp_k.names)}
    Z = {sensor_name: bp_k.Z[ind] for ind, sensor_name in enumerate(bp_k.names)}

    # Filter data
    data = __doFilter(data.copy(), time, HP_Freq, LP_Freq)
    data = remove_freq_band(data, time, freq_rm=500e3, freq_sigma=50)
    
    # Normalize data (zero mean, unit variance)
    # For now we'll use simple standardization;
    # This will go into the scaler used during training
    data -= np.mean(data)
    data /= np.max(data)


    # check lengths
    if data.shape[1] != len(time):
        time = time[:data.shape[1]]

    # Compute FFT (real and imag components)
    fft_out = np.fft.rfft(data,axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], time[1]-time[0])

    # Find dominant frequency in the FFT band (HP_Freq to LP_Freq)
    band_mask = (freqs >= HP_Freq) & (freqs <= LP_Freq)
    while not np.any(band_mask):
        HP_Freq *= 0.9
        LP_Freq *= 1.1
        band_mask = (freqs >= HP_Freq) & (freqs <= LP_Freq)
        print(f'Adjusted filter band to {HP_Freq}-{LP_Freq} Hz to find valid frequencies.')

    mag = np.abs(fft_out)  # Magnitude of FFT
    avg_mag = np.mean(mag, axis=0)  # Average magnitude across sensors
    band_avg_mag = avg_mag[band_mask]
    dominant_idx_in_band = np.argmax(band_avg_mag)
    target_freq_idx = np.where(band_mask)[0][dominant_idx_in_band]
    
    real_comp = np.abs(fft_out[:, target_freq_idx])
    real_comp /= np.max(real_comp)  # Normalize real part
    imag_comp = np.angle(fft_out[:, target_freq_idx])

    if doDebug:
        plt.close('Debug_Loaded_Signals')
        fig,ax = plt.subplots(1,1,num='Debug_Loaded_Signals',tight_layout=True,figsize=(6,2.5))
        ax.plot(freqs*1e-3,avg_mag); 
        ax.plot(freqs[band_mask]*1e-3,band_avg_mag,alpha=.6);
        ax.plot(freqs[target_freq_idx]*1e-3,np.mean(np.abs(fft_out[:,target_freq_idx])),'k*',\
                label=r'Chosen Freq: %2.1f kHz, $\Delta$T: %1.3f-%1.3f'%\
                    (freqs[target_freq_idx]*1e-3,tLim[0],tLim[1]));
        ax.grid()
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Avg FFT Magnitude')
        ax.set_xlim([0,LP_Freq*1.5*1e-3])
        ax.legend(fontsize=8)
        fig.savefig('../output_plots/Debug_Loaded_Signals_FFT.pdf',transparent=True)
        plt.show()

    return real_comp, imag_comp, phi, theta_dict
###############################################3
def predict_n_with_nn(shotno, tLim, HP_Freq, LP_Freq,bp_k, model_path='../output_plots/trained_mode_classifier_n.pth',
                      sensor_set='C_MOD_LIM', r_maj=0.68, cmod_shot=1051202011, zero_basing=False):
    """
    Helper function to predict n# using the trained NN model.
    Loads shot data, filters it, computes FFT real/imag, and predicts n.
    """
    # Load the PyTorch model
    # Weights_only=False to load scaler as well
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'),weights_only=False)
    print('Loaded model from ', model_path)
    model_state = checkpoint['model_state']
    label_classes = checkpoint['label_classes']
    sensor_names = checkpoint.get('sensor_names', None) # Pull sensor names associated with model
    le = LabelEncoder()
    le.classes_ = np.array(label_classes)
    n_classes = len(label_classes)
    
    # Initialize model (assuming ConvSpatialNet from cnn_predictor.py)
    model = ConvSpatialNet(in_channels=4, hidden_channels=128, n_classes=n_classes, dropout=0.4)  # Match training params
    model.load_state_dict(model_state)
    model.eval()
    
    # Load sensor coordinates (phi, theta_pol, R, Z)
    # phi, theta_pol, R, Z = Mirnov_Geometry_C_Mod(cmod_shot)
    # theta_dict = {sensor_name: np.arctan2(Z[sensor_name] - 0, R[sensor_name] - r_maj) for sensor_name in R}  # Adjust zmagx/rmagx as needed
    
    # Load and filter raw signal
    real_comp, imag_comp, phi, theta_dict = load_and_filter_raw_signal(shotno, tLim, r_maj, HP_Freq, LP_Freq, bp_k)

    # Filter to common sensors between sensor_names (upper) and bp_k.names (lower)
    common_sensors_lower = [name.lower() for name in sensor_names if name.lower() in bp_k.names]
    # common_sensors_lower = [name.lower() for name in common_sensors_upper]
    indices = [bp_k.names.index(name) for name in common_sensors_lower]

    # Slice real_comp and imag_comp to only include common sensors
    real_comp = real_comp[indices]
    imag_comp = imag_comp[indices]
    
    # Build input tensor (N=1, S=num_sensors, 4)
    # sensor_names = list(R.keys())  # Assuming R.keys() are sensor names
    theta = np.array([theta_dict[name.lower()] for name in sensor_names if name.lower() in theta_dict])
    phi_arr = np.array([load_and_filter_raw_signalphi[name.lower()] for name in sensor_names if name.lower() in phi])

    # Compute pairwise differences relative to the first sensor
    diffs_real = real_comp[1:] - real_comp[0]
    delta_imag = imag_comp[1:] - imag_comp[0] *(1 if  zero_basing else 0)
    delta_imag = np.angle(np.exp(1j * delta_imag))  # wrap-aware to (-pi, pi]
    sin_imag = np.sin(delta_imag)
    cos_imag = np.cos(delta_imag)

    theta_diff = theta[1:] - theta[0]
    phi_diff = phi_arr[1:] - phi_arr[0]

   
    # Load scaler from checkpoint
    scaler = checkpoint['scaler']

    # After computing diffs_real, diffs_imag, theta_diff, phi_diff
    # Stack diffs
    diffs = np.stack([diffs_real, sin_imag, cos_imag, theta_diff, phi_diff], axis=-1)  # (S-1, 5)

    # Apply the same standard scaling as in training
    scaled_diffs = scaler.transform(diffs)

    # Set X to the scaled diffs
    X = scaled_diffs[np.newaxis, :, :]  # (1, S-1, 5)

    # Plot feature stats after scaling
    # plot_feature_stats(X, sensor_names, do_n=True, save_ext=f'_data_{shotno}')
    
    # Predict
    with torch.no_grad():
        output = model(torch.from_numpy(X).float())
        pred_class = output.argmax(dim=1).item()
        predicted_n = le.inverse_transform([pred_class])[0]
    print('Output Probabilities: ', ['n=%d: %2.2f'%(le.inverse_transform([i])[0], p*100)\
                    for i,p in enumerate(torch.softmax(output,dim=1).squeeze())])        
    return predicted_n

########################################################################################################3
def load_fno_model(model_path='../output_models/fno_classifier_n.pth'):
    # Import FNO model definition from classifier_regressor
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..'))
    cls_reg_path = os.path.join(repo_root, 'classifier_regressor')
    if cls_reg_path not in sys.path:
        sys.path.append(cls_reg_path)
    try:
        from fno_predictor import FNO1dClassifier  # type: ignore
    except Exception as e:
        raise ImportError(f"Could not import FNO1dClassifier from {cls_reg_path}: {e}")

    # Load checkpoint (contains model weights, classes, scaler, and expected sensor order)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    print(f"Loaded FNO checkpoint from {model_path}")

    state_dict = checkpoint.get('state_dict', None)
    classes = checkpoint.get('classes', None)
    model_sensor_names = checkpoint.get('sensor_names', None)
    scaler = checkpoint.get('scaler', None)
    if state_dict is None or classes is None or model_sensor_names is None or scaler is None:
        raise ValueError("Checkpoint missing one of required keys: 'state_dict', 'classes', 'sensor_names', 'scaler'")

    # Infer architecture hyperparameters from the state_dict
    try:
        width = int(state_dict['lift.weight'].shape[0])
        # Determine depth by counting unique block indices
        block_indices = set()
        for k in state_dict.keys():
            if k.startswith('blocks.'):
                parts = k.split('.')
                if len(parts) > 1 and parts[1].isdigit():
                    block_indices.add(int(parts[1]))
        depth = max(block_indices) + 1 if block_indices else 4
        # Modes from first spectral conv weight
        # Find a key like 'blocks.0.spectral.weight'
        spectral_keys = [k for k in state_dict.keys() if k.endswith('spectral.weight')]
        if spectral_keys:
            modes = int(state_dict[spectral_keys[0]].shape[-1])
        else:
            modes = 16
        n_classes = int(state_dict['head.6.weight'].shape[0]) if 'head.6.weight' in state_dict else len(classes)
    except Exception as e:
        # Fallback to common defaults if shape introspection fails
        print(f"Warning: could not infer all hyperparameters from state_dict: {e}. Using defaults.")
        width, depth, modes = 128, 4, 16
        n_classes = len(classes)

    # Infer input channels from checkpoint (lift.weight shape = [width, in_channels, 1])
    in_channels = int(state_dict.get('lift.weight', torch.empty(width, 5, 1)).shape[1])
    model = FNO1dClassifier(in_channels=in_channels, width=width, modes=modes, depth=depth,
                            n_classes=n_classes, dropout=0.4)
    model.load_state_dict(state_dict)
    model.eval()

    return model, classes, model_sensor_names, scaler
####################################################################################################
def align_sensors_to_model(real_comp, imag_comp, phi, theta_dict,\
                            model_sensor_names, bp_k_names,\
                            sincos_imag: bool = False,\
                            sincos_only: bool = True, zero_baseline: bool = True) -> np.ndarray:
      # Align to model's expected sensor order; fill missing with zeros, warn once
    bp_names_lower = [nm for nm in bp_k_names]
    S = len(model_sensor_names)
    real_aligned = np.zeros(S, dtype=np.float32)
    imag_aligned = np.zeros(S, dtype=np.float32)
    theta_aligned = np.zeros(S, dtype=np.float32)
    phi_aligned = np.zeros(S, dtype=np.float32)
    missing = []
    for i, nm in enumerate(model_sensor_names):
        nml = str(nm).lower()
        if nml in bp_names_lower:
            j = bp_names_lower.index(nml)
            real_aligned[i] = float(real_comp[j]) 
            imag_aligned[i] = float(imag_comp[j])
        else:
            missing.append([nml,i])
        # Geometry lookups
        theta_aligned[i] = float(theta_dict.get(nml, 0.0))
        phi_aligned[i] = float(phi.get(nml, 0.0)) #* np.pi / 180.0  # Convert to radians
    if missing:
        print(f"Warning: {len(missing)} sensors missing in bp_k; filled with zeros: {missing[:4]}{'...' if len(missing)>4 else ''}")

    # Build difference features relative to the first sensor (S, 5)
    diffs_real = real_aligned - real_aligned[0]
    delta_imag = imag_aligned - imag_aligned[0] * (1 if zero_baseline else 0)
    delta_imag = np.angle(np.exp(1j * delta_imag))
    # delta_imag *= __check_angle_slope(delta_imag)  # Ensure consistent slope direction

    sin_imag = np.sin(delta_imag)
    cos_imag = np.cos(delta_imag)
    diffs_theta = theta_aligned - theta_aligned[0]
    diffs_phi = phi_aligned - phi_aligned[0]
    if sincos_only: 
        feats = np.stack([ sin_imag, cos_imag], axis=1)  # (S,2)
    elif sincos_imag:
        feats = np.stack([diffs_real, sin_imag, cos_imag, diffs_theta, diffs_phi], axis=1)  # (S,5)
    else: 
        feats = np.stack([diffs_real, delta_imag, diffs_theta, diffs_phi], axis=1)  # (S,4)
    # Ensure that missing sensors are zeroed out in features
    for nml, i in missing: feats[i, :] = 0.0

    return feats

def __check_angle_slope(delta_imag: np.ndarray, n_sensors: int = 5) -> int:
    # Check if angle differences are on-average increasing or decreasing
    diffs = np.diff(delta_imag)[:n_sensors]
    return np.sign(np.mean(diffs)) * -1 
#######################################################################################################
def plot_features_raw(feats, savePath=''):
    plt.close('Raw_Features_Debug')
    fig,ax = plt.subplots(3,2,num='Raw_Features_Debug',tight_layout=True,figsize=(7,6))
    feature_labels = [r'$\Delta$Mag FFT', r'$\sin(\Delta\angle\mathrm{FFT})$', r'$\cos(\Delta\angle\mathrm{FFT})$', r'$\Delta\theta$', r'$\Delta\phi$']
    ax = ax.flatten()
    for i in range(min(feats.shape[1], len(ax))):
        ax[i].plot(feats[:,i], '-')
        ax[i].set_xlabel('Sensor Index')
        ax[i].set_ylabel(feature_labels[i])
        ax[i].grid()
    for j in range(feats.shape[1], len(ax)):
        ax[j].set_visible(False)
    plt.show()

    if savePath:  fig.savefig(savePath,transparent=True)
####################################################################################################
###################################################################################################33
def predict_n_with_fno(shotno, tLim, HP_Freq, LP_Freq, bp_k, sensor_names_model_order,\
                       fno_model_path='../output_models/fno_classifier_n.pth',
                       r_maj: float = 0.68):
    """
    Predict toroidal mode number n using the trained FNO classifier on real C-Mod data.

    Steps:
    - Load checkpoint (expects keys: 'state_dict', 'classes', 'sensor_names', 'scaler').
    - Load and bandpass/filter raw Mirnov data, then take dominant FFT component in band.
    - Align sensors to the model's expected order; fill missing sensors with zeros if needed.
    - Build difference features relative to the first sensor and apply saved StandardScaler.
    - Run FNO model and map predicted class back to physical n via LabelEncoder(classes).

    Args:
        shotno: C-Mod shot number
        tLim: (t_start, t_end) time window in seconds
        HP_Freq: high-pass frequency (Hz)
        LP_Freq: low-pass frequency (Hz)
        bp_k: bp_k structure from __loadData (must include .names)
        model_path: path to saved FNO checkpoint
        r_maj: major radius used for theta computation consistency

    Returns:
        predicted_n (int)
    """



    # Load FNO model and associated info
    model, classes, model_sensor_names, scaler = load_fno_model(fno_model_path)


    # Pull real data features
    # Note: load_and_filter_raw_signal returns (real_comp, imag_comp, phi, theta_dict)
    real_comp, imag_comp, phi, theta_dict = load_and_filter_raw_signal(shotno, tLim, r_maj, HP_Freq, LP_Freq, bp_k)
    real_comp *= .5e-4 / np.max(real_comp)  # Scale real part to match training data range


    # Align sensors to model's expected order; fill missing with zeros
    feats = align_sensors_to_model(real_comp, imag_comp, phi, theta_dict, model_sensor_names,\
                                    sensor_names_model_order)
    
    ## Debug plots
    # plot_features_raw(feats,'../output_plots/Debug_FNO_Features_Shot_%d.pdf'%shotno)

    # Debug: print feature stats before scaling
    raw_min = feats.min(axis=0); raw_max = feats.max(axis=0); raw_med = np.median(feats, axis=0)
    print("Raw diffs stats per channel: min", raw_min, "max", raw_max, "median", raw_med)

    z = (feats - scaler.mean_) / scaler.scale_
    z_min = z.min(axis=0); z_max = z.max(axis=0); z_med = np.median(z, axis=0)
    print("Z-score stats per channel: min", z_min, "max", z_max, "median", z_med)
    print("Fraction |z|>6 per channel:", np.mean(np.abs(z) > 6, axis=0))
    # Apply saved scaler (fit on flattened (N*S, 4))
    feats_scaled = scaler.transform(feats)
    X = feats_scaled[np.newaxis, :, :]  # (1, S, 4)


    


    # Predict
    with torch.no_grad():
        logits = model(torch.from_numpy(X).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_idx = int(np.argmax(probs))
    le = LabelEncoder(); le.classes_ = np.array(classes)
    predicted_n = int(le.inverse_transform([pred_idx])[0])
    print('Output Probabilities: ', [f"n={int(le.inverse_transform([i])[0])}: {p*100:.2f}" for i, p in enumerate(probs)])
    return predicted_n, feats_scaled
######################################################################################################
def load_lr_model(lr_model_path='../output_models/logistic_regression_n.pkl'):
    """Load logistic regression model checkpoint."""
    import joblib, os
    if not os.path.exists(lr_model_path):
        raise FileNotFoundError(f"LR model not found: {lr_model_path}")
    if not lr_model_path.endswith('.pkl'):
        raise ValueError(f"LR model must be a .pkl saved with joblib.dump, got: {lr_model_path}")
    checkpoint = joblib.load(lr_model_path)
    return checkpoint['model'], checkpoint['scaler'], checkpoint['label_encoder'], checkpoint['sensor_names']

def predict_n_with_lr(shotno, tLim, HP_Freq, LP_Freq, bp_k, sensor_names_model_order,
                      lr_model_path='../output_models/logistic_regression_n.pkl',
                      r_maj: float = 0.68):
    """
    Predict toroidal mode number n using logistic regression on real C-Mod data.
    
    Args:
        shotno: C-Mod shot number
        tLim: (t_start, t_end) time window in seconds
        HP_Freq: high-pass frequency (Hz)
        LP_Freq: low-pass frequency (Hz)
        bp_k: bp_k structure from __loadData (must include .names)
        sensor_names_model_order: Sensor names in model's expected order
        model_path: path to saved logistic regression checkpoint
        r_maj: major radius used for theta computation
        
    Returns:
        predicted_n (int), feats_scaled (array)
    """
    # Load model and scaler
    lr_model, scaler, le, model_sensor_names = load_lr_model(lr_model_path)
    
    # Pull real data features
    real_comp, imag_comp, phi, theta_dict = load_and_filter_raw_signal(shotno, tLim, r_maj, HP_Freq, LP_Freq, bp_k)
    real_comp *= .5e-4 / np.max(real_comp)  # Scale real part to match training data range
    
    # Align sensors to model's expected order; fill missing with zeros
    feats = align_sensors_to_model(real_comp, imag_comp, phi, theta_dict, model_sensor_names,
                                    sensor_names_model_order)
    
    # Debug: print feature stats before scaling
    raw_min = feats.min(axis=0); raw_max = feats.max(axis=0); raw_med = np.median(feats, axis=0)
    print("Raw diffs stats per channel: min", raw_min, "max", raw_max, "median", raw_med)
    
    z = (feats - scaler.mean_) / scaler.scale_
    z_min = z.min(axis=0); z_max = z.max(axis=0); z_med = np.median(z, axis=0)
    print("Z-score stats per channel: min", z_min, "max", z_max, "median", z_med)
    print("Fraction |z|>6 per channel:", np.mean(np.abs(z) > 6, axis=0))
    
    # Apply scaler (flatten for sklearn)
    feats_scaled = scaler.transform(feats)  # (S, 4)
    X_flat = feats_scaled.reshape(1, -1)  # (1, S*4)
    
    # Predict
    pred_idx = int(lr_model.predict(X_flat)[0])
    predicted_n = int(le.inverse_transform([pred_idx])[0])
    probs = lr_model.predict_proba(X_flat)[0]
    
    print('Output Probabilities: ', [f"n={int(le.inverse_transform([i])[0])}: {p*100:.2f}" for i, p in enumerate(probs)])
    return predicted_n, feats_scaled
######################################################################################################

if __name__ == '__main__':
    # Example usage with NN
    # data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/new_hlicity_low_mn/"
    cmod_shot = 1160714026#1110316018#1160930034
    cmod_shot_model = 1160930034
    # cmod_shot = 1110316031#1051202011 
    # save_ext = '_test_Dropout0.2_Hidden256_Mask0.1'
    # out= mode_id_n(wavy_file='/home/rianc/Documents/Synthetic_Mirnov/output_plots/training_plots/' +\
    #           f's{cmod_shot}.nc', shotno=cmod_shot, doSave='../output_plots/', use_nn=True,
    #           model_path=data_dir+f'trained_mode_classifier_C-Mod_Sensors_Shot_{cmod_shot}_n{save_ext}.pth',
    #           doPlot=False)
    wavy_file='/home/rianc/Documents/Synthetic_Mirnov/output_plots/training_plots/' +\
              f's{cmod_shot}.nc'
    save_ext = '_FNO_C-Mod_Sensors_Reduced_n'

    f_band = [1,40]
    t_band =[]# [1.1,1.5]#[1.03,1.09]#[1.391,1.41]

    save_ext = ''
    out = mode_id_n(
        wavy_file=wavy_file, shotno=cmod_shot, doSave='../output_plots/', use_nn=True,
        fno_model_path=f'../output_models/fno_classifier_n_Sensor_Reduced_{cmod_shot_model}.pth',
        lr_model_path=f'../output_models/logistic_regression_n_Sensor_Reduced_{cmod_shot_model}.pkl',
        doPlot=True, f_band=f_band, t_band=t_band, min_contour_w=2, min_contour_h=3, saveExt=save_ext,
        shotno_model=cmod_shot_model, use_lr=False,
    )
    
    print('Finished example usage.')