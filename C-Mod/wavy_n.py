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

from header_Cmod import plt, np, cv2, iio, sys, xr
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

sys.path.append('../C-Mod/')
from header_Cmod import __doFilter, remove_freq_band


def mode_id_n(wavy_file='Spectrogram_C_Mod_Data_BP_1051202011_Wavy.png',
              tLim_orig=[.75,1.1],fLim_orig=[0,625],shotno=1051202011,
              f_band=[],doSave='',saveExt='',cLim=[], use_nn=False,\
                 min_contour_w=5, min_contour_h=5,doPlot=False, **kwargs):  # Add use_nn parameter
    # Identify n# from segmented spectrogram associated with a shot
    
    # Load in png
    wavy, contour, hierarchy, c_out, tLim, fLim = \
        gen_contours(wavy_file,tLim_orig,fLim_orig, f_band, min_contour_w,min_contour_h)
    
    # step through contours, set filtering range
    filter_limits = gen_filter_ranges(c_out, tLim, fLim, wavy.shape[:2])
    
    # Get raw signal data (only if using NN, to avoid redundant loading)
    # if use_nn:
    bp_k = __loadData(shotno, pullData=['bp_k'], forceReload=['bp_k'] * False)['bp_k']

    # Calculate n# in the filterbands identified
    n_opt_out = []
    for ind, lims in enumerate(filter_limits):
        print('Time: %3.3f-%3.3f, Freq: %3.3f-%3.3f' % (*lims['Time'], *lims['Freq']))
        
        if use_nn:
            # Use NN predictor instead of run_n
            n_opt = predict_n_with_nn(shotno, lims['Time'], lims['Freq'][0] * 1e3,\
                                       lims['Freq'][1] * 1e3, bp_k, **kwargs)
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
               saveExt+('_nn' if use_nn else '_linear'),shotno,cLim)
    
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
                 min_contour_w=5,min_contour_h=5,
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
###############################################3
def predict_n_with_nn(shotno, tLim, HP_Freq, LP_Freq,bp_k, model_path='../output_plots/trained_mode_classifier_n.pth',
                      sensor_set='C_MOD_LIM', r_maj=0.68, cmod_shot=1051202011):
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
    
    # Load and filter data (similar to estimate_n_improved.py)
    bp_k = __loadData(shotno, pullData='bp_k', forceReload=['bp_k'] * False)['bp_k'] if bp_k is None else bp_k
    data = bp_k.data
    f_samp = bp_k.f_samp
    inds = np.arange(*[np.argmin(np.abs(bp_k.time - t)) for t in tLim])
    data = data[:, inds]
    time = bp_k.time[inds]
    phi = {sensor_name: bp_k.Phi[ind] for ind, sensor_name in enumerate(bp_k.names)}
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
    from scipy.fft import fft
    fft_out = fft(data, axis=1) / len(time)
    freqs = np.fft.fftfreq(len(time), d=1/f_samp)
    
    # Find dominant frequency in the FFT band (HP_Freq to LP_Freq)
    band_mask = (freqs >= HP_Freq) & (freqs <= LP_Freq)
    mag = np.abs(fft_out)  # Magnitude of FFT
    avg_mag = np.mean(mag, axis=0)  # Average magnitude across sensors
    band_avg_mag = avg_mag[band_mask]
    dominant_idx_in_band = np.argmax(band_avg_mag)
    target_freq_idx = np.where(band_mask)[0][dominant_idx_in_band]
    
    real_comp = np.abs(fft_out[:, target_freq_idx])
    real_comp /= np.max(real_comp)  # Normalize real part
    imag_comp = np.angle(fft_out[:, target_freq_idx])

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
    phi_arr = np.array([phi[name.lower()] for name in sensor_names if name.lower() in phi])

    # Compute pairwise differences relative to the first sensor
    diffs_real = real_comp[1:] - real_comp[0]
    diffs_imag = imag_comp[1:] - imag_comp[0]
    theta_diff = theta[1:] - theta[0]
    phi_diff = phi_arr[1:] - phi_arr[0]

   
    # Load scaler from checkpoint
    scaler = checkpoint['scaler']

    # After computing diffs_real, diffs_imag, theta_diff, phi_diff
    # Stack diffs
    diffs = np.stack([diffs_real, diffs_imag, theta_diff, phi_diff], axis=-1)  # (S-1, 4)

    # Apply the same standard scaling as in training
    scaled_diffs = scaler.transform(diffs)

    # Set X to the scaled diffs
    X = scaled_diffs[np.newaxis, :, :]  # (1, S-1, 4)

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


if __name__ == '__main__':
    # Example usage with NN
    data_dir = "/home/rianc/Documents/Synthetic_Mirnov/data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/"
    cmod_shot = 1160714026#1110316018#1160930033
    save_ext = '_test_Dropout0.2_Hidden256_Mask0.1'
    out= mode_id_n(wavy_file='/home/rianc/Documents/Synthetic_Mirnov/output_plots/training_plots/' +\
              f's{cmod_shot}.nc', shotno=cmod_shot, doSave='../output_plots/', use_nn=True,
              model_path=data_dir+f'trained_mode_classifier_C-Mod_Sensors_Shot_{cmod_shot}_n{save_ext}.pth',
              doPlot=False)
    
    print('Finished example usage.')