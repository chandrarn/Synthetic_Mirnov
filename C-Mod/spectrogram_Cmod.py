#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:21:32 2025
    C-Mod data spectrogram
    
@author: rianc
"""

from header_Cmod import np, plt, Normalize, cm, rolling_spectrogram, __doFilter, \
    gaussianHighPassFilter, rolling_spectrogram_improved, grouped_average, \
        gaussian_filter, downscale_local_mean


import get_Cmod_Data as gC

###############################################################################

def signal_spectrogram_C_Mod(shotno=1051202011,sensor_set='BP',diag=None,
                            doSave='',pad=800,fft_window=400,tLim=[.75,1.1],
                            signal_reduce=2,f_lim=None,sensor_name='BP2T_GHK',
                            debug=True,plot_reduce=(4,1),doColorbar=True,
                            clabel='',HP_Freq=100,cLim=None,
                            doSave_Extractor=True,block_reduce=[200,40000],
                            data_archive='',cmap='viridis',figsize=(6,6),
                            params={},save_Ext='',batch=False,sigma_plot_reduce=(3,5)):
    '''
    

    Parameters
    ----------
    shotno : INT
        Shot number. The default is 1051202011.
    sensor_set : STR
        Which diagnostic to operate on. The default is 'BP'.
    diag : OBJ, optional
        Data object for sensor, can be reinput for faster loading. The default is None.
    doSave : Str, optional
        File in which to save plots. The default is ''.
    pad : INT
        Pad signal with zeros (increases FFT resolution). The default is 1000.
    fft_window : INT
        FFT smoothing window width. The default is 500.
    tLim : [INT,INT]
        Time range to operate on, in [s]. The default is [.75,1.1].
    signal_reduce : INT, optional
        Downsample diagnostic signal. The default is 2.
    f_lim : [FLOAT, FLOAT], optional
        Frequency range for plot. The default is None.
    sensor_name : STR, optional
        Name of specific sensor within diagnostic if needed. The default is 'BP2T_ABK'.
    debug : BOOL, optional
        Debug output printing. The default is True.
    plot_reduce : INT, optional
        Downsample spectrogram for plotting. The default is 3.
    doColorbar : BOOL, optional
        Add colorbar. The default is True.
    clabel : STR, optional
        Colorbar label. The default is r'$\tilde{\mathrm{B}}_\theta$ [G]'.
    doSave_Extractor : BOOL, optional
        Save .png without axes labels/etc, for feature extraction. The default is True.
    block_reduce : [INT, INT]
        Reduce sample in blocks (preserves high frequency signals, at the cost o
         introducing aliasing "noise"): cut the signal in blocks of [keep samples, drop samples]

    Returns
    -------
    diag : OBJ
        Diagnostic data output, returned for faster access.
    signals : ARRAY
        Raw signals.
    time : ARRAY
        Time vector.
    out_spect : ARRAY
        Spectrogram.
    freq : ARRAY
        Frequency vector.

    '''
    
    if tLim[1]-tLim[0]>.3: 
        print('Warning! The requested time range may be too' + \
            ' large to plot, consider increasing signal or plot reduction term')
    
    # Pull and select data
    signals, time, clabel,diag = get_data(shotno,diag,sensor_name,sensor_set,data_archive,debug)
    
    # # Desired time range to operate on
    # t_inds = np.arange((tLim[0]-time[0])/(time[1]-time[0]),\
    #           (tLim[1]-time[0])/(time[1]-time[0]),dtype=int)[::signal_reduce]
        
    t_inds = np.arange(*[np.argmin((time-t)**2) for t in tLim],dtype=int)
    
    
    signals = __doFilter(signals[:,t_inds], time[t_inds], HP_Freq, None)
    sampling_rate = 1/(time[1]-time[0])
    n_samples = block_reduce[0]
    m_skip = block_reduce[1]
    nfft = n_samples + pad
    freq, time, out_spect = compute_averaged_spectrogram_from_blocks(\
        signals, sampling_rate, n_samples, m_skip, window_type='hanning', nfft=nfft)
    
    #return out_spect, time, freq
    '''
    if block_reduce:
        del_inds=np.array([],dtype=int)
        for ind in np.arange(0,len(t_inds)-block_reduce[0],np.sum(block_reduce)):
            del_inds = np.append(del_inds,np.arange(ind,ind+block_reduce[0],dtype=int))
        #return t_inds,del_inds
        t_inds = np.delete(t_inds,del_inds)
    
    # Run filtering 
    signals = downscale_local_mean(signals[:,t_inds],signal_reduce)
    time = downscale_local_mean(time[t_inds], signal_reduce)
    
    signals = __doFilter(signals, time, HP_Freq, None)
    if debug: print('Finished Filter')
    
    #return t_inds,time, signals
    # Run spectrogram
    # TODO: issue: performing the spectrogram all at once fails due to memory limitations
    # # Could change to running on chunks
    # time_=[];out_spect=[]
    # for t_Start in np.arange(0,len(t_inds),1e5,dtype=int):
    #return t_inds, time, signals, pad, fft_window
    # time, freq, out_spect = rolling_spectrogram(time, signals,pad=pad,
    #                                             fft_window=fft_window)
    t_0 = time[0]
    time, freq, out_spect = rolling_spectrogram_improved(time, signals,pad=pad,
                                                fft_window=fft_window)
    time += t_0
    
    #return time, freq, out_spect
    # Average across channels
    out_spect=np.mean(out_spect,axis=0)
    '''
    if debug: print('Computed Spectrogram, Size ',out_spect.shape)    
        
    # Plot Spectrogram
    # Reduce plot resolution:
    # if plot_reduce > 1:
    #     time = grouped_average(time,plot_reduce)
    #     out_spect = grouped_average(out_spect,plot_reduce)
    #return out_spect,freq,time
    if plot_reduce != (1,1): 
        out_spect,time, freq = smooth_and_downsample_spectrogram(out_spect,time,\
                         freq,sigma=sigma_plot_reduce,factors=plot_reduce)
        
    plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,
                 save_Ext,'',f_lim,shotno,doSave_Extractor=doSave_Extractor,tScale=1,
                 doColorbar=doColorbar,clabel=clabel,cLim=cLim,cmap=cmap,
                 figsize=figsize,batch=batch)
    
    if debug: print('Finished Plot')
    
    return diag,signals,time,out_spect, freq

###############################################################################
def get_data(shotno,diag,sensor_name,sensor_set,data_archive,debug):
    # Pull C-Mod data from server for a given diagnostic
    if sensor_set == 'BP_T':
        if diag is None: diag = gC.__loadData(shotno,pullData=['bp_t'],
                                   data_archive=data_archive)['bp_t']
            
        # Extract specific sensor, if desired. Otherwise FFTs will be averaged
        signals = diag.gh_data if sensor_name == '' else \
            diag.ab_data[(np.array(diag.gh_names) ==  sensor_name)]
        time = diag.time
        
        clabel=r'$\partial_t{\mathrm{B}}_\theta$ [T/s]'
    elif sensor_set == 'BP_K':
        if diag is None: diag = gC.__loadData(shotno,pullData=['bp_k'],
                   data_archive=data_archive,params={'skipInteger':0})['bp_k']
        signals = diag.data 
        # Check which sensor exists if undefined
        if not sensor_name:
            sensors=[]
            for ind,name in enumerate(['bp6t_ghk','bp4t_ghk','bp2t_ghk']):
                if name in diag.names:sensors.append(name)
                if ind ==2:break                
            signals=signals[[np.argwhere(np.array(diag.names)==n).squeeze() for n in sensors]]#,'bp2t_ghk']]]
        else:
            signals=signals[[np.argwhere(np.array(diag.names)==n).squeeze() for n in [sensor_name]]]
        signals=signals.squeeze()
        if np.ndim(signals) == 1: signals = signals[np.newaxis,:]
        time = diag.time
        clabel = r'$\partial_t{\mathrm{B}}_\theta$ [T/s]'
    elif sensor_set == 'FRCECE':
        if diag is None: diag = gC.__loadData(shotno,pullData=['frcece'],
                                   data_archive=data_archive)['frcece']
        signals = diag.ECE*1e3 # convert from keV to eV
        # For the FRC-ECE, sensor_name is just a channel number
        signals = signals[sensor_name] if sensor_name else signals
        if np.ndim(signals) == 1: signals = signals[np.newaxis,:]
        time = diag.time
        clabel = r'$\tilde{\mathrm{T}}_e$ [eV]'
        
    elif sensor_set == 'GPC' or sensor_set == 'GPC_2':
        if diag is None:
            if sensor_set == 'GPC':diag = gC.__loadData(shotno,pullData=['gpc'],
                                   data_archive=data_archive)['gpc']
            else: diag = gC.__loadData(shotno,pullData=['gpc_2'],
                                   data_archive=data_archive)['gpc_2']
        signals = diag.Te*1e3 
        time = diag.time
        clabel = r'$\tilde{\mathrm{T}}_e$ [eV]'
        
    elif sensor_set == 'BP':
        if diag is None: diag = gC.__loadData(shotno,pullData=['bp'],
                                   data_archive=data_archive)['bp']
        signals = diag.BC['SIGNAL'] if sensor_name == '' else\
            diag.BC['SIGNAL'][diag.BC['NAMES']==sensor_name]
        time = diag.time
        clabel=r'$\tilde{\mathrm{B}}_\theta$ [G]'
        
    else: raise SyntaxError('Selected Diagnostic Not Yet Implemented')
    
    if debug: print('Loaded: %s'%sensor_set)
    
    return signals, time, clabel,diag

###############################################################################
def plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,filament,
                     save_Ext,f_lim,shotno=None,
                     doSave_Extractor=False,tScale=1e3,doColorbar=False,
                     clabel='',cLim=None,cmap='viridis',figsize=(6,6),
                     batch=False):
    
    name = 'Spectrogram_%s'%'-'.join(sensor_set) if type(sensor_set) is list else 'Spectrogram_%s'%sensor_set
    name='%s%s'%(name,save_Ext)
    fName = __gen_fName(params,sensor_set,save_Ext,filament,shotno)   
    plt.close(name)
    
    fig,ax=plt.subplots(1,1,tight_layout=True,num=name,figsize=figsize)
    
    # Get fLim
    f_inds = np.arange(*[np.argmin((freq*1e-3-f)**2) for f in f_lim]) if f_lim\
        else np.arange(*[0,len(freq)])
    
    vmin,vmax = cLim if cLim else [0,5*np.std(out_spect[f_inds])]
    if not cLim and vmax < 1: vmax = 1 # protect against noise dominated signals
    ax.pcolormesh(time*tScale,freq[f_inds]*1e-3,out_spect[f_inds],
                  shading='auto',rasterized=True,vmin=vmin,vmax=vmax,cmap=cmap)
    
    if doSave_Extractor and doSave: 
        # For mode extractor code, remove axes labels/etc
        plt.tick_params(left = False, bottom = False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(doSave+'Spectrogram_%s.png'%fName,
                    bbox_inches='tight',pad_inches=0)
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.tick_params(left = True, bottom = True)
    
    ax.set_xlabel('Time [%s]'%('s' if tScale==1 else 'ms'))
    ax.set_ylabel('Freq [kHz]')
    
    if doColorbar:
        norm = Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax,
                     label=clabel)
        
    if shotno:ax.text(.05,.95,'%d'%shotno,transform=ax.transAxes,fontsize=8,
            verticalalignment='top',bbox={'boxstyle':'round','alpha':.7,
                                          'facecolor':'white'})
    
    #if f_lim:ax.set_ylim(f_lim)
    
    if doSave:
        fig.savefig(doSave+'Spectrogram_%s.pdf'%fName,transparent=True)
        print('Saved: '+doSave+'Spectrogram_%s.pdf'%fName)
    
    
    if batch: 
        plt.close(fName) # terminal blocks untill plot is closed for some reason    
    else:plt.show()
###############################################################################
def __gen_fName(params,sensor_set,save_Ext,filament,shotno):
    if 'm' in params:
        if type(params['m']) is int:
            fName = 'floops_%s_m-n_%s-%s_f_%s%s'%\
                ( '-'.join(sensor_set),'-'.join(['%d'%param['m'] for param in params]),
                '-'.join(['%d'%param['n'] for param in params]),
                '-'.join(['%d'%(param['f']*1e-3) for param in params]),save_Ext) if \
                    type(params) is list else 'floops_%s_%s_m-n_%d-%d_f_%d%s.pdf'%\
                ('filament' if filament else 'surface', sensor_set,params['m'],
                params['n'],params['f']*1e-3,save_Ext)
        else:
            fName = 'floops_%s_m-n_%s_f_%s%s'%\
                ( '-'.join(sensor_set),'-'.join(['%d-%d'%(m_,n_) for m_,n_ in zip(params['m'],params['n'])]),
                '-'.join(['%d'%(param['f']*1e-3) for param in params]),save_Ext) if \
                    type(params) is list else 'floops_%s_%s_m-n_%s_f_%s%s.pdf'%\
                ('filament' if filament else 'surface', sensor_set,'-'.join(['%d-%d'%(m_,n_) for m_,n_ in zip(params['m'],params['n'])]),
                'custom',save_Ext)
    elif 'tLim' in params:
        fName = 'C_Mod_Data_%s_%d_t_%1.1f-%1.1f_f_%d-%d%s'%(sensor_set,shotno,\
               params['tLim'][0],params['tLim'][1],params['f_lim'][0],params['f_lim'][1],
               save_Ext)
    else:
        fName = 'C_Mod_Data_%s_%d%s'%(sensor_set,shotno,save_Ext)
    return fName

###############################################################################
def gen_lf_signals():
    file = 'cmod_logbook_ntm.csv'
    shotnos = np.loadtxt(file,skiprows=1,delimiter=',',usecols=0,dtype=int)

    shotnos = np.append(shotnos,[1051202011,1160930034])
    print(shotnos)
    # Split up in time chunks, frequency range chunks [ to make it easier to see lf, hf signals]
    tLims=[[.7,1.5]]
    # Break up into two frequency bins[.5,.8],
    # dataRanges = {'tLim':[[.8,1.1],[1.1,1.4]], 'signal_reduce':[15,1],'block_reduce':[[450,2500],[1000,250]],
    #               'f_lim':[[0,100],[100,600]]}
    
    dataRanges = {'tLim':[[.7,1.5]], 'signal_reduce':2,\
                  'block_reduce':[1000,1000],'sigma':(3,5),'plot_reduce':(4,1)}
    f_lim=[0,650]; c_lim=None
    pad = 14000;fft_window=500;HP_Freq=2e3
    for shot in shotnos:
        diag = None
        plt.close('all') # Clear plots    
        for ind_t,tLim in enumerate(dataRanges['tLim']):
            #for ind_f, f_lim in enumerate(dataRanges['f_lim']):
               
                diag,signals,time,out_spect, freq = \
                    signal_spectrogram_C_Mod(shot,sensor_set='BP_K',diag=diag,\
                         sensor_name='',tLim=tLim,HP_Freq=HP_Freq,f_lim=f_lim,\
                             block_reduce=dataRanges['block_reduce'],
                             signal_reduce=dataRanges['signal_reduce'],
                             pad=pad,fft_window=fft_window,cmap='gray',
                             figsize=(6,6),doSave='../output_plots/training_plots/'*True,\
                             params={'tLim':tLim,'f_lim':f_lim},save_Ext='_Training',
                             batch=True,sigma_plot_reduce=dataRanges['sigma'],\
                                 plot_reduce=dataRanges['plot_reduce'],cLim=c_lim)
                
###############################################################################
def smooth_and_downsample_spectrogram(spectrogram,time,freq, sigma=1.0, factors=(2, 2)):
    """
    Applies a Gaussian blur and then downsamples the spectrogram.

    Args:
        spectrogram (np.ndarray): The 2D spectrogram (frequencies x time).
        sigma (float or tuple): Standard deviation for Gaussian kernel.
                                If tuple, (sigma_freq, sigma_time).
        factors (tuple): Downsampling factors (downsample_freq, downsample_time).

    Returns:
        np.ndarray: The smoothed and downsampled spectrogram.
    """
    # Ensure spectrogram is 2D for image processing functions
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array (frequencies x time).")

    # Apply Gaussian filter
    # sigma can be a single value or a tuple for anisotropic smoothing
    smoothed_spectrogram = gaussian_filter(spectrogram, sigma=sigma)

    # Downsample using local mean pooling
    # This averages blocks of pixels, which is a good way to downsample
    # after blurring to prevent aliasing.
    downsampled_spectrogram = downscale_local_mean(smoothed_spectrogram, factors)
    time = downscale_local_mean(time, factors[1])
    freq = downscale_local_mean(freq, factors[0])
    return downsampled_spectrogram, time, freq

# Example Usage:
# Assuming 'your_spectrogram_2d' is your 2D spectrogram array
# smoothed_downsampled_spect = smooth_and_downsample_spectrogram(
#     your_spectrogram_2d,
#     sigma=(0.5, 2.0),  # Example: more smoothing along time axis
#     factors=(4, 4)     # Example: reduce size by 4x in both dimensions
# )


#import numpy as np
from scipy.signal.windows import hann, hamming, blackman
from scipy.signal import stft

def extract_and_window_blocks(signal, n_samples, m_skip, window_type='hanning'):
    """
    Extracts repeating blocks of n_samples from a signal, skipping m_skip samples
    in between, and applies a specified windowing function to each block.

    Args:
        signal (np.ndarray): The input 1D signal.
        n_samples (int): The number of samples to keep in each block.
        m_skip (int): The number of samples to skip between kept blocks.
        window_type (str): The type of window to apply ('hanning', 'hamming', 'blackman').

    Returns:
        list: A list of windowed signal blocks.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if m_skip < 0:
        raise ValueError("m_skip must be a non-negative integer.")

    window_func = None
    if window_type == 'hanning':
        window_func = hann
    elif window_type == 'hamming':
        window_func = hamming
    elif window_type == 'blackman':
        window_func = blackman
    else:
        raise ValueError("Unsupported window_type. Choose from 'hanning', 'hamming', 'blackman'.")

    window = window_func(n_samples)

    windowed_blocks = []
    current_index = 0

    while current_index + n_samples <= len(signal):
        block = signal[current_index : current_index + n_samples]
        windowed_block = block * window
        windowed_blocks.append(windowed_block)
        current_index += n_samples + m_skip

    return windowed_blocks

def compute_averaged_spectrogram_from_blocks(signals, sampling_rate, n_samples, m_skip, window_type='hanning', nfft=None):
    """
    Computes spectrograms for multiple signals by extracting windowed blocks and performing FFT on each.
    Returns the average of the individual spectrogram magnitudes.

    Args:
        signals (np.ndarray): A 1D or 2D array of signals. If 2D, each row is a signal.
        sampling_rate (float): The sampling rate of the original signals (samples per second).
        n_samples (int): The number of samples to keep in each block (FFT window size).
        m_skip (int): The number of samples to skip between kept blocks.
        window_type (str): The type of window to apply ('hanning', 'hamming', 'blackman').
        nfft (int, optional): The number of FFT points. If None, defaults to n_samples.
                               Should be >= n_samples. Padding with zeros if nfft > n_samples.

    Returns:
        tuple: A tuple containing:
            - frequencies (np.ndarray): Array of sample frequencies.
            - block_time_centers (np.ndarray): Array of time points corresponding to the center of each block.
            - averaged_spectrogram_magnitude (np.ndarray): The computed average spectrogram magnitude,
                                                           shape (num_frequencies, num_blocks).
    """
    if nfft is None:
        nfft = n_samples
    elif nfft < n_samples:
        raise ValueError("nfft must be greater than or equal to n_samples.")

    if signals.ndim == 1:
        # If a single 1D signal is passed, wrap it in a 2D array for consistent iteration
        signals_to_process = signals[np.newaxis, :]
    else:
        signals_to_process = signals

    all_individual_spectrograms = []
    frequencies = None
    block_time_centers = None

    for i, signal in enumerate(signals_to_process):
        # 1. Extract and window blocks for the current signal
        windowed_blocks = extract_and_window_blocks(signal, n_samples, m_skip, window_type)

        if not windowed_blocks:
            print(f"Warning: No blocks extracted for signal {i}. Skipping this signal.")
            continue

        # Calculate frequencies and time centers only once (they are the same for all signals)
        if frequencies is None:
            frequencies = np.fft.rfftfreq(nfft, d=1/sampling_rate)

        if block_time_centers is None:
            current_index = 0
            block_time_centers_list = []
            for _ in range(len(windowed_blocks)):
                time_center = (current_index + n_samples / 2.0) / sampling_rate
                block_time_centers_list.append(time_center)
                current_index += n_samples + m_skip
            block_time_centers = np.array(block_time_centers_list)

        # Initialize a list to store the FFT results for current signal's blocks
        current_signal_ffts = []

        # 2. Perform FFT on each windowed block for the current signal
        for block in windowed_blocks:
            padded_block = np.pad(block, (0, nfft - n_samples), 'constant')
            fft_result = np.abs(np.fft.rfft(padded_block))
            current_signal_ffts.append(fft_result)

        # 3. Stack the FFT results to form the spectrogram for the current signal
        current_spectrogram = np.column_stack(current_signal_ffts)
        all_individual_spectrograms.append(current_spectrogram)

    if not all_individual_spectrograms:
        return np.array([]), np.array([]), np.array([]) # Return empty if no spectrograms were computed

    # 4. Average the individual spectrograms
    # Stack all individual spectrograms along a new axis (axis=0)
    # Then take the mean along that new axis
    averaged_spectrogram_magnitude = np.mean(np.array(all_individual_spectrograms), axis=0)

    return frequencies, block_time_centers, averaged_spectrogram_magnitude


#############################################################################
# Batch launch
if __name__ == '__main__':

    gen_lf_signals()


