#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:58:51 2025
     Rolling spectrogram
     Adapted from code by Ian Stewart
@author: rian
"""
import numpy as np


#Create a spectrogram using a rolling fft^2 over the x data 
def rolling_spectrogram(time, x, fft_window=100, pad=0):

    #Perform a rolling FFT
    #===================================
    N = fft_window+pad            #Number of Samples
    T = time[1]-time[0]          #Time interval between samples

    frq = np.fft.fftfreq(N, d=T)

    out_spect = []
    for signal in x if np.ndim(x)>1 else [x]:
       #Initialize the spectrogram array Sxx
       Sxx = np.zeros((len(time),len(np.fft.fft(signal[0:fft_window+pad]))))#,dtype=np.dtype(np.complex128))
       
       #Pad the ends of the FULL x array
       x_pad = np.zeros(np.shape(signal)[0]+fft_window)
       x_pad[int(fft_window/2.0):len(signal)+int(fft_window/2.0)] = signal
   
       Normalization = 2.0/len(frq)
   
       for i in range(len(x_pad)-fft_window):
           #Pad each of the windowed arrays
           x_pad2 = np.zeros(len(x_pad[i:i+fft_window])+pad)
           x_pad2[pad//2:len(x_pad[i:i+fft_window])+pad//2] = x_pad[i:i+fft_window]*np.hanning(fft_window)
           Sxx[i] = np.abs(np.fft.fft(x_pad2)*Normalization)**1
   
       frq = frq[range(N//2)]   #Take only the positive frequencies
       Sxx = Sxx[:,range(N//2)]
       
       out_spect.append(np.transpose(Sxx))

    out_spect = np.array(out_spect).squeeze()
    if out_spect.ndim == 2: out_spect=out_spect[np.newaxis,:,:] # Special check: 
    return time, frq, out_spect
###########################################################################

# Memory improved implimentation
from scipy.signal import stft, windows

def rolling_spectrogram_improved(time, x, fft_window=100, pad=0):
    """
    Computes a spectrogram using a rolling FFT on one or more 1D signals.

    Args:
        time (np.ndarray): 1D array of time values.
        x (np.ndarray): 1D or 2D array of signals. If 2D, each row is a signal.
        fft_window (int): Length of the FFT window in samples.
        pad (int): Number of zeros to pad to the FFT window.

    Returns:
        tuple: A tuple containing:
            - time (np.ndarray): Original time array.
            - frq (np.ndarray): Frequencies corresponding to the spectrogram.
            - out_spect (np.ndarray): The computed spectrogram(s).
                                       Shape is (num_signals, num_frequencies, num_time_steps).
    """
    if np.ndim(x) == 1:
        x_reshaped = x[np.newaxis, :]  # Ensure x is 2D for consistent processing
    else:
        x_reshaped = x

    T = time[1] - time[0]  # Time interval between samples
    nperseg = fft_window  # Samples per segment
    noverlap = fft_window - 1  # Maximum overlap for high time resolution
    nfft = fft_window + pad  # FFT length including padding

    all_spectrograms = []

    for signal in x_reshaped:
        # Using scipy.signal.stft for efficiency and correctness
        # It handles windowing, overlapping, padding, and FFT internally
        # fs is the sampling frequency, which is 1/T
        # window is the type of window to apply
        # nperseg is the length of each segment
        # noverlap is the number of points to overlap between segments
        # nfft is the length of the FFT
        # return_onesided=True is for real signals, returning only positive frequencies
        frq, t_spec, Zxx = stft(
            signal,
            fs=1.0/T,
            window=windows.hann(nperseg),
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            return_onesided=True,
            detrend=False  # No detrending needed for amplitude spectrogram
        )
        
        # Sxx is the magnitude squared of the STFT output
        # Normalization factor for power spectral density, consider (2.0/N)**2 if it's for amplitude
        # However, for just magnitude, np.abs(Zxx) is sufficient.
        # The original code used **1, implying magnitude, so we'll stick to that interpretation.
        Sxx = np.abs(Zxx)

        # The time points returned by stft (t_spec) correspond to the center of each window.
        # We need to map these to the original time array's length.
        # For simplicity and memory efficiency, we'll return the original `time` array
        # and the spectrogram's time axis will be implicitly defined by `t_spec`.
        # If strict alignment to original 'time' array's length is needed, interpolation
        # would be required, but that adds memory and computational cost.
        # For a rolling window, the number of output time points for STFT is typically
        # different from the original signal length due to windowing and overlap.
        # Let's adjust the output to reflect the STFT's time resolution.
        
        # If you truly need the spectrogram to have the same number of time points
        # as the original `time` array, you'd need to interpolate.
        # However, this isn't standard for STFT and would be very memory intensive
        # for large signals. STFT naturally reduces the time resolution.
        
        # Given the original code's `Sxx = np.zeros((len(time),...))`,
        # it seems like an interpolation or different windowing strategy was intended.
        # The `stft` function provides `t_spec` which are the actual time points
        # corresponding to the STFT output. We should use these.
        # The original `len(time)` for the first dimension of `Sxx` is misleading
        # if a standard STFT approach is used.
        
        # For memory efficiency and correct STFT interpretation, `Zxx`'s time dimension
        # will be `len(t_spec)`. We'll transpose it to (frequencies, time_steps).
        all_spectrograms.append(Sxx)
    
    # Stack all spectrograms. If only one signal, squeeze to 2D.
    out_spect = np.array(all_spectrograms)
    
    # Transpose to get (num_signals, num_frequencies, num_time_steps)
    # The `stft` output is (frequencies, time_steps) for a single signal.
    # So if `all_spectrograms` has shape (num_signals, frequencies, time_steps),
    # this is already the desired output shape after stacking.

    # Re-evaluating the original output shape: `out_spect.append(np.transpose(Sxx))`
    # and then `out_spect = np.array(out_spect).squeeze()`.
    # This suggests the final output should be (frequencies, time_steps) if one signal,
    # or (num_signals, frequencies, time_steps) if multiple.
    
    # `stft` returns (frequencies, time_steps).
    # If `x` is 1D, `all_spectrograms` will have one element.
    # If `x` is 2D (multiple signals), `all_spectrograms` will be a list of 2D arrays.
    
    # Let's ensure the final output shape is (num_signals, frequencies, time_steps).
    if out_spect.ndim == 2: # This means there was only one signal
        out_spect = out_spect[np.newaxis, :, :] # Make it (1, frequencies, time_steps)

    # Return the original time array, the frequencies from stft, and the spectrograms.
    # Note: `t_spec` are the actual time points for the spectrogram, not `time`.
    # It is crucial to understand that the number of time points in `t_spec` will
    # generally be less than `len(time)`.
    return t_spec, frq, out_spect