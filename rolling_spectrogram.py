#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 17:58:51 2025
     Rolling spectrogram
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
           Sxx[i] = np.abs(np.fft.fft(x_pad2)*Normalization)**2
   
       frq = frq[range(N//2)]   #Take only the positive frequencies
       Sxx = Sxx[:,range(N//2)]
       
       out_spect.append(np.transpose(Sxx))

    out_spect = np.array(out_spect).squeeze()
    if out_spect.ndim == 2: out_spect=out_spect[np.newaxis,:,:] # Special check: 
    return time, frq, out_spect
###########################################################################

