#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:22:03 2025
    Header file for C-Mod data access/plotting
@author: rianc
"""


import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.io import loadmat 
from scipy.signal import lombscargle, hilbert, ShortTimeFFT, csd
from scipy.optimize import minimize

from skimage.transform import downscale_local_mean

import imageio.v3 as iio
import cv2

import pickle as pk
import sys
from os import getlogin
import os
import xarray as xr
import json
from pathlib import Path
import mdsthin as mds # Needs to be separately installed through pip
try:import MDSplus
except:MDSplus=False# Doesn't exist on all systems, needed for one get_Cmod_data function
try:from mirnov_ted import Mirnov
except:Mirnov=False # same issue
#from socket import getfqdnheader_Cmod

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rc,cm
from matplotlib.patches import Rectangle
import matplotlib
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
try:
    matplotlib.use('TkAgg')
    plt.ion()
except:pass
from rolling_spectrogram import rolling_spectrogram, rolling_spectrogram_improved

# TODO: verift atht this works for other users
try:
    data_archive_path = '/home/rianc/Documents/data_archive/' if \
        getlogin() == 'rianc' else '/mnt/home/rianc/Documents/data_archive/' 
    if getlogin() == 'rian': data_archive_path = '/home/rian/Documents/data_archive/'
except: data_archive_path = ''

###############################################################################
# Helper functions
# Helper function for filtering     
def gaussianLowPassFilter(y,t,timeWidth):

    dt=t[1]-t[0]
    #1/(dt*timeWidth*2*_np.pi)#
    sigma= (1./(2*np.pi))*timeWidth/dt #2.355*timeWidth/dt#  #TODO(John)  This equation is wrong.  Should be dividing by 2.355, not multiplying.  Fix here and with all dependencies
    yFiltered=gaussian_filter1d(y,sigma)
    return yFiltered
   
def gaussianHighPassFilter(y,t,timeWidth): return y-gaussianLowPassFilter(y, t, timeWidth)    
   
def __doFilter(data,time,HP_Freq, LP_Freq):
    # General Finite Impulse Response filter
    # Can handle highpass, lowpass = False
    
    # Standardize data to 2D signal
    if np.ndim(data)==1: data = np.array(data)[np.newaxis,:]

    if HP_Freq or LP_Freq:
        for i,dat in enumerate(data):
            if HP_Freq:# Isolate Mode
                data[i] =   gaussianHighPassFilter(data[i], time,1./HP_Freq)
            if LP_Freq:# Remove Noise
                data[i] =  gaussianLowPassFilter(  data[i],time,1./LP_Freq)
                                
            
    return data

###############################################################################
###############################################################################
def remove_freq_band(signal,time,freq_rm=500e3, freq_sigma=100):
    #print('Runnning Frequency Removal')
    rfft = np.fft.rfft(signal,axis=1)
    fftFreq = np.fft.rfftfreq(signal.shape[1], time[1]-time[0])
    
    rm_inds = np.arange(*[np.argmin(np.abs(fftFreq-f)) \
                          for f in [freq_rm-freq_sigma, freq_rm+freq_sigma]])
    rfft[:,rm_inds] = 0
    
    return np.fft.irfft(rfft,axis=1)
###############################################################################
def correct_Bode(signal,time,sensor_name):
    '''
    Bode correction
    Based on calibrate_xpose.m by Jason Sears
    '''
    
    rfft = np.fft.rfft(signal)
    fftFreq = np.fft.rfftfreq(len(signal), time[1]-time[0])
    
    # check if signal is outside of calibration range [otherwise returns garbage]
    if fftFreq[np.argmax(np.abs(rfft))] < 200e3: return signal,None,None,None,None
    
    #f,H_spline,fine_cal,fine_caln = __cal_Correction(sensor_name,fftFreq)
    #return  f,H_spline,fine_cal,fine_caln
    f, H = __cal_Correction(sensor_name,fftFreq)

    sig_new = np.fft.irfft( rfft / H ) 
    
    if len(sig_new) < len(signal): # sometimes
        sig_new = np.hstack((sig_new,[sig_new[-1]]*(len(signal)-len(sig_new)) ))
    
    return sig_new,  H, f, fftFreq,rfft
  
def __cal_Correction(sensor_name,freq):
    CAL_PATH = ('/mnt' if getlogin() == 'mfews-rianc' else '') + \
        '/mnt/home/sears/Matlab/Calibration/cal_v2/'
    try:mat = loadmat(CAL_PATH+'451_responses/'+sensor_name +'_cal.mat')
    except: mat = loadmat(CAL_PATH+'451_responses/'+'BP1T_GHK' +'_cal.mat')
    f = mat['f'][0]; H_spline = mat['H_spline'][0]
    try:
        mat = loadmat(CAL_PATH+'fine_tuning/'+sensor_name +'_cal+.mat')
        fine_cal= mat['fine_cal'][0,0]
        mat = loadmat(CAL_PATH+'fine_tuning/'+sensor_name +'_caln.mat')
        fine_caln = mat['fine_caln'][0,0]
    except: fine_cal = fine_caln = 1
    #return f,H_spline,fine_cal,fine_caln 
    
    
    return  f, np.interp(freq,f,H_spline) * fine_cal * fine_caln
################################################################################
def grouped_average(arr, n, axis=-1):
    """
    Averages a NumPy array along a specific axis in groups of n sequential samples.

    Args:
        arr (np.ndarray): The input NumPy array.
        n (int): The number of sequential samples to group together for averaging.
        axis (int): The axis along which to perform the grouping and averaging.
                    Defaults to the last axis (-1).

    Returns:
        np.ndarray: The averaged array.
    """
    if n <= 0:
        raise ValueError("Group size 'n' must be a positive integer.")

    # Move the target axis to the last position for easier reshaping
    arr_transposed = np.moveaxis(arr, axis, -1)

    original_shape = arr_transposed.shape
    target_axis_length = original_shape[-1]

    if target_axis_length % n != 0:
        # Handle cases where the length of the axis is not perfectly divisible by n
        # You have a few options here:
        # 1. Pad the array: Add zeros or a repeating pattern to make it divisible.
        # 2. Truncate the array: Discard the last few samples that don't form a full group.
        # 3. Raise an error (current implementation): Let the user know.
        
        # For this function, we'll truncate for simplicity and efficiency.
        # This keeps the output clean without needing to handle partial groups.
        print(f"Warning: Axis length ({target_axis_length}) is not divisible by group size ({n}). "
              "Truncating the last few samples.")
        arr_transposed = arr_transposed[..., :target_axis_length - (target_axis_length % n)]
        target_axis_length = arr_transposed.shape[-1]


    # Calculate the new shape for reshaping
    # The new last dimension will be (original_length / n, n)
    new_shape = original_shape[:-1] + (target_axis_length // n, n)

    # Reshape the array to create the groups
    arr_reshaped = arr_transposed.reshape(new_shape)

    # Calculate the mean along the newly created last dimension (which is 'n')
    averaged_arr = np.mean(arr_reshaped, axis=-1)

    # Move the axis back to its original position if it was moved
    if axis != -1:
        averaged_arr = np.moveaxis(averaged_arr, -1, axis)

    return averaged_arr
