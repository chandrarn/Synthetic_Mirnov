#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:22:03 2025
    Header file for C-Mod data access/plotting
@author: rianc
"""


import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat 
from scipy.signal import lombscargle, hilbert
from scipy.optimize import minimize

import imageio.v3 as iio
import cv2

import pickle as pk
import sys
from os import getlogin
import json
from pathlib import Path
import mdsthin as mds # Needs to be separately installed through pip
try:import MDSplus
except:MDSplus=False# Doesn't exist on all systems, needed for one get_Cmod_data function
try:from mirnov_ted import Mirnov
except:Mirnov=False # same issue
from socket import getfqdn

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import rc,cm
from matplotlib.patches import Rectangle
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

from rolling_spectrogram import rolling_spectrogram

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
    CAL_PATH = ('/mnt' if getfqdn() == 'mfews-rianc' else '') + \
        '/home/sears/Matlab/Calibration/cal_v2/'
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