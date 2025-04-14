#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:22:03 2025
    Header file for C-Mod data access/plotting
@author: rianc
"""


import numpy as np
from scipy.ndimage import gaussian_filter1d
import pickle as pk
import sys
from os import getlogin
import json
from pathlib import Path
import mdsthin as mds # Needs to be separately installed through pip
import MDSplus
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

data_archive_path = '/home/rianc/Documents/data_archive/' if \
    getlogin() == 'rianc' else '/mnt/home/rianc/Documents/data_archive/' 


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
