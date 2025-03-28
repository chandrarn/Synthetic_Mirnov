#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:22:03 2025
    Header file for C-Mod data access/plotting
@author: rianc
"""


import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import mdsthin as mds # Needs to be separately installed through pip
from matplotlib.colors import Normalize
from matplotlib import rc,cm
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

from rolling_spectrogram import rolling_spectrogram


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
    
    
    # Standardize data to 2D signal
    if np.ndim(data)==1: data = np.array(data)[np.newaxis,:]
    # Detect compound shot
    if np.any(np.diff(time)<0):t_break_ind=np.argwhere(np.diff(time)<0)[:,0]+1
    else:t_break_ind=np.array([])
    t_break_ind=np.concatenate(([0],t_break_ind,[len(time)])).astype(int)
    if HP_Freq or LP_Freq:
        for i,dat in enumerate(data):
            for t in np.arange(len(t_break_ind)-1):
                if HP_Freq:# Isolate Mode
                    data[i,t_break_ind[t]:t_break_ind[t+1]] = \
                        gaussianHighPassFilter(\
                            dat[t_break_ind[t]:t_break_ind[t+1]],\
                            time[t_break_ind[t]:t_break_ind[t+1]],1./HP_Freq)[0]
                if LP_Freq:# Remove Noise
                    data[i,t_break_ind[t]:t_break_ind[t+1]] = \
                        gaussianLowPassFilter(\
                            dat[t_break_ind[t]:t_break_ind[t+1]],\
                            time[t_break_ind[t]:t_break_ind[t+1]],1./LP_Freq)
                                
            
    return data

