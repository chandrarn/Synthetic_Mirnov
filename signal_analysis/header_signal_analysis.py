#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:04:59 2024
    header file for synthetic mirnov data
@author: rian
"""

import struct
import sys
import importlib.util
import os
try:import h5py
except: pass
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc,cm
from matplotlib.colors import Normalize
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

# import pyvista
# MDS load may not work on all machines
try:import MDSplus as mds
except:import mdsthin as mds
#pyvista.set_jupyter_backend('static') # Comment to enable interactive PyVista plots



sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release/python/')
sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Intel_Compiled/python/')
# from OpenFUSIONToolkit.ThinCurr import ThinCurr
# from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors,flux_loop
# from OpenFUSIONToolkit.util import build_XDMF, mu0
from OpenFUSIONToolkit.io import histfile
# from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh, build_torus_bnorm_grid, build_periodic_mesh, write_periodic_mesh

    
from freeqdsk import geqdsk
sys.path.append('/orcd/home/002/rianc/Documents/eqtools-1.0/') # Necessary for Engaging
from eqtools import CModEFITTree
import cv2
from scipy.interpolate import make_smoothing_spline
from scipy.special import factorial
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
import json
from socket import gethostname
server = (gethostname()[:4] == 'orcd') or (gethostname()[:4]=='node')

# from Time_dep_Freq import I_KM, F_KM, I_AE, F_AE, F_AE_plot,F_KM_plot



#####################3
# Add paths
#ssys.path.append('signal_analysis/')
sys.path.append('../C-Mod/')
from rolling_spectrogram import rolling_spectrogram
##############################################
def doFFT(time,signal,doPlot=False):
    # Do and return fft
    # assumes signal is already zero-mean subtracted
    if type(signal) == list: signal = np.array(signal)
    if signal.ndim == 1: signal = np.expand_dims(signal,0)
    
    fs = (time[1]-time[0]) # sampling frequency
    fft_freq = np.fft.fftfreq(len(signal[0]),fs)[:len(time)//2]
    
    fft_out = []
    for sig in signal: 
        fft_out.append( np.fft.fft(sig)[:len(sig)//2]/(fs * len(sig)) )
    fft_out = np.array(fft_out)
    
    if doPlot:
        plt.close('FFT')
        fig,ax = plt.subplots(1,1,num='FFT',tight_layout=True,figsize=(4,3))
        ax.plot(fft_freq*1e-3,np.abs(fft_out).T,alpha=.8)
        ax.set_xlabel(r'Frequency [kHz]')
        ax.set_ylabel(r'PSD')
        ax.grid()
        plt.show()

    return fft_freq, fft_out