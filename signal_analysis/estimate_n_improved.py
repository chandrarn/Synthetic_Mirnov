#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 22:57:33 2025
 Rewrite of phase analysis code for n#
 Use Ted's data pulling functinoality
 May run into issues for large data frames
 
 - select rings of sensors
 - 
 
 - Eventually: Interface with fft spectrogram selection code:
     Given fourier signal, return n# within frequencyt peak
  Eventually: needs to be able to pull in from senosrs ( need to add all HF sensors)
@author: rian
"""

from mirnov_ted import Mirnov
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from eqtools import CModEFITTree
from sys import path; path.append('../C-Mod/')
from C_mod_header import __doFilter
from get_Cmod_data import __loadData


# Assume that we have some way of grouping in frequency/time
def run_n(shotno=1051202011, tLim=[1,1.01], fLim=None, HP_Freq=1e3, LP_Freq=None):
    
    bp_k = __loadData(shotno,pullData='bp_k')['bp_k']
    
    phi = bp_k.Phi
    
    R = bp_k.Phi
    Z = bp_k.Phi
    
    theta = get_theta(R,Z,shotno,tLim)
    
    # Get data
    data = bp_k.data #TODO: unclear how frequency calibration is or is not being done
    
    # get Time range
    inds = np.arange(*[np.argmin(np.abs(bp_k.time-t)) for t in tLim])
    # Do Filtering 
    __doFilter(data[:,inds], bp_k[inds],HP_Freq, LP_Freq)
    
    
    # FFT processing: more than just Hilbert
    # Use dominant frequency within bandpassfilteresed region for now
    phase = np.unwrap(np.angle(hilbert(data,axis=1)))
    
    
    angles = bp_k.Phi
    angles_group = np.array([])
    phase_group=np.array([])
    # normalize off phase by rings in poloidal plane?
    z_levels = [8,10,14,30] #+/-, in cm
    # SWitch this to theta
    for z in z_levels:
        z_inds = np.argwhere(z-.5<=bp_k.Z*1e2<=z+0.5)
        angles_group = np.append(angles_group, angles[z_inds]-angles[z_inds][0])
        phase_group = np.append(phase_group[z_inds]-phase[z_inds][0])
    # zero base to one sensor
    
    # run fit for n
    fn = lambda n: np.mean(np.abs(phase_group-(n[0]*(angles_group[1:]-angles_group[1]))%360 )**2)
    res = minimize(fn,[n[-1]],bounds=[[-20,0]])
    n_opt = res.x
    
    # Run for m?
    
    
    
    
    
def get_theta(R,Z,shotno,tLim):
        # Assume using the upper row always
        try:eq = CModEFITTree(shotno)
        except:eq = CModEFITTree(1160930034)
        time = eq.getTimeBase()
        tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] ) 
        # if tLim points are closer than dt-EFIT, it won't work
        if np.size(tInd)==0:tInd = np.argmin((time-tLim[0])**2) 
        zmagx = np.mean(eq.getMagZ()[tInd])
        rmagx = np.mean(eq.getMagR()[tInd])
        
        
        #sensor = __loadData(shotno,pullData=['bp_t'])['bp_t']
        theta = np.arctan2(Z-zmagx,R-rmagx)[0]*180/np.pi
        
        return theta