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
import matplotlib.pyplot as plt
from eqtools import CModEFITTree
from C_mod_header import __doFilter

# Assume that we have some way of grouping in frequency/time

def run_n(shotno=1051202011, tLim=[1,1.01], fLim, HP_Freq, LP_Freq):
    
    sensors = Mirnov(shotno,t1=tLim[0],t2=tLim[1])
    
    phi = sensors.getPhi(sensors.coil_names)
    
    R = sensors.getR(sensors.coil_names)
    Z = sensors.getZ(sensors.coil_names)
    
    theta = get_theta(R,Z,shotno,tLim)
    
    # Get data
    data = sensors.getSig(rawSig=False) #TODO: unclear how frequency calibration is or is not being done
    
    # Do Filtering 
    __doFilter(data,sensors.getT,HP_Freq, LP_Freq)
    
    # FFT processing: more than just Hilbert
    # Use dominant frequency within bandpassfilteresed region for now
    phase = np.unwrap(np.angle(hilbert(data,axis=1)))
    
    
    # normalize off phase by rings in poloidal plane?
    z_levels = [8,10,14,30] #+/-, in cm
    # SWitch this to theta
    
    # zero base to one sensor
    
    # run fit for n
    
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