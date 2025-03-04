#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:58:42 2025
    Sandbox code for Mirnov noise profile detection
    
@author: rianc
"""

from header import np, plt

import get_Cmod_Data as gC

def rawSignal(shotno,tLim,bpt=None):
    if bpt is None: bpt = gC.BP_T(shotno)
    
    time = bpt.time
    t_inds = np.arange((tLim[0]-time[0])/(time[1]-time[0]),\
              (tLim[1]-time[0])/(time[1]-time[0]),dtype=int)
    time = time[t_inds]
    sig = bpt.ab_data[-1,t_inds] # last signal has largest amplitude
    
    sig -= np.mean(sig)
    # FFT over time range
    
    fft = np.fft.fft(sig)
    freq = np.fft.fftfreq(len(sig),time[1]-time[0])
    
    plt.close('Mirnov_FFT')
    fig,ax=plt.subplots(1,2,num='Mirnov_FFT',tight_layout=True,figsize=(6,3))
    ax[0].plot(time,sig,label=shotno)
    ax[0].legend(fontsize=8,handlelength=1)
    
    ax[1].plot(freq[:len(freq)//2]*1e-3,np.abs(fft)[:len(fft)//2])
    ax[1].set_xlabel('Freq [kHz]')
    ax[1].set_ylabel('PSD [arb]')
    
    for i in range(2):ax[i].grid()
    
    plt.show()
    
    return bpt