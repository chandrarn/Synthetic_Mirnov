#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:58:42 2025
    Sandbox code for Mirnov noise profile detection
    
@author: rianc
"""

from header_signal_analysis import np, plt, gaussian_filter1d

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
    
    plt.close('Mirnov_FFT_%d'%shotno)
    fig,ax=plt.subplots(1,3,num='Mirnov_FFT%d'%shotno,tight_layout=True,figsize=(6,3))
    ax[0].plot(time,sig,label=shotno)
    ax[0].legend(fontsize=8,handlelength=1)
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Signal [G?]')
    
    ax[1].plot(freq[:len(freq)//2]*1e-3,np.abs(fft)[:len(fft)//2])
    ax[1].set_xlabel('Freq [kHz]')
    ax[1].set_ylabel('PSD [arb]')
    
    for i in range(2):ax[i].grid()
    
    
    #######
    # do filtering
    fft_abs = fft#fft[:len(fft)//2]
    fft_f=freq#freq[:len(freq)//2]*1e-3
    fft_abs=fft_abs[::10]
    fft_f=fft_f[::10]
    f_width=1
    df=fft_f[1]-fft_f[0]
    sigma= (1./(2*np.pi))*f_width/df
    yFiltered=gaussian_filter1d(fft_abs,sigma)
    ax[1].plot(fft_f[:len(fft_f)//2]*1e-3,
               np.abs(yFiltered)[:len(fft_abs)//2],alpha=.6)
    
    ########### histogram
    counts,bins=np.histogram(sig,bins=50)
    ax[2].stairs(counts,bins)
    ax[2].grid()
    ax[2].set_xlabel(r'$\Delta$-Signal [G?]')
    ax[2].set_ylabel(r'Counts [\#]')
    sigma=2.5;A=25000*6.5
    fn_norm = lambda x: A*np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    x_=np.linspace(-15,15,100)
    ax[2].plot(x_,fn_norm(x_),alpha=.6)
    
    plt.show()
    
    return bpt, yFiltered,sig