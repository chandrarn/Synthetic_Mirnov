#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:27:44 2025
 basic fft localizer, r, n
@author: rianc
"""

from header_Cmod import np, plt, __doFilter
from get_Cmod_Data import __loadData

###################################################
def ECE_R(shotno=1160930034,doSave='',tLim=[1,1.25],fLim=None,
          signal_reduce=1,block_reduce=[],HP_Freq=100,rawData=None):
    
    if rawData is None:
        rawData = __loadData(shotno,pullData=['frcece','bp_t'])
        print('Loaded signals')
    time = rawData['frcece'].time

    # Time selection and redunction
    t_inds = np.arange(*[np.argmin((time-t)**2) for t in tLim],dtype=int)[::signal_reduce]
    time = time[t_inds]
    signals = rawData['frcece'].ECE[:,t_inds]
    
    if block_reduce:
        del_inds=np.array([],dtype=int)
        for ind in np.arange(0,len(t_inds)-block_reduce[0],np.sum(block_reduce)):
            del_inds = np.append(del_inds,np.arange(ind,ind+block_reduce[0],dtype=int))
        #return t_inds,del_inds
        t_inds = np.delete(t_inds,del_inds)
        
    # Run filtering 
    print('Filtering signal')
    signals = __doFilter(signals, time, HP_Freq, None)
    print('Filtered')
    
    # FFT Stuff
    frequencies = np.fft.fftfreq(len(time), (time[1]-time[0]))*1e-3
    
    FFT = []
    for ind,s in enumerate(signals):
        print(ind)
        
        FFT.append(np.fft.fft(s))
    FFT= np.array(FFT)    
    
    # Plotting
    t_ind_r = np.argmin((rawData['frcece'].time_R-tLim[0])**2)
    R = rawData['frcece'].R[:,t_ind_r]
    
    return FFT, R, frequencies
    plt.close('FFT_R_%d'%shotno)
    fig,ax=plt.subplots(1,1,num='FFT_R_%d'%shotno,tight_layout=True)
    freq=frequencies[:len(frequencies)//2]
    contours = np.abs(FFT)[:,:len(freq)]
    
    ax.contourf(R,freq,contours.T)
    
    plt.show()
    
#####################
if __name__ == '__main__':ECE_R()