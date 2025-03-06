#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:08:15 2025

@author: rian
"""

from header_signal_analysis import rolling_spectrogram, np, plt, json, histfile, Normalize, cm

from get_signal_data import get_signal_data

import get_Cmod_Data as gC

from spectrogram_Cmod import plot_spectrogram

from add_noise import __noise_component

def multiple_spect(params=[{'m':12,'n':10,'f':50e3,'dt':2e-7},
                           {'m':2,'n':1,'f':7e3,'dt':2e-6}],
                   filament=[True,False],save_Ext='',phi_sensor=[[180],[0]],
                     sensor_file='MAGX_Coordinates_CFS.json',sensor_set=['MRNV','BP'],
                     file_geqdsk='geqdsk',doVoltage=True,doSave='',f_lim=[0,180]):
    # Assumes that the time, frequency base for the simulation runs is identical
    spects=[]
    all_spects=[]
    for ind,param in enumerate(params):
        print('Evaluating Sensor %s'%sensor_set[ind])
        time,freq,out_spect,signals = gen_single_spect(param,filament[ind],
               save_Ext,phi_sensor[ind],
               sensor_file,sensor_set[ind],file_geqdsk,doVoltage,doSave=False,
               doPlot=True)
        all_spects.append(out_spect)
        if ind == 0: spects=out_spect/np.max(out_spect)
        else:spects +=out_spect/np.max(out_spect)
    
    plot_spectrogram(time,freq,spects,doSave,sensor_set,params,
                               filament,save_Ext,f_lim,)
    return spects,time,freq,out_spect, all_spects

def gen_single_spect(params,filament,save_Ext,phi_sensor=[180],
                     sensor_file='MAGX_Coordinates_CFS.json',sensor_set='MRNV',
                     file_geqdsk='geqdsk',doVoltage=False,doSave='',f_lim=[0,175],
                     doPlot=True,plot_ext='',doNoise=True,pad=1900,fft_window=1500):
    
    time,_,signals,_ = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                        sensor_set,file_geqdsk,doVoltage)
    
    # TEMP HARDCODE
    time = time[0]
    signals = signals[0]
    signals /=np.max(signals)
    if doNoise:
        for ind, s in enumerate(signals):signals[ind] += __noise_component(s)
    time, freq, out_spect = rolling_spectrogram(time, signals,pad=pad,
                                                fft_window=fft_window)
    out_spect=np.mean(out_spect,axis=0)
    
    if doPlot:plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,
                               filament,plot_ext,f_lim,)
    
    return time, freq, out_spect,signals



        
    

###############################################################################

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
        
    











































