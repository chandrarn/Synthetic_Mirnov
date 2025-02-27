#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:08:15 2025

@author: rian
"""

from header import rolling_spectrogram, np, plt, json, histfile

from plot_sensor_output import __select_sensors, __gen_surface_data

def gen_single_spect(params,filament,save_Ext,phi_sensor=[180],
                     sensor_file='MAGX_Coordinates_CFS.json',sensor_set='MRNV',
                     file_geqdsk='geqdsk',doVoltage=False,doSave=''):
    
    time,_,signals = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                        sensor_set,file_geqdsk,doVoltage)
    
    # TEMP HARDCORE
    time = time[0]
    signals = signals[0]
    time, freq, out_spect = rolling_spectrogram(time, signals)
    out_spect=np.mean(out_spect,axis=0)
    
    plot_spectrogram(time,freq,out_spect,doSave,sensor_set)
    
    return time, freq, out_spect


def plot_spectrogram(time,freq,out_spect,doSave,sensor_set):
    
    plt.close('Spectrogram_%s'%sensor_set)
    fig,ax=plt.subplots(1,1,tight_layout=True,num='Spectrogram_%s'%sensor_set)
    ax.pcolormesh(time*1e3,freq*1e-3,out_spect)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Freq [kHz]')
    
    plt.show()
    
    
def get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                    sensor_set,file_geqdsk,doVoltage):
    # Load sensor parameters for voltage conversion
    sensor_params= json.load(open(sensor_file,'r'))
    # Load ThinCurr sensor output
    hist_file = histfile('data_output/floops_%s_%s_m-n_%d-%d_f_%d%s.hist'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,save_Ext))
    '''
    print('data_output/floops_%s_m-n_%d-%d_f_%d%s.hist'%\
                 (sensor_set,params['m'],params['n'],params['f']*1e-3,save_Ext))
    return hist_file
    '''    
    # Select usable sensors from set
    sensor_dict = __select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params)
    #return sensor_dict, sensor_params
    # build datasets
    X,Y,Z = __gen_surface_data(sensor_dict,hist_file,doVoltage,params,
                               sensor_set, sensor_params)
    return X,Y,Z
    