#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:08:15 2025

@author: rian
"""

from header import rolling_spectrogram, np, plt, json, histfile

from plot_sensor_output import __select_sensors, __gen_surface_data

def multiple_spect(params=[{'m':12,'n':10,'f':50e3,'dt':1e-6},
                           {'m':2,'n':1,'f':7e3,'dt':1e-6}],
                   filament=[True,False],save_Ext='',phi_sensor=[[180],[0]],
                     sensor_file='MAGX_Coordinates_CFS.json',sensor_set=['MRNV','BP'],
                     file_geqdsk='geqdsk',doVoltage=True,doSave='',f_lim=[0,180]):
    # Assumes that the time, frequency base for the simulation runs is identical
    spects=[]
    all_spects=[]
    for ind,param in enumerate(params):
        print('Evaluating Sensor %s'%sensor_set[ind])
        time,freq,out_spect = gen_single_spect(param,filament[ind],save_Ext,phi_sensor[ind],
               sensor_file,sensor_set[ind],file_geqdsk,doVoltage,doSave=False,
               doPlot=True)
        all_spects.append(out_spect)
        if ind == 0: spects=out_spect/np.max(out_spect)
        else:spects +=out_spect/np.max(out_spect)
    
    plot_spectrogram(time,freq,spects,doSave,sensor_set,params,
                               filament,save_Ext,f_lim)
    return spects,time,freq,out_spect, all_spects

def gen_single_spect(params,filament,save_Ext,phi_sensor=[180],
                     sensor_file='MAGX_Coordinates_CFS.json',sensor_set='MRNV',
                     file_geqdsk='geqdsk',doVoltage=False,doSave='',f_lim=[0,175],
                     doPlot=True,plot_ext=''):
    
    time,_,signals = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                        sensor_set,file_geqdsk,doVoltage)
    
    # TEMP HARDCORE
    time = time[0]
    signals = signals[0]
    time, freq, out_spect = rolling_spectrogram(time, signals,pad=350)
    out_spect=np.mean(out_spect,axis=0)
    
    if doPlot:plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,
                               filament,plot_ext,f_lim)
    
    return time, freq, out_spect


def plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,filament,save_Ext,f_lim):
    
    name = 'Spectrogram_%s'%'-'.join(sensor_set) if type(sensor_set) is list else 'Spectrogram_%s'%sensor_set
    name='%s%s'%(name,save_Ext)
    plt.close(name)
    fig,ax=plt.subplots(1,1,tight_layout=True,num=name)
    ax.pcolormesh(time*1e3,freq*1e-3,out_spect,shading='auto',rasterized=True)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Freq [kHz]')
    if f_lim:ax.set_ylim(f_lim)
    
    plt.show()
    fName = 'floops_%s_m-n_%s-%s_f_%s%s.pdf'%\
             ( '-'.join(sensor_set),'-'.join(['%d'%param['m'] for param in params]),
              '-'.join(['%d'%param['n'] for param in params]),
              '-'.join(['%d'%(param['f']*1e-3) for param in params]),save_Ext) if \
                 type(params) is list else 'floops_%s_%s_m-n_%d-%d_f_%d%s.pdf'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,save_Ext)
    if doSave: fig.savefig(doSave+'Spectrogram_%s'%fName,transparent=True)
    
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
    