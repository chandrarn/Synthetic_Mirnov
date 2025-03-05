#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:08:15 2025

@author: rian
"""

from header import rolling_spectrogram, np, plt, json, histfile, Normalize, cm

from plot_sensor_output import __select_sensors, __gen_surface_data

import get_Cmod_Data as gC

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
    
    time,_,signals = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
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

def __noise_component(signals):
    sigma=2.5;A=25000*6.5
    fn_norm = lambda x: A*np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    x_=np.linspace(-1,1,100)
    return 5*(np.max(signals)/15)*np.random.choice(x_,len(signals),True,fn_norm(x_)/np.sum(fn_norm(x_)))
def plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,filament,
                     save_Ext,f_lim,shotno=None,
                     doSave_Extractor=False,tScale=1e3,doColorbar=False,clabel='',):
    
    name = 'Spectrogram_%s'%'-'.join(sensor_set) if type(sensor_set) is list else 'Spectrogram_%s'%sensor_set
    name='%s%s'%(name,save_Ext)
    if params:
        fName = 'floops_%s_m-n_%s-%s_f_%s%s'%\
             ( '-'.join(sensor_set),'-'.join(['%d'%param['m'] for param in params]),
              '-'.join(['%d'%param['n'] for param in params]),
              '-'.join(['%d'%(param['f']*1e-3) for param in params]),save_Ext) if \
                 type(params) is list else 'floops_%s_%s_m-n_%d-%d_f_%d%s.pdf'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,save_Ext)
    else:
        fName = 'C_Mod_Data_%s_%d'%(sensor_set,shotno)
        
    plt.close(name)
    fig,ax=plt.subplots(1,1,tight_layout=True,num=name)
    ax.pcolormesh(time*tScale,freq*1e-3,out_spect,shading='auto',rasterized=True)
    
    if doSave_Extractor and doSave: 
        plt.tick_params(left = False, bottom = False)
        tix_x=ax.get_xticklabels()
        tix_y=ax.get_yticklabels()
        #ax.set_xticklabels([]);ax.set_yticklabels([])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(doSave+'Spectrogram_%s.png'%fName)
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        #ax.set_xticklabels(tix_x);ax.set_yticklabels(tix_y)
        plt.tick_params(left = True, bottom = True)
    
    ax.set_xlabel('Time [%s]'%('s' if tScale==1 else 'ms'))
    ax.set_ylabel('Freq [kHz]')
    
    if doColorbar:
        norm = Normalize(np.min(out_spect),np.max(out_spect))
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,
                     label=clabel)
        
    if shotno:ax.text(.05,.95,'%d'%shotno,transform=ax.transAxes,fontsize=8,
            verticalalignment='top',bbox={'boxstyle':'round','alpha':.7,
                                          'facecolor':'white'})
    if f_lim:ax.set_ylim(f_lim)
    
    plt.show()

    if doSave: fig.savefig(doSave+'Spectrogram_%s.pdf'%fName,transparent=True)
        
    
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

###############################################################################

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
        
    











































