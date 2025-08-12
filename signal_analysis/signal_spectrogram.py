#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:08:15 2025

@author: rian
"""

from header_signal_analysis import rolling_spectrogram, np, plt, json, histfile, Normalize, cm

from get_signal_data import get_signal_data

import get_Cmod_Data as gC

from spectrogram_Cmod import plot_spectrogram, signal_spectrogram_C_Mod

from header_Cmod import __doFilter

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

def gen_single_spect_multimode(params,sensor_names=['BP1T_ABK','BP01_ABK'],\
        sensor_set='C_MOD_LIM',mesh_file='C_Mod_ThinCurr_Combined-homology.h5',
        pad = 800, fft_window = 400,doSave='', filament = True,f_lim=[0,100],
        save_Ext='',HP_Freq=10e3,LP_Freq=None,cLim=[0,1],doSave_data=False):
    # Generate a spectrogram for a given sensor or set of sensors, averaged
    
    # Pull saved data
    filename = gen_filename(params,sensor_set,mesh_file,save_Ext=save_Ext)
    hist = histfile(filename+'.hist')
    time = hist['time'][:-1]

    signals = __doFilter([hist[name] for name in sensor_names],time,HP_Freq, LP_Freq)
    
    signals = np.array([hist[name] for name in sensor_names])
    signals = np.diff(signals,axis=1)/(time[1]-time[0]) # Remove DC offset

    plt.close('Test_Current')
    fig,ax=plt.subplots(1,1,tight_layout=True,num='Test_Current')
    for ind,s in enumerate(signals):
        ax.plot(time*1e3,s,label=sensor_names[ind])
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Signal [T]')
    ax.legend()
    # Calculate spectrogram 
    time, freq, out_spect = rolling_spectrogram(time, signals,pad=pad,
                                                fft_window=fft_window)
    out_spect=np.mean(out_spect,axis=0)

    # Run spectrogram
    plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,
                               filament,save_Ext,f_lim,doColorbar=True,\
                                clabel=r'B [T/s, Mode Amplitude Arb]',\
                                cLim=cLim,doSave_data=doSave_data)
    # Loop over sensors
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
                               filament,plot_ext,f_lim,doColorbar=True)
    
    return time, freq, out_spect,signals



        
    

###############################################################################
# File name generator new
def gen_filename(param,sensor_set,mesh_file,save_Ext='',archiveExt=''):
    f=param['f'];m=param['m'];
    n=param['n']

    if type(f) is float: f_out = '%d'%f*1e-3 
    else: f_out = 'custom'
    if type(m) is not list: mn_out = '%d-%d'%(m,n)
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])
        
    

    f_save = '../data_output/%sfloops_filament_%s_m-n_%s_f_%s_%s%s'%\
                (archiveExt,sensor_set,mn_out,f_out,mesh_file,save_Ext)
    
    return f_save
        
        
        
################################################################################
if __name__ == '__main__':
    # Example usage
    params={'m':[1, 6, 10],'n':[1, 2, 9],'f':[],'T':1e-3}
    #params={'m':[1,3,12],'n':[1,2,9],'f':[]}
    filament = True
    save_Ext = ''#'_Multimode'
    sensor_set='C_MOD_LIM'
    #sensor_set = 'C_MOD_ALL'
    mesh_file='C_Mod_ThinCurr_Combined-homology.h5'
    mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    #mesh_file='vacuum_mesh.h5'
    sensor_names=['BP01_ABK','BP1T_ABK','BP2T_GHK', 'BP02_GHK']
    fft_window = 230
    pad = 230
    doSave = '../output_plots/'
    f_lim= [0,220]
    cLim = [0,.2]
    doSave_data = True
    spectrogram_params = {'pad':pad,'fft_window':fft_window,'block_reduce':(fft_window,0)}


    gen_single_spect_multimode(params=params,sensor_set=sensor_set,fft_window=fft_window,
                               mesh_file=mesh_file,save_Ext=save_Ext,pad=pad,f_lim=f_lim,
                               doSave=doSave,filament=filament,sensor_names=sensor_names,
                               cLim=cLim,doSave_data=doSave_data)
        
    
    diag,signals,time,out_spect,out_spect_all_cplx, freq, sensor_name = signal_spectrogram_C_Mod(
            shotno=None,  # No shot number for synthetic data
            params=params,
            sensor_name=sensor_names,
            sensor_set=sensor_set,
            pad=spectrogram_params['pad'],
            fft_window=spectrogram_params['fft_window'],
            doSave=doSave,
            save_Ext='_New',
            doPlot=True,
            mesh_file = mesh_file,
            archiveExt='',
            tLim=[0,params['T']],
            block_reduce=spectrogram_params['block_reduce'],
            filament=True,
            plot_reduce=(1,1)
        )
    print('Done generating spectrograms')
        
        