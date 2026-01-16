#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 16:37:52 2025
    Plot C-Mod magnetic data in the contour style of the synthetic diagnostic plots
@author: rianc
"""

from header_Cmod import sys, np,__doFilter,plt, correct_Bode, lombscargle,Normalize,cm
sys.path.append('../signal_analysis/')
from plot_sensor_output import doPlot
from estimate_n import calc_phase_diff, doPlot_n
from get_signal_data import __select_sensors, __gen_surface_data
from get_Cmod_Data import BP, BP_T
from eqtools import CModEFITTree


def plot_Current_Surface(shotno=1160930034,sensor_set='BP_T', doVoltage=False,
                         phi_sensor=0,doSave='/home/rianc/Documents/Synthetic_Mirnov/output_plots/',save_Ext='',timeScale=1,
                         file_geqdsk='g1051202011.1000',tLim=[0.3,1.1],
                         HP_Freq=.2e3, LP_Freq=30e3,cLims=None,
                         doBode=False,calc_n=True,n=[-1],chan='BP1T_ABK'):
    
    # Generate data
    # Select usable sensors from set
    sensor_dict = __select_sensors('C_Mod_'+sensor_set,None,phi_sensor,
                                   file_geqdsk,None,shotno,tLim)
    
    #return sensor_dict
    hist_file =  build_data_dict(sensor_dict)
    
    do1Dplot(hist_file['time'], hist_file, HP_Freq, LP_Freq, sensor_set, shotno, chan, doSave,
             save_Ext,timeScale)
    
    doFilter_hist(hist_file,HP_Freq,LP_Freq)
    
    
    #return hist_file
    if doBode:doCorrectBode(hist_file)
    
    #return hist_file
    # build datasets
    X,Y,Z = __gen_surface_data(sensor_dict,hist_file,doVoltage,{'dt':1,'f':1},
                               'C_Mod_'+sensor_set, None)
    
   
    '''
    doFilter(X,Z,HP_Freq, LP_Freq)
   '''
   
    #return X,Y,Z,sensor_dict,hist_file

    doPlot(sensor_set,save_Ext,sensor_dict,X,Y,Z,timeScale,doSave,False,
           doVoltage,False,{'unit':'T','scale':1},cLims,shotno=shotno)
    
    if calc_n:
        doSpect(X,Y,Z,doSave,save_Ext)
        
        Y=Y[::-1]; Z=Z[::-1]
        
        phases, angle, res, phase_out  = calc_phase_diff(Y,Z,n)

        doPlot_n(angle,phases,n,res,X,Z,doSave,'_Real')
    
    return X,Y,Z,sensor_dict,hist_file
###############################################################################
def doCorrectBode(hist_file):
    for sensor_name in hist_file:
        if sensor_name == 'time': continue
        hist_file[sensor_name] = correct_Bode(hist_file[sensor_name],hist_file['time'],sensor_name)[0]
###############################################################################
def build_data_dict(sensor_dict):
    hist_file ={}
    
    hist_file['time'] = sensor_dict[0][0]['Time']
    for data_set in sensor_dict:
        for signal in data_set:
            hist_file[signal['Sensor']]=signal['Signal']
    
    return hist_file
###############################################################################
def doFilter(X,Z,HP_Freq, LP_Freq):
    for i in range(len(X)):
        Z[i]=__doFilter(Z[i], X[i], HP_Freq, LP_Freq)
def doFilter_hist(hist_file,HP_Freq, LP_Freq,tReduce=50):
    
    for sens_name in hist_file:
        if sens_name == 'time': continue
        hist_file[sens_name]=__doFilter(hist_file[sens_name], hist_file['time'],\
                                        HP_Freq, LP_Freq)[0][tReduce:-tReduce]
    hist_file['time'] = hist_file['time'][tReduce:-tReduce]
###############################################################################
def do1Dplot(time, signal, HP_Freq, LP_Freq, sensor_set, shotno, chan, doSave='',
         save_Ext='',timeScale=1):
    plt.close('1D_%s_%s'%(sensor_set,chan))
    fig,ax=plt.subplots(1,1,num='1D_%s_%s'%(sensor_set,chan),tight_layout=True,
                        figsize=(4,2))
    ax.plot(time*timeScale,signal[chan],label='Raw Signal')
    
    #filt = doFilter(X[0][0],Z[0][0],None, HP_Freq)
    filt_h = __doFilter(signal[chan], time, None, HP_Freq)
    filt_l = __doFilter(signal[chan], time, None, LP_Freq)
    ax.plot(time*timeScale,filt_h[0],  label='%d kHz High-pass'%(HP_Freq*1e-3))
    ax.plot(time*timeScale,filt_l[0], label='%d kHz Lowpass'%(LP_Freq*1e-3))
    ax.grid()
    ax.legend(fontsize=8)
    if timeScale==1: tUnit = 's'
    if timeScale==1e3: tUnit = 'ms'
    if timeScale==1e6: tUnit = r'$\mu$s'
    ax.set_xlabel('Time [%s]'%tUnit)
    ax.set_ylabel(r'B$_\theta$ [arb]')
    plt.show()
    if doSave:
        fig.savefig(doSave+'Sensor_C_Mod_%s_%s_%d_t_%2.2f_%2.2f%s.pdf'%\
            (sensor_set,chan,shotno,time[0],time[-1],save_Ext),transparent=True)
###############################################################################
def doSpect(X,Y,Z,doSave,save_Ext):
    
    y_ = np.concatenate((Y[0],Y[1]))/360
    spatial_frq = np.arange(-50,51)
    out = []
    for ind in range(len(X[0])):
        z_ = np.concatenate((Z[0][:,ind],Z[1][:,ind]))
        out.append(lombscargle(y_, z_, spatial_frq,normalize=True,floating_mean=True))
        
    # do plot
    plt.close('Periodogram%s'%save_Ext)
    fig,ax = plt.subplots(2,1,tight_layout=True,num='Periodogram%s'%save_Ext)
    norm = Normalize(np.min(out),np.max(out)) 
    
    ax[0].contourf(X[0]*1e6,spatial_frq,np.array(out).T,cmap='plasma',zorder=-5,levels=50)
    #fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax[0],
    #             label= r'$n\#$')
    ax[0].set_xlabel(r'Time [$\mu$s]')
    ax[0].set_ylabel(r'$n\#$')
    
    ax[1].plot(spatial_frq,np.mean(out,axis=0))
    ax[1].fill_between(spatial_frq,np.mean(out,axis=0)-np.std(out,axis=0),
                       np.mean(out,axis=0)+np.std(out,axis=0),alpha=.4)
    
    ax[1].grid()
    ax[1].set_xlabel(r'$n\#$')
    ax[1].set_ylabel(r'[?]')
    plt.show()
    if doSave:fig.savefig(doSave+'Periodogram%s.pdf'%save_Ext,transparent=True)
###############################################################################
if __name__ == '__main__':
    
    plot_Current_Surface()
    print('Done')