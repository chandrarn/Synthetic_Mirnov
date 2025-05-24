#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 21:00:26 2025
     Basic linear fit for m#
     
     
@author: rianc
"""
from get_signal_data import get_signal_data
from scipy.signal import hilbert
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def estimate_n(params={'m':8,'n':7,'r':.25,'R':1,'n_pts':50,'m_pts':70,\
           'f':1e3,'dt':1e-5,'T':3e-3,'periods':3,'n_threads':64,'I':10},coil_currs=None,\
           sensor_file='../signal_generation/C_Mod_Mirnov_Geometry_Phi.json',
                     sensor_set='C_MOD_MIRNOV_T',doVoltage=False,phi_sensor=[340],
                     doSave='../output_plots/',save_Ext='',timeScale=1e3,file_geqdsk='geqdsk',
                     filament=True,n=[-1,-7]):

    
    X,Y,Z, sensor_dict = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                    sensor_set,file_geqdsk,doVoltage)
    
    phases, angle, n_opt, phase_out  = calc_phase_diff(Y,Z,n)

    doPlot_n(angle,phases,n,n_opt,X,Z,doSave,save_Ext)
    
    return phase_out, X, Y, Z, angle, phases

###############################################################################
def calc_phase_diff(Y,Z,n):
    phases = []
    angle = []
    for ind_1,z_ in enumerate(Z):
        for ind_2,z in enumerate(z_):
            phases.append(np.unwrap(np.angle(hilbert(z))))
            angle.append(Y[ind_1][ind_2])
         
    phases = np.array(phases); angle = np.array(angle)
    phase_out=phases.copy()
    
    
    # Normalize 
    if angle[3]<0: angle[3:] += 360
    phases -= phases[0]
    
    
    phases = np.mean(phases,axis=1) *180/np.pi
    
    
    phases = phases[1:]; #angle=angle[1:]
    phases -= phases[0]
    phases = phases%360
    
    # best n fit
    fn = lambda n: np.mean(np.abs(phases-(n[0]*(angle[1:]-angle[1]))%360 )**2)
    res = minimize(fn,[n[-1]],bounds=[[-20,0]])
    n_opt = res.x
    
    #n_opt = np.arange(-30,0,dtype=int)[np.argmin([fn([n]) for n in np.arange(-30,0,dtype=int)])]

    plt.figure();plt.plot(np.arange(-30,0,dtype=int),[fn([n]) for n in np.arange(-30,0,dtype=int)]);plt.grid()
    #return angle, phases
    
    return phases, angle, n_opt, phase_out

#############################################################################
def doPlot_n(angle,phases,n,n_opt,X,Z,doSave,save_Ext):
    # Figure
    plt.close("Linear Phase Comparison%s"%save_Ext)
    fig,ax = plt.subplots(2,1,tight_layout=True,num='Linear Phase Comparison%s'%save_Ext)
    angle=np.insert(angle,3,np.nan)
    phases=np.insert(phases,2,np.nan)
    ax[0].plot(angle[1:],phases,'-*',label='Signal Phase')
    
    for n_ in n: ax[0].plot(np.linspace(*[0,angle[5]],100),(n_*np.linspace(*[0,angle[5]],100)-n_*angle[1])%360,'--',label=r'$n$=%d'%n_)
    for n_ in [n_opt]: ax[0].plot(np.linspace(*[0,angle[5]],100),(n_*np.linspace(*[0,angle[5]],100) -n_*angle[1] )%360,'--',label=r'$\hat{n}$=%2.2f'%n_)
    ax[0].set_xlabel(r'Geometric Phase $\Delta\phi$ [deg]')
    ax[0].set_ylabel(r'Signal Phase $\bar{\Delta\phi}$ [deg]')
    ax[0].legend(fontsize=8,handlelength=1)
    
    ax[1].plot(X[0],Z[0].T) 
    ax[1].plot(X[0],Z[1].T)
    ax[1].legend([r'$\phi=%1.1f^\circ$'%a for a in angle],fontsize=8,handlelength=1,ncols=2)
    
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('B [?]')
    for i in range(2):ax[i].grid()
    
    plt.show()
    
    if doSave: fig.savefig(doSave+'Linear_Phase_Comparison%s.pdf'%save_Ext,transparent=True)
################################################
if __name__ == '__main__': estimate_n()