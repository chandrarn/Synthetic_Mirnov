#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:10:11 2025

@author: rian
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm,rc, colors
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
# helper function for time dependent frequency

# Sawtooth cycle: Assume kink-tearing mode hassinusoidal f(t), amplitude inverse
# proportion to frequency
# Assume AE has parabolic frequency 

# Assume F(t), I(t) accepts t in [0,1], any sub "frequency" is normalized to this

debug_plot = False

# Kink-tearing mode
period=.3e-3
I_KM = lambda t: 10 - 1*np.sin(2*np.pi*t*1e3)
F_KM = lambda t:  2*np.pi*7e3*t + 2e3*np.sin(2*np.pi*t*1e3)/1e3 
F_KM_plot = lambda t: 7e3 + 2e3*np.sin(2*np.pi*t*1e3) 

# AE 
def I_AE(t,dead_time=.2):
    # Break into repeating bands
    local_t = t % period
    I_out = np.zeros((len(t)))
    I_out[local_t < (period)*(1-dead_time)] = \
         5 + 1*local_t[local_t < (period)*(1-dead_time)]/((period)*(1-dead_time))
    
    t_shift = local_t[local_t >= (period)*(1-dead_time)]
    step_val = np.where(local_t >= (period)*(1-dead_time))[0][0]-1
    I_out[local_t >= (period)*(1-dead_time)] = I_out[step_val]*np.exp(-1e5*(t_shift-(period)*(1-dead_time)) )
    return I_out
    
def F_AE(t,dead_time=.2,lam=5e3,f_carrier=50e3,f_mod=30e3):
    # Break into repeating bands
    t=np.array(t)
    local_t = t % (period)
    f_out = np.zeros((len(t)))
    
    # Local times slices
    t_shift = local_t[local_t < (period)*(1-dead_time)]#/((1/periods)*(1-dead_time))
    f_out[local_t < (period)*(1-dead_time)] = \
        2*np.pi*(f_carrier*t_shift - f_mod*(t_shift+np.exp(-t_shift*lam)/lam))
        #300e3 - 30e3 * (1-np.exp(-t_shift*4))
         
    t_shift = local_t[local_t >= (period)*(1-dead_time)]
    step_val = np.where(local_t >= (period)*(1-dead_time))[0][0]-1
    f_out[local_t >= (period)*(1-dead_time)] = f_out[step_val]*np.exp(-1*(t_shift-(period)*(1-dead_time)) )
    return f_out

def F_AE_plot(t,dead_time=.2,lam=5e3,f_carrier=50e3,f_mod=30e3):
    t=np.array(t,ndmin=1)
    # Break into repeating bands
    local_t = t % (period)
    f_out = np.zeros((len(t)))
    # Local times slices
    t_shift = local_t[local_t < (period)*(1-dead_time)]#/((1/periods)*(1-dead_time))
    f_out[local_t < (period)*(1-dead_time)] = \
        (f_carrier - f_mod*(1-np.exp(-t_shift*lam)))
        #300e3 - 30e3 * (1-np.exp(-t_shift*4))
         
         
    f_out[local_t >= (period)*(1-dead_time)] = 0
    return f_out

if debug_plot:
    plt.close('test_f_i')
    fig = plt.figure(tight_layout=True,num='test_f_i')
    
    time = np.linspace(0,.001,1000)
    
    I_k = I_KM(time); f_k = F_KM_plot(time)#F_KM(time)
    I_ae = I_AE(time)
    f_ae = F_AE_plot(time)
    f_ae[f_ae==0] = np.nan
    
    norm = colors.Normalize(5,12)
    #norm_ae = colors.Normalize(min(I_ae),max(I_ae))
    ax = fig.add_subplot(1,2,1)
    for i,t in enumerate(time):ax.plot(t,f_k[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_k[i])))
    for i,t in enumerate(time):ax.plot(t,f_ae[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_ae[i])))
    ax.grid()
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('f [kHz]')
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='Current [A]')
    
    ax=fig.add_subplot(2,2,2)
    time = np.linspace(0,4e-3,1000)
    ax.plot(time*1e3,I_KM(time)*np.cos(F_KM(time)))
    ax.grid()
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Current [A]')
    ax=fig.add_subplot(2,2,4)
    time = np.linspace(0,1e-3,2000)
    ax.plot(time*1e3,I_AE(time)*np.cos(F_AE(time)))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Current [A]')
    ax.grid()
    plt.show()