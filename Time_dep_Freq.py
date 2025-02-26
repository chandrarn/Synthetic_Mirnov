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
periods=3
I_KM = lambda t: 10 + 2*np.sin(2*np.pi*t*3)
F_KM = lambda t: 7e3 - 2e3*np.sin(2*np.pi*t*3)

# AE 
def I_AE(t,dead_time=.2):
    # Break into repeating bands
    local_t = t % (1/periods)
    I_out = np.zeros((len(t)))
    I_out[local_t < (1/periods)*(1-dead_time)] = \
         5 + 4*local_t[local_t < (1/periods)*(1-dead_time)]/((1/periods)*(1-dead_time))
    I_out[local_t >= (1/periods)*(1-dead_time)] = 0
    return I_out
    
def F_AE(t,dead_time=.2):
    # Break into repeating bands
    local_t = t % (1/periods)
    f_out = np.zeros((len(t)))
    # Local times slices
    t_shift = local_t[local_t < (1/periods)*(1-dead_time)]/((1/periods)*(1-dead_time))
    f_out[local_t < (1/periods)*(1-dead_time)] = \
         500e3 - 150e3 * (1-np.exp(-t_shift*4))
    f_out[local_t >= (1/periods)*(1-dead_time)] = 0
    return f_out

if debug_plot:
    plt.close('test_f_i')
    fig,ax = plt.subplots(1,1,tight_layout=True,num='test_f_i')
    time = np.linspace(0,1,100)
    
    I_k = I_KM(time); f_k = F_KM(time)
    I_ae = I_AE(time)
    f_ae = F_AE(time)
    f_ae[f_ae==0] = np.nan
    
    norm = colors.Normalize(5,12)
    #norm_ae = colors.Normalize(min(I_ae),max(I_ae))
    
    for i,t in enumerate(time):ax.plot(t,f_k[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_k[i])))
    for i,t in enumerate(time):ax.plot(t,f_ae[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_ae[i])))
    ax.grid()
    ax.set_xlabel('Time [norm]')
    ax.set_ylabel('f [kHz]')
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='Current [A]')
    
    plt.show()