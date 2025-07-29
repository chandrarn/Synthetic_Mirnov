#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:10:11 2025

@author: rian
"""
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm,rc, colors
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
# helper function for time dependent frequency

# Sawtooth cycle: Assume kink-tearing mode has sinusoidal f(t), amplitude inverse
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

##################################################
# generate coupled f(t), I(t)
def gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,
                     I_chirp_smooth_percent=0.05,random_seed=None):
    # Critical point: the numeric value of time isn't important till the very end
    # The value of frequency is what is slowly being modulated, time is the conversion
    # to phase factor at the end
    # periods: number of repeating periods in t
    # dead_fraction: percentage of period at the end of the chip which is dark
    # f_carrier: carrier frequency in Hz
    # f_mod: frequency modulation as a function of a single period
    # I_mod: current modulation as a function of a single period
    f_out = np.zeros((len(time),))
    f_out_plot = np.zeros((len(time),))
    I_out = np.zeros((len(time),))


    # bands of indicies for each chirp period
    chirp_boundary = np.linspace(0,len(time),periods+1,endpoint=True,dtype=int)
    chirp_bands = [np.arange(chirp_boundary[ind], chirp_boundary[ind+1],dtype=int) \
                   for ind in range(len(chirp_boundary)-1)]
    # Current smoothing
    I_chirp_smooth = int(I_chirp_smooth_percent*len(chirp_bands[0]))

    for ind, band in enumerate(chirp_bands):
        local_t = np.linspace(0,1, int(len(band)*(1-dead_fraction)), endpoint=False)
        f_chirp = f_mod(local_t) # frequency modulation
        I_chirp = I_mod(local_t) # current modulation
        # Give I_chirp a more gentle rise and fall
        I_chirp[0:I_chirp_smooth] = np.linspace(0,I_chirp[I_chirp_smooth],I_chirp_smooth)
        I_chirp[-I_chirp_smooth:] = np.linspace(I_chirp[-I_chirp_smooth],0,I_chirp_smooth)

        f_out[band[0:len(local_t)]] = f_chirp*2*np.pi*time[chirp_bands[0][0:len(local_t)]]
        f_out_plot[band[0:len(local_t)]] = f_chirp
        
        # Random noise component goes into each filament separately

        I_out[band[0:len(local_t)]] = I_chirp 
    return f_out, I_out, f_out_plot
##################################################
if debug_plot:
    plt.close('test_f_i')
    fig = plt.figure(tight_layout=True,num='test_f_i')
    
    time = np.linspace(0,10e-3,int(10e-3/1e-6))
    periods = 5
    dead_fraction = 0.4
    f_mod = lambda t: 7e3 + 3e3*t
    I_mod = lambda t: .1*4*(5 + 7*t**4)
    
    f_out_1, I_out_1, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                      random_seed=42)

    
    periods = 2
    dead_fraction = 0.2
    f_mod = lambda t: 35e3 + 5e3*t
    I_mod = lambda t: .05*(6 + 3*np.sin(periods*2*np.pi*t))
    
    f_out_2, I_out_2, f_out_plot_2 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                        random_seed=42)

    periods = 3
    dead_fraction = 0.2
    f_mod = lambda t: 200e3 - 5e3*t**2
    I_mod = lambda t: .02*(2 + 4*t**2)
    
    f_out_3, I_out_3, f_out_plot_3 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                        random_seed=42)


    f_out_plot_1[f_out_plot_1==0] = np.nan
    f_out_plot_2[f_out_plot_2==0] = np.nan
    f_out_plot_3[f_out_plot_3==0] = np.nan
    
    norm = colors.Normalize(1,7)
    #norm_ae = colors.Normalize(min(I_ae),max(I_ae))
    ax = fig.add_subplot(1,2,1)
    for i,t in enumerate(time):
        ax.plot(t*1e3,f_out_plot_1[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_out_1[i])))
        ax.plot(t*1e3,f_out_plot_2[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_out_2[i])))
        ax.plot(t*1e3,f_out_plot_3[i]*1e-3,'*',c=plt.get_cmap('viridis')(norm(I_out_3[i])))

    ax.grid()
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('f [kHz]')
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='Current [A]')
    
    ax=fig.add_subplot(3,2,2)
    # time = np.linspace(0,4e-3,5000)
    ax.plot(time*1e3,I_out_1*np.cos(f_out_1))
    
    #ax.plot(time*1e3,I_out_3*np.cos(f_out_3),alpha=.5)
    ax.grid()
    #ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Current [A]')
   
    ax=fig.add_subplot(3,2,4)
    # time = np.linspace(0,1e-3,10000)
    ax.plot(time*1e3,I_out_2*np.cos(f_out_2))
    #ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Current [A]')
    ax.grid()

    ax=fig.add_subplot(3,2,6)
    # time = np.linspace(0,1e-3,10000)
    ax.plot(time*1e3,I_out_3*np.cos(f_out_3))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Current [A]')
    ax.grid()
    plt.show()
    plt.savefig('../output_plots/multimode_target_spectrum.pdf',transparent=True)

if __name__ == '__main__':
    # Test the coupled frequency generation
    time = np.linspace(0,10e-3,1000)
    periods = 3
    dead_fraction = 0.2
    f_mod = lambda t: 1e3 + 2e3*t
    I_mod = lambda t: 5 + 1*t**2
    
    f_out, I_out, f_out_plot = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod)
    
    plt.figure()
    plt.plot(time*1e3,f_out_plot*1e-3,label='f(t)')
    plt.plot(time*1e3,I_out,label='I(t)')
    plt.xlabel('Time [ms]')
    plt.ylabel('Frequency [kHz] / Current [A]')
    plt.legend()
    plt.grid()
    plt.show()
    

    print('Done')