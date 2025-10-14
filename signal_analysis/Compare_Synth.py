#!/usr/bin/env python3
# -*- coding: utf-8 -

# Comparison between synthetic Mirnov signals generated for a specific shot, and the
# Actual measured signals
# Should be able to handle time and frequency domain signals

from header_signal_analysis import np, plt, sys, histfile
sys.path.append('../C-Mod/')
from get_Cmod_Data import __loadData
from header_Cmod import __doFilter
import scipy.optimize as opt



##########################################################################
def compare_time_domain(synthDataFileNameInfo, C_mod_shot, C_mod_time, comparison_sensor_names,\
                        doSave, time_window=1e-3,saveExt=''):
    # Load in synthetic signals from a time-dependent run
    # Load real signals, window in time
    # Plot

    # Load synthetic signals
    data_synth, time_synth, sensor_names_synth = \
        __load_synth_signals_time(synthDataFileNameInfo,R=6,L=60e-6,C=760e-12)
    
    # Load real signals 
    fft_magnitude, sensor_names, fft_freq, signals_window, time, R_mirnovs, Z_mirnovs = \
        __load_real_data(C_mod_shot,C_mod_time,time_window=1e-3,target_freq=target_freq,doPlot=True)
    
    # Plot Comparison
    plot_comparison_time(data_synth,time_synth, sensor_names_synth, \
                         signals_window, time, sensor_names, comparison_sensor_names, \
                            doSave, saveExt, )
    
########################################
def compare_frequency_domanin(synthDataFileNameInfo, C_mod_shot, C_mod_time, doSave,target_freq,\
                              saveExt='' ):
    # Load in synthetic signals from frequency scan
    # Load in real signals, compute filtered magnitude
    # Plot

    # Load synthetic signals
    probe_magnitude, sensor_names, freqs = \
        __load_synth_signals(synthDataFileNameInfo,R=6,L=60e-6,C=760e-12)

    # Load real signals
    fft_magnitude, sensor_names_real, fft_freq, signals_window, time, R_mirnovs, Z_mirnovs = \
        __load_real_data(C_mod_shot,C_mod_time,time_window=1e-3,target_freq=target_freq,doPlot=True)
    
    # Plot cooresponding sensors
    plot_comparison(probe_magnitude, sensor_names, freqs,
                    fft_magnitude, sensor_names_real, fft_freq,R_mirnovs,Z_mirnovs,\
                        doSave=doSave,save_ext=saveExt)

##################################################################################
def plot_comparison_time(data_synth,time_synth, sensor_names_synth, \
                         signals_window, time, sensor_names, comparison_sensor_names,\
                              doSave,save_ext):
    plt.close('Time Domain Comparison')
    fig, ax = plt.subplots(len(comparison_sensor_names),1,tight_layout=True,sharex=True,
                           num='Time Domain Comparison',squeeze=False)
    time_synth += time[0]
    for ind, comp_sensor in enumerate(comparison_sensor_names):
        arg_ind_real = sensor_names.index(comp_sensor)
        ax[ind,0].plot(time,signals_window[arg_ind_real],label=comp_sensor)

        arg_ind_synth = np.argwhere(sensor_names_synth==comp_sensor)
        ax[ind,0].plot(time_synth,data_synth[arg_ind_real],label='Synth.' if ind==0 else '',alpha=.6)

    ax[-1,0].set_xlabel('Time [s]')
    for ax_ in ax[:,0]:
        ax_.set_ylabel('Signal [T/s]')
        ax_.grid()
        ax_.legend(fontsize=8,loc='upper right')

    if doSave:
        fig.savefig('../output_plots/Time_Domain_Comparison_%s%s.pdf'%(doSave,save_ext), transparent=True)
################################################################################
def plot_comparison(probe_magnitude, sensor_names, freqs,
                    fft_magnitude, sensor_names_real, fft_freq,R_mirnovs,Z_Mirnovs,
                    target_freq=650e3, doSave='', save_ext=''):
    # Plot comparison between synthetic and real signals
    # No ordering of sensors yet

    # Reorder to match sensor order
    common_sensors = [name for name in sensor_names if name in sensor_names_real]
    indices_synth = [sensor_names.tolist().index(name) for name in common_sensors]
    indices_real = [sensor_names_real.index(name) for name in common_sensors]
    sensor_names = common_sensors
    probe_magnitude = probe_magnitude[indices_synth]
    fft_magnitude = fft_magnitude[indices_real]
    sensor_names_real = common_sensors

    # Build plot
    plt.close('Frequency Domain Comparison')
    fig,ax = plt.subplots(1,1,num='Frequency Domain Comparison',tight_layout=True,figsize=(5,4))
    ax.plot(sensor_names, probe_magnitude,'bo-', label= 'Synthetic Data',alpha=.7)
    ax.plot(sensor_names_real, fft_magnitude,'rs--', label='Real Data',alpha=.7)
    ax.set_ylabel('Signial Magnitude [T/s]')
    ax.set_xlabel('Sensor Name')

    ax.grid()

    ax.legend(fontsize=8)

    # Rotate x tick labels vertically and label every 3rd
    ax.set_xticks(range(0, len(sensor_names), 2))
    ax.set_xticklabels([sensor_names[i] for i in range(0, len(sensor_names), 2)])
    ax.tick_params(axis='x', rotation=90)

    if doSave:
        fig.savefig('../output_plots/Frequency_Domain_Comparison_%s%s.pdf'%(doSave,save_ext), transparent=True)

    # Separate plot of FFT real amplitude vs probe radial location
    plt.close('FFT Amplitude vs R')
    fig,ax = plt.subplots(1,1,num='FFT Amplitude vs R',tight_layout=True,figsize=(4,3))
    a_mirnovs = np.sqrt((R_mirnovs-.69)**2 + Z_Mirnovs**2)
    ax.plot(a_mirnovs[indices_real], fft_magnitude/target_freq,'rs', label='Real Data',alpha=.7)
    
    objective = lambda A: np.sum((fft_magnitude/target_freq - (A*1e-12)/a_mirnovs[indices_real]**12)**2)*1e8
    
    res = opt.minimize(objective, 3)
    A = res.x[0]
    print(f'Fitted A: {A}')
    a_range = np.linspace(min(a_mirnovs[indices_real]),max(a_mirnovs[indices_real]),20)
    ax.plot(a_range, A*1e-12/a_range**12,'k--', \
            label=r'Fit: $A/a^{12}$')
    
    # ax.plot(R_mirnovs[indices_synth], probe_magnitude[indices_synth]/target_freq,'bo', label='Synthetic Data',alpha=.7)
    ax.set_xlabel(r'$||\vec{a}||$ [m]')
    ax.set_ylabel(r'Signial Magnitude [$\sim$T]')
    ax.grid()
    
    ax.legend(fontsize=8)
    if doSave:
        fig.savefig('../output_plots/FFT_Amplitude_vs_R_%s%s.pdf'%(doSave,save_ext), transparent=True)
    print('Done')
#################################################################################
def __load_real_data(C_mod_shot,C_mod_time,time_window=1e-3,target_freq=650e3,doPlot=False):

    # Get raw data object
    bp_k = __loadData(C_mod_shot,pullData=['bp_k'],forceReload=['bp_k'*False],debug=True)['bp_k']
    time = bp_k.time
    signals = bp_k.data
    sensor_names = [n.upper() for n in bp_k.names]
    R_mirnovs = np.array(bp_k.R)
    Z_mirnovs = np.array(bp_k.Z)

    # Find time indicies within time window
    time_inds = np.where((time >= C_mod_time - time_window/2) & (time <= C_mod_time + time_window/2))[0]
    if len(time_inds) == 0:
        raise ValueError('No time points found within time window')
    
    # Do FFT
    signals_window = signals[:,time_inds]
    signals_window -= np.mean(signals_window,axis=1)[:,np.newaxis]
    fs = 1/(time[1]-time[0])
    fft_freq = np.fft.rfftfreq(len(signals_window[0]),1/fs)
    # Find index of target frequency
    freq_ind = np.argmin(np.abs(fft_freq - target_freq))
    print('Target frequency: {0:.2f} kHz, Closest frequency: {1:.2f} kHz'.format(target_freq*1e-3,fft_freq[freq_ind]*1e-3))
    
    # Do the FFT, get magnitude at target frequency
    fft_out = []
    for sig in signals_window:
        fft_out.append( np.fft.rfft(sig)/(len(sig)/2) )
    fft_out = np.array(fft_out)
    fft_magnitude = np.abs(fft_out[:,freq_ind])

    if doPlot:
        plt.close('Real Data FFT')
        fig,ax = plt.subplots(1,1,num='Real Data FFT',tight_layout=True,figsize=(4,3))
        ax.plot(fft_freq*1e-3,np.abs(fft_out).T,alpha=.8)
        ax.set_xlabel(r'Frequency [kHz]')
        ax.set_ylabel(r'FFT Magnitude')
        ax.set_title('C-Mod Shot %d at t=%2.3f ms'%(C_mod_shot,C_mod_time*1e3))
        ax.axvline(fft_freq[freq_ind]*1e-3,color='k',ls='--',label='Target Freq: %2.1f kHz'% (fft_freq[freq_ind]*1e-3))
        ax.legend()
        plt.show()

    # Run a highpass filter to isolate the target frequency
    signals_window = __doFilter(signals_window,time[time_inds],HP_Freq=target_freq*.9,LP_Freq=target_freq*1.1)
    
    return fft_magnitude, sensor_names, fft_freq, signals_window, time[time_inds], R_mirnovs, Z_mirnovs

####################################################################################
def __load_synth_signals(synthDataFileNameInfo,R,L,C):
    # Load in synthetic signals from frequency scan
    # Return frequency, real/imag, sensor names

    # Add on correction for RLC circuit, best guess
    fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz%s.npz'%\
        (synthDataFileNameInfo['mesh_file'],synthDataFileNameInfo['sensor_set'],\
          *synthDataFileNameInfo['calibration_frequency_limits'],\
            synthDataFileNameInfo['save_ext_input'])
    mirnov_Bode = np.load('../data_output/'+fName)

    print('Loaded Synthetic Bode data from ../data_output/%s'%fName)
    probe_signals = mirnov_Bode['probe_signals'] # Comes in as scaled flux: V = -N*A * B
    freqs = mirnov_Bode['freqs']
    probe_signals *= freqs[:,np.newaxis] * 2*np.pi * 1j # convert T/s (divide out A, switch current to voltage source)

    # Get RLC circuit correction
    mag_RLC, phase_RLC = __prep_RLC_Transfer_Function(R,L,C)

    sensor_names = mirnov_Bode['sensor_names']

    probe_magnitude = np.abs(probe_signals) * mag_RLC(2*np.pi*freqs)
    probe_magnitude = probe_magnitude[np.argmin(np.abs(freqs - 650e3)),:] # Get magnitude at 650 kHz    

    return probe_magnitude, sensor_names, freqs

#################################################################################
def __load_synth_signals_time(synthDataFileNameInfo,R,L,C):
    f_save = '../data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                    (synthDataFileNameInfo['sensor_set'],synthDataFileNameInfo['m'][0],\
                     synthDataFileNameInfo['n'],synthDataFileNameInfo['f']*1e-3,\
                        synthDataFileNameInfo['save_Ext'])
     
    hist_data = histfile(f_save)

    time = hist_data['time']
    sensor_names = [k for k in hist_data.keys() if k != 'time']
    data = np.array([hist_data[sig] for sig in sensor_names])


    data = np.diff(data,axis=1)/np.diff(time)[0] # Differentiate to get dB/dt
    time = time[:-1] + np.diff(time)[0]/2 # Center time vector
    mag_RLC, _ = __prep_RLC_Transfer_Function(R,L,C)
    # Convert to [T/s], include contribution from circuit 
    data *= mag_RLC(np.array([synthDataFileNameInfo['f']*2*np.pi])) 

    return data, time, sensor_names

####################################################################################3
def __prep_RLC_Transfer_Function(R = 6, L = 60e-6, C = 760e-12, plot_R=False,R_0=0.7):

    # RLC Circuit transfer function:
    # Transfer function H(w) for series RLC (output across the capacitor)
    # RLC circuit parameters (eventually, correct for individual sensors)
    # R = 28     # Ohms 
    # L = 60e-6      # Henry
    # C = 760e-12#780e-12     # Farads

    def Z_R(w):
        out = R_0 * np.ones_like(w)
        # Final factor is sqrt[resistivity/permeability]
        out[w/(2*np.pi)>=220e3] *= 1+(np.sqrt( w[w/(2*np.pi)>=220e3]/(2*np.pi))-np.sqrt(220e3))*0.0043 # Skin depth correction
        out += R # 
        return out
    Z_L =  lambda w: (1j * w * L)
    Z_C =  lambda w:1 / (1j * w * C)
    Z_total =  lambda w: Z_R(w) + Z_L(w) + Z_C(w)

    H =  lambda w: Z_C(w) / Z_total(w)  # Voltage across capacitor / input voltage

    # Magnitude and phase response
    mag_RLC =  lambda w: np.abs(H(w))
    phase_RLC =  lambda w: np.angle(H(w), deg=True)

    return mag_RLC, phase_RLC


########################`################################################
if __name__ == '__main__':
    synthDataFileNameInfo = {
        'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',#'vacuum_mesh.h5',#'C_Mod_ThinCurr_Combined-homology.h5',
        'sensor_set':'C_MOD_ALL',
        'calibration_frequency_limits':(650e3,650e3),
        'save_ext_input':'_FAR3D_NonLinear_Scale_30_3D_Tiles_Surface_Current',
        'm': [14,13,12,11,10,9,8],
        'n': 11,
        'f': 7e3,
        'save_Ext': '_FAR3D_NonLinear_Scale_30_3D_Tiles'
    }
    cmod_shot=1051202011
    time_point=1

    target_freq=567e3

    comparison_sensor_names =['BP02_GHK', 'BP1T_ABK']

    doSave=True

    compare_frequency_domanin(synthDataFileNameInfo, cmod_shot, time_point, doSave=doSave,\
                              target_freq=target_freq,saveExt='_Psi0.6')
    
    # synthDataFileNameInfo = {
    #     'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
    #     'sensor_set':'C_MOD_LIM',
    #     'calibration_frequency_limits':(650e3,650e3),
    #     'save_ext_input':'_FAR3D_NonLinear_Surface_Current',
    #     'm': [14,13,12,11,10,9,8],
    #     'n': 11,
    #     'f': 7e3,
    #     'save_Ext': '_FAR3D_NonLinear'
    # }
    compare_time_domain(synthDataFileNameInfo, cmod_shot, time_point, comparison_sensor_names,\
                        doSave, time_window=1e-3,saveExt='_Psi0.6')

    print('Done')