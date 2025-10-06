#!/usr/bin/env python3
# -*- coding: utf-8 -

# Comparison between synthetic Mirnov signals generated for a specific shot, and the
# Actual measured signals
# Should be able to handle time and frequency domain signals

from header_signal_analysis import np, plt, sys
sys.path.append('../C-Mod/')
from get_Cmod_Data import BP_K



##########################################################################
def compare_time_domain(synthDataFileNameInfo, C_mod_shot, C_mod_time, doSave, time_window=1e-3):
    # Load in synthetic signals from a time-dependent run
    # Load real signals, window in time
    # Plot

    # Load synthetic signals
    probe_signals, sensor_names, time = \
        __load_synth_signals_(synthDataFileNameInfo)

########################################
def compare_frequency_domanin(synthDataFileNameInfo, C_mod_shot, C_mod_time, doSave,target_freq ):
    # Load in synthetic signals from frequency scan
    # Load in real signals, compute filtered magnitude
    # Plot

    # Load synthetic signals
    probe_magnitude, sensor_names, freqs = \
        __load_synth_signals(synthDataFileNameInfo,R=6,L=60e-6,C=760e-12)

    # Load real signals
    fft_magnitude, sensor_names_real, fft_freq, signals_window, time = \
        __load_real_data(C_mod_shot,C_mod_time,time_window=1e-3,target_freq=target_freq,doPlot=True)
    
    # Plot cooresponding sensors
    plot_comparison(probe_magnitude, sensor_names, freqs,
                    fft_magnitude, sensor_names_real, fft_freq)

################################################################################
def plot_comparison(probe_magnitude, sensor_names, freqs,
                    fft_magnitude, sensor_names_real, fft_freq,
                    target_freq=650e3, doSave='', save_ext=''):
    # Plot comparison between synthetic and real signals
    # No ordering of sensors yet

    # Build plot
    plt.close('Frequency Domain Comparison')
    fig,ax = plt.subplots(1,1,num='Frequency Domain Comparison',tight_layout=True,figsize=(5,4))
    ax.plot(sensor_names, probe_magnitude,'bo-', label= 'Synthetic Data',alpha=.7)
    ax.plot(sensor_names_real, fft_magnitude,'rs--', label='Real Data',alpha=.7)
    ax.set_ylabel('Signial Magnitude [T/s]')
    ax.set_xlabel('Sensor Number')

    ax.grid()

    ax.legend(fontsize=8)

    if doSave:
        fig.savefig('Frequency_Domain_Comparison_%s%s.pdf'%(doSave,save_ext), transparent=True)

#################################################################################
def __load_real_data(C_mod_shot,C_mod_time,time_window=1e-3,target_freq=650e3,doPlot=False):

    # Get raw data object
    bp_k = BP_K(C_mod_shot)
    time = bp_k.time
    signals = bp_k.data
    sensor_names = bp_k.names

    # Find time indicies within time window
    time_inds = np.where((time >= C_mod_time - time_window/2) & (time <= C_mod_time + time_window/2))[0]
    if len(time_inds) == 0:
        raise ValueError('No time points found within time window')
    
    # Do FFT
    signals_window = signals[:,time_inds]
    signals_window -= np.mean(signals_window,axis=1)[:,np.newaxis]
    fs = 1/(time[1]-time[0])
    fft_freq = np.fft.fftfreq(len(signals_window[0]),1/fs)[:len(signals_window[0])//2]
    # Find index of target frequency
    freq_ind = np.argmin(np.abs(fft_freq - target_freq))
    print('Target frequency: {0:.2f} kHz, Closest frequency: {1:.2f} kHz'.format(target_freq*1e-3,fft_freq[freq_ind]*1e-3))
    
    # Do the FFT, get magnitude at target frequency
    fft_out = []
    for sig in signals_window:
        fft_out.append( np.fft.fft(sig)[:len(sig)//2]/(fs * len(sig)) )
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
    
    return fft_magnitude, sensor_names, fft_freq, signals_window, time[time_inds]

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
        'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
        'sensor_set':'C_MOD_ALL',
        'calibration_frequency_limits':(650e3,650e3),
        'save_ext_input':'_FAR3D_NonLinear_Surface_Current'
    }
    cmod_shot=1051202011
    time_point=1

    target_freq=560e3

    compare_frequency_domanin(synthDataFileNameInfo, cmod_shot, time_point, doSave='C-Mod_1051202011_t1ms',\
                              target_freq=target_freq)