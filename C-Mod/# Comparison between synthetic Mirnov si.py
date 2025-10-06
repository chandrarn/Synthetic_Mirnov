# Comparison between synthetic Mirnov signals generated for a specific shot, and the
# Actual measured signals
# Should be able to handle time and frequency domain signals

from header_signal_analysis import np, plt



########################################
def compare_frequency_domanin(synthDataFileNameInfo, C_mod_shot, doSave):
    # Load in synthetic signals from frequency scan
    # Load in real signals, compute filtered magnitude
    # Plot

    # Load synthetic signals



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