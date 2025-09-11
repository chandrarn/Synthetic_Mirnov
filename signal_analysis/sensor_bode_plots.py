# Output and compare magnitude and frequency response of Mirnov sensors
# to the expected response from the C-Mod Mirnov sensors.
# This script is used to generate the Bode plots for the C-Mod Mirnov sensors.

from header_signal_analysis import np, plt, sys
sys.path.append('../C-Mod')
from get_Cmod_Data import __loadData


def make_plots_bode(mesh_file, sensor_set, calibration_magnitude, \
                    calibration_frequency_limits, comparison_shot,doSave):

    """
    Generate Bode plots for the Mirnov sensors.
    
    Parameters:
    mesh: The mesh object containing the Mirnov sensor data.
    sensor_set: The set of sensors to plot.
    params: Parameters for the simulation.
    """
    
    # Load the Mirnov Bode data
    fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz.npz'%\
        (mesh_file,sensor_set, *calibration_frequency_limits)
    mirnov_Bode = np.load('../data_output/'+fName)
    probe_signals = mirnov_Bode['probe_signals'] # Comes in as scaled flux: V = -N*A * B
    freqs = mirnov_Bode['freqs']
    probe_signals *= 1j*freqs[:,np.newaxis] # convert to Voltage
    sensor_names = mirnov_Bode['sensor_names']

    # RLC Circuit transfer function:
    # Transfer function H(w) for series RLC (output across the capacitor)
    # RLC circuit parameters (eventually, correct for individual sensors)
    R = 6      # Ohms 
    L = 60e-6      # Henry
    C = 1.2e-9     # Farads
    def Z_R(w):
        out = R * np.ones_like(w)
        out[w/(2*np.pi)>=220e3] *= 1+(np.sqrt( w[w/(2*np.pi)>=220e3]/(2*np.pi))-np.sqrt(220e3))*0.0043 # Skin depth correction
        return out
    Z_L =  lambda w: (1j * w * L)
    Z_C =  lambda w:1 / (1j * w * C)
    Z_total =  lambda w: Z_R(w) + Z_L(w) + Z_C(w)

    H =  lambda w: Z_C(w) / Z_total(w)  # Voltage across capacitor / input voltage

    # Magnitude and phase response
    mag_RLC =  lambda w: np.abs(H(w))
    phase_RLC =  lambda w: np.angle(H(w), deg=True)

    
    # Initialize lists to store transfer function data
    transfer_mag = []
    transfer_phase = []

    # Need to pull actual sensors for comparison
    mirnovs  =  __loadData(comparison_shot,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']


    # Build plot
    plt.close('Transfer_Function_Synthetic_Mirnov')
    fig, ax = plt.subplots(7,7, num='Transfer_Synthetic_Mirnov',tight_layout=True,sharex=True,
                               sharey=True,figsize=(10,10))
    ax = ax.flatten()
    
    # Loop through each sensor in the set
    for ind, name in enumerate(mirnovs.names):
        # Calculate transfer function magnitude and phase

        # Verify that the sensor exists in the probe_signals
        try: # All sensors should exist, but just in case
            sensor_ind = np.char.lower(sensor_names).tolist().index(name)
        except: continue
        # Magnitude signal needs to be in T/s, not just T
        transfer_mag.append(np.abs(probe_signals[:,sensor_ind]) / calibration_magnitude * mag_RLC(2*np.pi*freqs))
        transfer_phase.append(np.angle(probe_signals[:,sensor_ind],deg=True) + phase_RLC(2*np.pi*freqs)+360)

        # Plot the transfer function
        ax[ind].plot(freqs*1e-6, transfer_mag[ind], label=name)
        ax_ = ax[ind].twinx()
        ax_.plot(freqs*1e-6, transfer_phase[ind], label='Phase', color='orange', alpha=.6)
        ax_.set_ylim([0,360])
        ax[ind].grid()
        ax[ind].legend(fontsize=8,handlelength=0.5,loc='upper left')

        if ind == 7 * 3: ax[ind].set_ylabel('Transfer Magnitude [T/s/A]')
        if ind == 7*6 + 3: ax[ind].set_xlabel('Frequency [MHz]')
        if ind == 7*3 + 6: ax_.set_ylabel('Transfer Phase [deg]')
        if (ind+1) % 7 != 0: ax_.set_yticklabels([])

    if doSave: fig.savefig(doSave+'Cmod_Transfer_Function_Synthetic.pdf',dpi=300,transparent=True)
    plt.show()

###################################################
if __name__ == '__main__':
    # Example usage
    mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    sensor_set = 'C_MOD_ALL' # Example sensor names
    calibration_magnitude = 6.325 # Replace with actual calibration magnitude
    calibration_frequency_limits = (10, 1e6)  # Frequency range in Hz
    comparison_shot = 1151208901 # Example shot number for comparison
    doSave = '../output_plots/'  # Directory to save plots

    make_plots_bode(mesh_file, sensor_set, calibration_magnitude,\
                     calibration_frequency_limits, comparison_shot,doSave)
    print('Bode plots generated and saved to %sCmod_Transfer_Function_Synthetic.pdf'%doSave)