# Emperical Bode Calibration

import get_Cmod_Data as gC
from header_Cmod import np, plt,sys,ShortTimeFFT, csd, hilbert
from Emperical_RLC_C_mod import __prep_RLC_Transfer_Function
sys.path.append('../signal_generation/')
from header_signal_generation import gen_coupled_freq, histfile, make_smoothing_spline, BSpline


# def compareBode(shotno=1151208900,doSave='',doPlot=True,tLim=[.05, .997]):

#     # Get driver signal
#     time, calib_mag, calib_freq, calib_phase, calib_complex = \
#         bode_driver(shotno,tLim=tLim)


#     # Get Magnetics signals
#     mirnov, mirnov_Bode = bode_Mirnov(shotno,tLim=tLim)

#     # Generate transfer function
#     transfer_mag, transfer_phase, transfer_complex =\
#         make_transfer_function(mirnov_Bode, calib_mag, calib_phase, \
#                                calib_freq,calib_complex, shotno, mirnov)

#     plot_transfer_function(transfer_mag, transfer_phase, transfer_complex,shotno)
#     print('Finsihed')
#######################################################[.05, .997]
def compareBode(shotno=1151208900, doSave='', doPlot=True,\
                 tLim=[0,1], fLim =[0,1e6],bad_sensor=['11_abk'],
                 needs_correction = ['12_abk', '1t_ghk', '3t_ghk'],
                 compareSynthetic=True,calibration_magnitude=6.325,
                 synthDataFileNameInfo={'mesh_file':'C_Mod_ThinCurr_Limiters-homology.h5',
                                        'sensor_set':'C_MOD_ALL',
                                        'calibration_frequency_limits':(10,1e6),
                                        'save_ext_input':''},
                 plot_sensors='all',input_channel=16, ACQ_board=3, B0_normalize=False,
                 freq_domain_calculation=False,save_Ext='_f-sweep',R=6,L=60e-6,C=760e-12,
                 yLim_mag=[0,10],yLim_phase=[60,460],save_Ext_plot=''):
    
    # Get driver signal
    calib, time = __load_calib_signal(shotno, tLim, input_channel, ACQ_board)

     # Get Magnetics signals
    mirnovs, t_inds = __load_mirnov_signals(shotno,tLim)

    # Get empirical transfer function
    transfer_mag, transfer_phase, f, sensors = \
        __gen_Empirical_Transfer_Functions(shotno,calib,needs_correction,fLim,mirnovs,plot_sensors,\
                                           t_inds, B0_normalize)
    

    # Pull synthetic transfer function for comparison
    if compareSynthetic:
        # return __gen_Synthetic_Transfer_Time_Domain(synthDataFileNameInfo,sensors,save_Ext)
        synthetic_transfer_mag,synthetic_transfer_phase,f_synth = \
            __get_Synthetic_Transfer_Function(synthDataFileNameInfo,sensors,\
                        calibration_magnitude,R,L,C, B0_normalize) if freq_domain_calculation else \
            __gen_Synthetic_Transfer_Time_Domain(synthDataFileNameInfo,sensors,save_Ext,R,L,C)
        
    else: synthetic_transfer_mag,synthetic_transfer_phase,f_synth = None,None,None


    # Plot data
    if doPlot:
        plot_transfer_function(transfer_mag, transfer_phase, f, sensors, shotno,doSave,\
                                      bad_sensor, synthetic_transfer_mag,synthetic_transfer_phase,f_synth,
                                      synthDataFileNameInfo, B0_normalize,yLim_mag=yLim_mag,
                                      yLim_phase=yLim_phase,save_ext=save_Ext_plot)
        __plot_ratios(transfer_mag, transfer_phase, f, sensors,  bad_sensor, \
                            synthetic_transfer_mag, synthetic_transfer_phase, f_synth,shotno,\
                            B0_normalize=False, yLim_mag=[0,10], \
                            yLim_phase=[60,460], save_ext='_ratio')
    print('Finished')
    return transfer_mag, transfer_phase, f, sensors, synthetic_transfer_mag, \
        synthetic_transfer_phase,f_synth, calib, time, mirnovs, t_inds








################################################################################################
################################################################################################
def __gen_Empirical_Transfer_Functions(shotno,calib,needs_correction,fLim,mirnovs,\
                                       plot_sensors,t_inds, B0_normalize=False):
    # Generate empirical transfer function using cross spectral density

    fs = 1 / (mirnovs.time[1] - mirnovs.time[0])
    nperseg = 500

    transfer_mag = []
    transfer_phase = []
    sensors = mirnovs.names

    if plot_sensors != 'all':
        sensors = [s for s in sensors if s in plot_sensors]

        if len(sensors) == 0:
            raise ValueError('No valid sensors found in plot_sensors list.')
    for i, name in enumerate(sensors):
        # Cross spectral density between calib and mirnov signal
        f, Pxy = csd(calib, mirnovs.data[mirnovs.names.index(name)][t_inds], fs=fs, scaling='spectrum', nperseg=nperseg, average='mean')
        _, Pxx = csd(calib, calib, fs=fs, scaling='spectrum', nperseg=nperseg, average='mean')

        f_inds = np.argwhere( (f <= fLim[1]) & (f >= fLim[0]) )
        f = f[f_inds]
        Pxy = Pxy[f_inds]
        Pxx = Pxx[f_inds] * ( -1j * f * 1.28e-6 if B0_normalize else 1)

        # Relative amplitude and phase
        mag = np.abs(Pxy / Pxx) 
        phase = np.angle(Pxy / Pxx, deg=True)
        phase =  __manual_Phase_Correction(phase,name,needs_correction,shotno) - (0 if B0_normalize else 0)
        
        transfer_mag.append(mag)
        transfer_phase.append(phase)

    transfer_mag = np.array(transfer_mag)
    transfer_phase = np.array(transfer_phase)

    return transfer_mag, transfer_phase, f, sensors
############################3
def __manual_Phase_Correction(phase,name,needs_correction,shotno):
    
    phase = phase % 360
    # Manually correct phase jumps

    phase_corrected = phase.copy()
    if name[2:] in needs_correction or f'{shotno}'[4] == '3':
        for i in range(1, len(phase)):
            if phase[i] - phase[i-1] < -180:
                phase_corrected[i:] += 360
            elif phase[i] - phase[i-1] > 180:
                phase_corrected[i:] -= 360
        phase_corrected += 360

    # print(f'{shotno}', f'{shotno}'[4] )
    if f'{shotno}'[4] == '3': phase_corrected -= 180 # There's a -1 sign somewhere on the 1150300 series

    return phase_corrected








#####################################################################
#####################################################################3
def __get_Synthetic_Transfer_Function(synthDataFileNameInfo,mirnov_names,calibration_magnitude,R,L,C,\
                                       B0_normalize=False):
        # Load the Mirnov Bode data
    fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz%s.npz'%\
        (synthDataFileNameInfo['mesh_file'],synthDataFileNameInfo['sensor_set'],\
          *synthDataFileNameInfo['calibration_frequency_limits'],\
            synthDataFileNameInfo['save_ext_input'])
    mirnov_Bode = np.load('../data_output/'+fName)
    print('Loaded Synthetic Bode data from ../data_output/%s'%fName)
    probe_signals = mirnov_Bode['probe_signals'] # Comes in as scaled flux: V = -N*A * B
    freqs = mirnov_Bode['freqs']
    probe_signals *= freqs[:,np.newaxis] * 2*np.pi * 1j # convert T/s (divide out A, switch current to voltage source)

    sensor_names = mirnov_Bode['sensor_names']

    # Initialize lists to store transfer function data
    transfer_mag = []
    transfer_phase = []

    # Get RLC circuit correction
    # Generate dictionary for RLC correction based on calibrated R, L values
    RLC_Calib = __prep_RLC_Transfer_Function(C_0=C)
    # mag_RLC, phase_RLC = __prep_RLC_Transfer_Function_Fixed(R=R,L=L,C=C,plot_R=False,R_0=R)

    # Normalize into [T/T]
    if B0_normalize: calibration_magnitude *= freqs * 1.28e-6

    # Loop through each sensor in the set
     # Loop through sensors in Mirnov name ordering to ensure consistency
    for ind, name in enumerate(mirnov_names):
        # Calculate transfer function magnitude and phase

        # Verify that the sensor exists in the probe_signals
        try: # All sensors should exist, but just in case
            sensor_ind = np.char.lower(sensor_names).tolist().index(name)
        except: continue

        # Calibrated R, L system
        # Magnitude signal needs to be in T/s, not just T
        transfer_mag.append(np.abs(probe_signals[:,sensor_ind]) / calibration_magnitude * \
                            RLC_Calib[name.upper()]['mag_RLC'](2*np.pi*freqs))
        transfer_phase.append(np.angle(probe_signals[:,sensor_ind],deg=True) + \
                              RLC_Calib[name.upper()]['phase_RLC'](2*np.pi*freqs)+360)
        
        # # Fixed R, L system
        # transfer_mag.append(np.abs(probe_signals[:,sensor_ind]) / calibration_magnitude * \
        #                     mag_RLC(2*np.pi*freqs))
        # transfer_phase.append(np.angle(probe_signals[:,sensor_ind],deg=True) + \
        #                       phase_RLC(2*np.pi*freqs)+360)

    transfer_mag = np.array(transfer_mag)
    transfer_phase = np.array(transfer_phase)

    return transfer_mag, transfer_phase, freqs



################################################################################################
################################################################################################
def __gen_Synthetic_Transfer_Time_Domain(synthDataFileNameInfo,mirnov_names,save_Ext,R,L,C,doPlot=True):
    
    
    # Generate ThinCurr driver coil signal, identically to simulation
    driver_current, time, synthDataFileNameInfo = __gen_ThinCurr_driver_signal(synthDataFileNameInfo)

    # Load ThinCurr time domain output data 
    signals, time, sensor_names = __load_in_ThinCurr_data(synthDataFileNameInfo,mirnov_names,save_Ext,archiveExt='')

    # Get RLC circuit correction
    mag_RLC, phase_RLC = __prep_RLC_Transfer_Function(R,L,C)

    # Spectral parameters
    fs = 1 / (time[1] - time[0])
    nperseg = 100

    # return driver_current, time, signals, sensor_names

    # transfer_mag, transfer_phase, name, f = __gen_ThinCurr_Transfer_CDS(mirnov_names,\
    #              driver_current,signals,fs,nperseg,sensor_names,mag_RLC, phase_RLC)
    # return __gen_ThinCurr_Transfer_Hilbert(mirnov_names,\
    # # #                  driver_current,signals,fs,nperseg,sensor_names,mag_RLC, phase_RLC,time)
    transfer_mag, transfer_phase, name, f =  __gen_ThinCurr_Transfer_Hilbert(mirnov_names,\
                     driver_current,signals,fs,nperseg,sensor_names,mag_RLC, phase_RLC,time)
    if doPlot:
        plt.close('Time_Domain_Synthetic_Mirnov')
        fig,ax = plt.subplots(2,1,num='Time_Domain_Synthetic_Mirnov', figsize=(4,3), tight_layout=True)
        ax[0].plot(time, driver_current)
        ax[0].set_ylabel('Driver Current [A]')
        ax[1].set_xlabel('Time [s]')
        ax[1].plot(time, signals[sensor_names.index(mirnov_names[0])][:])
        ax[1].set_ylabel(f'{mirnov_names[0]} Signal [T/s]')
        for i in range(2): ax[i].grid()
        plt.show()
        plt.savefig('../output_plots/time_domain_synthetic_mirnov%s.pdf'%save_Ext,transparent=True)

        
        trim_signal = 1000#int(len(driver_current)*trim_percent)
        trim_inds = np.arange(trim_signal,len(driver_current)-trim_signal,dtype=int)
        
        nperseg, noverlap, win, detrend = 100, 40, 'hann', 'linear'
        fs = 1/(time[1]-time[0])
        x = driver_current[trim_inds]
        y = signals[sensor_names.index(name)][trim_inds]
        N = len(x)

        SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap,
            scale_to='magnitude', fft_mode='onesided2X',
            phase_shift=None)

        Sxy1 = SFT.spectrogram(y, x, detr=detrend, k_offset=nperseg//2,
                p0=0, p1=(N-noverlap) // SFT.hop)

        plt.close('Cross Spectrogram')
        plt.figure(num='Cross Spectrogram');
        plt.contourf(np.linspace(0,1,Sxy1.shape[1]),SFT.f*1e-6,np.abs(Sxy1),zorder=-5,\
                    levels=np.linspace(0,.1,10),cmap='plasma');
        plt.colorbar(label='Cross Power [T/s * A]')
        plt.xlabel('Time [s]'); plt.ylabel('Frequency [MHz]')
        plt.gca().set_rasterization_zorder(-1)

        plt.show()
    return transfer_mag, transfer_phase, f
##########################################################################
def __gen_ThinCurr_Transfer_Hilbert(mirnov_names,driver_current,signals,fs,nperseg,sensor_names,mag_RLC, phase_RLC,time):
    
    
    transfer_mag = []
    transfer_phase = []
    trim_percent=0.001
    trim_signal = 1000#int(len(driver_current)*trim_percent)
    trim_inds = np.arange(trim_signal,len(driver_current)-trim_signal,dtype=int)
    print(trim_inds,trim_inds.shape)
    print()
    dt = time[1] - time[0]
    # Loop through desired mirnov sensors, extract PSD result
    for i, name in enumerate(mirnov_names):
        print(signals[sensor_names.index(name)].shape)
        sig = np.array(signals[sensor_names.index(name)])
        
        #return sig, trim_signal
        hilbert_mirnov = hilbert(sig)
        hilbert_driver = hilbert(driver_current)
        
        mag_mirnov = np.abs(hilbert_mirnov)[trim_inds][1:]
        mag_driver = np.abs(hilbert_driver)[trim_inds][1:]
        
        phase_mirnov = np.unwrap(np.angle(hilbert_mirnov,deg=True),period=360)[trim_inds]
        phase_driver = np.unwrap(np.angle(hilbert_driver,deg=True),period=360)[trim_inds]
        
        
        freq = np.diff(phase_driver) / ( 360 * dt)
        
        mag = mag_mirnov/mag_driver * mag_RLC(2*np.pi*freq)
        phase = (phase_mirnov-phase_driver)[1:] +  phase_RLC(2*np.pi*freq)
        
        # freq = np.diff(phase) / (2.0 * np.pi * dt)
        
        transfer_mag.append(mag)
        transfer_phase.append(phase)

    transfer_mag = np.array(transfer_mag)
    transfer_phase = np.array(transfer_phase)
    
    
    return transfer_mag, transfer_phase, name, freq
    
def __gen_ThinCurr_Transfer_CDS(mirnov_names,driver_current,signals,fs,nperseg,sensor_names,mag_RLC, phase_RLC):
    
    transfer_mag = []
    transfer_phase = []
    trim_signal = 1000#int(len(driver_current)*trim_percent)
    trim_inds = np.arange(trim_signal,len(driver_current)-trim_signal,dtype=int)
    fLim=[0,50e3]
    # Loop through desired mirnov sensors, extract PSD result
    for i, name in enumerate(mirnov_names):
        # Cross spectral density between calib and mirnov signal
        f, Pxy = csd(driver_current[trim_inds], signals[sensor_names.index(name)][trim_inds], fs=fs, scaling='spectrum', nperseg=nperseg, average='mean')
        _, Pxx = csd(driver_current[trim_inds], driver_current[trim_inds], fs=fs, scaling='spectrum', nperseg=nperseg, average='mean')

        f_inds = np.argwhere( (f <= fLim[1]) & (f >= fLim[0]) ).squeeze()
        f = f[f_inds]
        Pxy = Pxy[f_inds]
        Pxx = Pxx[f_inds] * ( -1j * f * 1.28e-6 if B0_normalize else 1)

        # Relative amplitude and phase
        mag = np.abs(Pxy / Pxx) * mag_RLC(2*np.pi*f)
        phase = np.angle(Pxy / Pxx, deg=True) + phase_RLC(2*np.pi*f)
        #phase =  __manual_Phase_Correction(phase,name,needs_correction,shotno) - (0 if B0_normalize else 0)
        
        transfer_mag.append(mag)
        transfer_phase.append(phase)

    transfer_mag = np.array(transfer_mag)
    transfer_phase = np.array(transfer_phase)
    
    return transfer_mag, transfer_phase, name, f
######################################################################
def __gen_ThinCurr_driver_signal(synthDataFileNameInfo):
    time = np.linspace(0,synthDataFileNameInfo['T'],int(synthDataFileNameInfo['T']/synthDataFileNameInfo['dt']))
    # Frequency sweep
    periods = 1
    dead_fraction = 0.0
    f_mod = lambda t: 100e3 + 2e3*t
    I_mod = lambda t: synthDataFileNameInfo['I']*np.ones_like(t)
    f_out_driver, I_out_driver, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod)
    driver_current = I_out_driver * np.cos(f_out_driver)

    synthDataFileNameInfo['f'] = f_out_driver
    synthDataFileNameInfo['I'] = I_out_driver

    return driver_current, time, synthDataFileNameInfo
######################################################################
def __load_in_ThinCurr_data(synthDataFileNameInfo=None, sensor_name=None, save_Ext='', archiveExt=''):
        # Synthetic dataget_data
    filename = __gen_filename(synthDataFileNameInfo,synthDataFileNameInfo['sensor_set'],\
                              synthDataFileNameInfo['mesh_file'],\
                              save_Ext=save_Ext,archiveExt=archiveExt)
    hist = histfile(filename+'.hist')
    time = hist['time'][:-1]
    sensor_name = list(hist.keys())[1:] if sensor_name is None else sensor_name
    
    # Extract signals from hist
    signals = np.array([hist[name.upper()] for name in sensor_name])
    signals = np.diff(signals,axis=1)/(time[1]-time[0]) # Remove DC offset, convert to dB/s

    return signals, time, sensor_name

##########################################################################
# File name generator for synthetic data loading
def __gen_filename(param,sensor_set,mesh_file,save_Ext='',archiveExt=''):
    f=param['f'];m=param['m'];
    n=param['n']

    # Rename output 
    if type(f) is float: f_out = '%d'%f*1e-3 
    else: f_out = 'custom'
    if type(m) is not list: mn_out = '%d-%d'%(m,n)
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])
    f_save = '../data_output/%sfloops_filament_%s_m-n_%s_f_%s_%s%s'%\
                    (archiveExt,sensor_set,mn_out,f_out,mesh_file,save_Ext)

    
    return f_save
        

################################################################################################
def __prep_RLC_Transfer_Function_Fixed(R = 6, L = 60e-6, C = 760e-12, plot_R=False,R_0=6):

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
    mag_RLC =  lambda w: 1#np.abs(H(w))
    phase_RLC =  lambda w: 0#np.angle(H(w), deg=True)

    if plot_R: 
        plt.close('R_Function')
        fig,ax = plt.subplots(1,1,num='R_Function', figsize=(6,4), tight_layout=True)
        f=np.linspace(10,1e6,100)
        ax.plot(f*1e-6,Z_R(2*np.pi*f),label='$|R|$, left',color='C0')
        ax1= ax.twinx()
        ax1.plot(f*1e-6,mag_RLC(2*np.pi*f),label='$|H|$, right',color='C1')
        ax.grid()
        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Resistance [Ohms]')
        ax1.set_ylabel('Transfer RLC [$V_{out}/V_{in}$]')
        fig.savefig('../output_plots/R_function.pdf',dpi=300,transparent=True)
        ax.legend(loc='upper left',fontsize=8)
        ax1.legend(loc='upper right',fontsize=8)
        plt.show()
    return mag_RLC, phase_RLC






################33

################################################################################################3
################################################################################################
def plot_transfer_function(transfer_mag, transfer_phase, f, sensors, shotno, doSave, bad_sensor, \
                           transfer_mag_synth=None, transfer_phase_synth=None, f_synth=None,
                           synthDataFileNameInfo=None, B0_normalize=False, yLim_mag=[0,10], \
                            yLim_phase=[60,460], save_ext='',phase_normalization=False):
    # Plot transfer function

    fig_mag, ax_mag, fig_phase, ax_phase, sensors = __prep_plots(shotno,save_ext,sensors)


    # Temporarily test: normalize phase to first sensor
    if phase_normalization:
        for i in np.arange(1,len(sensors)):
            transfer_phase[i] -=  transfer_phase[0]
            transfer_phase_synth[i] -=  transfer_phase_synth[0] 


    # Plot individual transfer functions
    __plot_individual_transfer_function(transfer_mag, transfer_phase, f, sensors, ax_mag, ax_phase, bad_sensor, \
                            transfer_mag_synth=transfer_mag_synth, transfer_phase_synth=transfer_phase_synth,\
                            f_synth=f_synth,
                            B0_normalize=B0_normalize, yLim_mag=yLim_mag, \
                            yLim_phase=yLim_phase, save_ext=save_ext,break_plot = ['bp28_ghk','bp6t_ghk'])

    # Save plots
    if doSave:
        synth=f'_{synthDataFileNameInfo["mesh_file"].replace(".h5","")}_{synthDataFileNameInfo["sensor_set"]}_{save_ext}'\
            if synthDataFileNameInfo is not None else ''
        synth += '_B0norm' if B0_normalize else ''
        if len(sensors) == 1:
            fig_mag.savefig(f'{doSave}_Cmod_Transfer_Function_CSD_{shotno}_{sensors[0]}{synth}.pdf', dpi=300, transparent=True)
        else:
            fig_mag.savefig(f'{doSave}_Cmod_Transfer_Magnitude_CSD_{shotno}{synth}.pdf', dpi=300, transparent=True)
            fig_phase.savefig(f'{doSave}_Cmod_Transfer_Phase_CSD_{shotno}{synth}.pdf', dpi=300, transparent=True)

    plt.show()

#################################################################
def __load_in_RLC_Spline_Fit(filename='Empirical_Bode_Fits_RLC.npz'):
    # Load RLC spline fit data
    fit_data = np.load('../C-Mod/'+filename)
    spline_fit_mag = BSpline(t=fit_data['t_mag'],c=fit_data['c_mag'],k=fit_data['k_mag'],axis=0)
    spline_fit_phase = BSpline(t=fit_data['t_phase'],c=fit_data['c_phase'],k=fit_data['k_phase'],axis=0)
    return spline_fit_mag, spline_fit_phase, fit_data['sensor_names']

#3##################################################################
def __plot_ratios(transfer_mag, transfer_phase, f, sensors,  bad_sensor, \
                            transfer_mag_synth, transfer_phase_synth, f_synth,shotno,\
                            B0_normalize=False, yLim_mag=[0,10], \
                            yLim_phase=[60,460], save_ext=''):
    
    # Gen plot of ratio between empirical and synthetic transfer functions
    fig_mag, ax_mag, fig_phase, ax_phase, sensors = __prep_plots(shotno,save_ext,sensors)

    f_synth_inds = np.array([np.argmin(np.abs(f - f_synth[i])) for i in range(len(f_synth))])
    ratio_mag = transfer_mag[:,f_synth_inds,0] / transfer_mag_synth
    ratio_phase = transfer_phase[:,f_synth_inds,0] - transfer_phase_synth

    # Filter: for signals below 40kHz, set phase ratio to zero, and mag ratio to and interpolation of subsequent points
    inds_replace = np.argwhere(f[f_synth_inds,0] < 40e3).squeeze()
    
    ratio_phase_cleaned = ratio_phase.copy()
    ratio_mag_cleaned = ratio_mag.copy()    
    ratio_phase_cleaned[:,inds_replace] = 0

    mag_interp =np.arange(1,5)+(inds_replace[-1] if np.size(inds_replace)>1 else inds_replace)
    for i in range(ratio_mag.shape[0]):
        ratio_mag_cleaned[i,inds_replace] = np.polyval(np.polyfit(
                                    f[f_synth_inds][mag_interp].squeeze(),\
                                    ratio_mag[i,mag_interp],1),f[f_synth_inds][inds_replace])
  
    # Replace some sensors with swapouts
    bad_sensor_swap = {'bp05_abk':'bp06_abk','bp08_ghk':'bp07_ghk'}

    # Run fits on ratio data here if desired
    spline_fit_mag = make_smoothing_spline(f[f_synth_inds].squeeze(),ratio_mag_cleaned,axis=1)
    spline_fit_phase = make_smoothing_spline(f[f_synth_inds].squeeze(), ratio_phase_cleaned,axis=1)

    ratio_mag_cleaned = spline_fit_mag(f[f_synth_inds])
    ratio_phase_cleaned = spline_fit_phase(f[f_synth_inds])


     # Plot individual transfer functions
    __plot_individual_transfer_function(ratio_mag, ratio_phase, f[f_synth_inds], sensors, ax_mag, ax_phase, bad_sensor, \
                            transfer_mag_synth=ratio_mag_cleaned, transfer_phase_synth=ratio_phase_cleaned,\
                            f_synth=f_synth,
                            B0_normalize=B0_normalize, yLim_mag=yLim_mag, \
                            yLim_phase=yLim_phase, save_ext=save_ext,break_plot = ['bp28_ghk','bp6t_ghk'])



     # Save plots
    if doSave:
        # Save spline fit
        if np.size(sensors)>1: # Only save if using all the sensors
            np.savez('Empirical_Bode_Fits_RLC.npz',t_mag=spline_fit_mag.t,c_mag=spline_fit_mag.c,k_mag=spline_fit_mag.k,t_phase=spline_fit_phase.t,\
                 c_phase=spline_fit_phase.c,k_phase=spline_fit_phase.k,sensor_names=sensors)
        
        synth=f'_{synthDataFileNameInfo["mesh_file"].replace(".h5","")}_{synthDataFileNameInfo["sensor_set"]}_{save_ext}'\
            if synthDataFileNameInfo is not None else ''
        synth += '_B0norm' if B0_normalize else ''
        if len(sensors) == 1:
            fig_mag.savefig(f'{doSave}_Cmod_Transfer_Function_CSD_{shotno}_{sensors[0]}{synth}.pdf', dpi=300, transparent=True)
        else:
            fig_mag.savefig(f'{doSave}_Cmod_Transfer_Magnitude_CSD_{shotno}{synth}.pdf', dpi=300, transparent=True)
            fig_phase.savefig(f'{doSave}_Cmod_Transfer_Phase_CSD_{shotno}{synth}.pdf', dpi=300, transparent=True)

    plt.show()
#################################################################
def __plot_individual_transfer_function(transfer_mag, transfer_phase, f, sensors, ax_mag, ax_phase, bad_sensor, \
                            transfer_mag_synth=None, transfer_phase_synth=None, f_synth=None,
                            B0_normalize=False, yLim_mag=[0,10], \
                            yLim_phase=[60,460], save_ext='',break_plot = ['bp28_ghk','bp6t_ghk']):
    
    # Loop through each sensor
    index_plot =0
    for i, name in enumerate(sensors):
        if name[2:] in bad_sensor: continue


        ax_mag[index_plot].plot(f * 1e-6, transfer_mag[i],\
                        label=f'{name}'+(' Magnitude' if len(sensors)==1 else ''))
        
        ax_phase[index_plot].plot(f * 1e-6, transfer_phase[i], color='C1' if len(sensors)==1 else 'C0',\
                          label=f'{name}'+(' Phase' if len(sensors)==1 else ''), \
                            alpha=.6 if len(sensors)==1 else 1)
        
        # Plot synthetic transfer function for comparison
        if transfer_mag_synth is not None:
            
            ax_mag[index_plot].plot(f_synth * 1e-6, transfer_mag_synth[i], ls='dashdot', color='C0' if len(sensors)==1 else 'C1',\
                            label='Synthetic + Circuit' if i==0 else '',
                            alpha=.6)
            ax_phase[index_plot].plot(f_synth* 1e-6, transfer_phase_synth[i], ls='dashdot', color='C1',\
                                 label='Synthetic Signal' if i == 0 and len(sensors) != 1 else '', \
                                    alpha=.4 if len(sensors)==1 else .6)

        __clean_up_plot(ax_mag,ax_phase,sensors,i,yLim_mag,yLim_phase,B0_normalize,index_plot)

        if name in break_plot: index_plot += 7 - index_plot%7
        else:index_plot += 1
    if len(sensors) > 1 and np.size(yLim_mag[0])>1:
        for i in range (4): 
            for j in range(7): 
                ax_mag[i*7+j].set_ylim(yLim_mag[0])
                if j != 0: ax_mag[i*7+j].set_yticklabels([])
        for i in range (4,6):
            for j in range(7): 
                ax_mag[i*7+j].set_ylim(yLim_mag[1])
                if j != 0: ax_mag[i*7+j].set_yticklabels([])
        for i in [6]:
            for j in range(7): 
                ax_mag[i*7+j].set_ylim(yLim_mag[2])
                if j != 0: ax_mag[i*7+j].set_yticklabels([])
    else: 
        ax_mag[0].set_ylim(yLim_mag[0])
        ax_phase[0].set_ylim(yLim_phase)
###################################################################
def __prep_plots(shotno,save_ext,sensors):

    if len(sensors) == 1:
        plt.close('Transfer_Function_%d_%s%s'%(shotno,sensors[0],save_ext))
        fig_mag,ax_mag = plt.subplots(1,1, num='Transfer_Function_%d_%s%s'%(shotno,sensors[0],save_ext), tight_layout=True, sharex=True,sharey=True, figsize=(6,4))
        ax_phase=ax_mag.twinx()
        fig_phase = fig_mag
    else:
        plt.close('Transfer_Function_Magnitude_%d%s' % (shotno,save_ext))
        plt.close('Transfer_Function_Phase_%d%s' % (shotno,save_ext))
        fig_mag, ax_mag = plt.subplots(7,7, num='Transfer_Function_Magnitude_%d%s' % (shotno,save_ext), tight_layout=True, sharex=True,sharey=False, figsize=(9,8))
        fig_phase, ax_phase = plt.subplots(7,7, num='Transfer_Function_Phase_%d%s' % (shotno,save_ext), tight_layout=True, sharex=True,sharey=True, figsize=(9,8))
        ax_mag = ax_mag.flatten()
        ax_phase = ax_phase.flatten()
        fig_mag.delaxes(ax_mag[-2]);fig_mag.delaxes(ax_mag[-1])
        fig_phase.delaxes(ax_phase[-2]);fig_phase.delaxes(ax_phase[-1])
    if len(sensors) == 1:
        ax_mag = [ax_mag]
        ax_phase = [ax_phase]
    
    # Sort sensors according to the pattern: BPXX_*, BPXT_*, then BP_XX_Top/Bot
    def sort_key(name):
        if name.startswith('bp') and len(name) > 4 and name[2:4].isdigit() and name[4] == '_':
            # BPXX_* : e.g., bp01_abk -> category 0, number 1
            return (0, int(name[2:4]), name)
        elif name.startswith('bp') and len(name) > 4 and name[2].isdigit() and name[3] == 't' and name[4] == '_':
            # BPXT_* : e.g., bp1t_abk -> category 1, number 1
            return (1, int(name[2]), name)
        else:
            # BP_XX_Top/Bot or others : e.g., bp_ef_top -> category 2, full name
            return (2, name)
    
    sensors = sorted(sensors, key=sort_key)

    return fig_mag, ax_mag, fig_phase, ax_phase, sensors
######################################
def __clean_up_plot(ax_mag,ax_phase,sensors,i,yLim_mag,yLim_phase,B0_normalize,index_plot):
    ax_mag[index_plot].grid()
    if np.size(yLim_mag[0])==1:ax_mag[index_plot].set_ylim(yLim_mag)
    ax_phase[index_plot].set_ylim(yLim_phase)
    if len(sensors) != 1: ax_phase[i].grid()
    if len(sensors) == 1:
        ax_mag[index_plot].legend(fontsize=8, loc='upper left',handlelength=2.5)
        ax_phase[index_plot].legend(fontsize=8, loc='upper right')
        ax_mag[index_plot].set_ylabel('Transfer Magnitude ' +'[T/T]' if B0_normalize else '[T/s/A]')
        ax_phase[index_plot].set_ylabel('Transfer Phase [deg]')
        ax_mag[index_plot].set_xlabel('Frequency [MHz]')
    else:
        ax_mag[index_plot].legend(fontsize=8, loc='upper right',handlelength=0.5)
        ax_phase[index_plot].legend(fontsize=8, loc='upper right',handlelength=0.5)
        ax_mag[index_plot].set_xticks([0,.5,1])
        ax_phase[index_plot].set_xticks([0,.5,1])
        if index_plot == 7*3: 
            ax_mag[index_plot].set_ylabel(r'$||H(\omega)||$ [T/s/A]')
            ax_phase[index_plot].set_ylabel(r'$\angle H(\omega)$ [deg]')
        if index_plot == 7*6+3 : 
            ax_mag[index_plot].set_xlabel('Frequency [MHz]')
            ax_phase[index_plot].set_xlabel('Frequency [MHz]')
            

# ###################################################################
###################################
# Load calibration signal
def __load_calib_signal(shot, tLim=[0,1],input_channel=16, ACQ_board=3):
        conn=gC.openTree(shot,treeName='MAGNETICS')
        time=conn.get(f'dim_of(\\MAGNETICS::TOP.ACTIVE_MHD.DATA_ACQ.CPCI.ACQ_216_{ACQ_board}.INPUT_{input_channel:02d})').data()
      
        calib=conn.get(f'\\MAGNETICS::TOP.ACTIVE_MHD.DATA_ACQ.CPCI.ACQ_216_{ACQ_board}.INPUT_{input_channel:02d}').data()
        # Trim signal
        t_inds = np.where((time >= tLim[0]) & (time <= tLim[1]))[0]
        time = time[t_inds]
        calib = calib[t_inds]* 10 
        return calib, time

###################################
def __load_mirnov_signals(shotno, tLim=[0,1]):
    # return mirnovs and time trim inds, including reloading if needed
    mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,2]})['bp_k']
    if mirnovs.time[0] > 1e-5+tLim[0] or mirnovs.time[-1] < 1e-5+tLim[1]:
        print(mirnovs.time[0],mirnovs.time[-1])
        mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,2]},forceReload=['bp_k'])['bp_k']
    t_inds = np.where((mirnovs.time >= tLim[0]) & (mirnovs.time <= tLim[1]))[0]

    return mirnovs, t_inds
###################################3###################################

################################################################################3
def sandbox_(shotno,tLim=[0,1],input_channel=16,ACQ_board=3,mirnov_channel='bp_ef_top'):
    # Test cross-correlation method

    # Load driver signal
    calib,time = __load_calib_signal(shotno,tLim,input_channel,ACQ_board)

    # Load Mirnov signals

    mirnovs, t_inds = __load_mirnov_signals(shotno,tLim)

    mirnov_ind = mirnovs.names.index(mirnov_channel)

    from scipy import signal
    # Calculate cross spectral density

    plt.close('Raw Input Signals')
    fig,ax = plt.subplots(2,1,sharex=True,num='Raw Input Signals',tight_layout=True)
    ax[0].plot(time[::100],calib[::100],label='Calibration Signal')
    ax[0].set_ylabel('Calibration Signal [A]')
    ax[0].legend(fontsize=8)
    ax[0].grid()
    ax[1].plot(mirnovs.time[t_inds][::100],mirnovs.data[mirnov_ind][t_inds][::100],label=mirnovs.names[mirnov_ind])
    ax[1].set_ylabel(r'Mirnov Signal $\frac{d}{dt}B_\theta$ [T/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].legend(fontsize=8)
    ax[1].grid()
    plt.show()
    fig.savefig('../output_plots/Cmod_Input_Signals_%d.pdf'%shotno,dpi=300,transparent=True)

    
    # Cross spectral density between calib and mirnov signal
    #  and auto spectral density of calib signal
    #  to get transfer function
    #  H = Pxy / Pxx
    nperseg, noverlap, win, detrend = 3000, 1000, 'hann', 'linear'

    f1, Pxx = csd(calib,calib,fs=1/(mirnovs.time[1]-mirnovs.time[0]),scaling='spectrum',
                         nperseg=nperseg,noverlap=noverlap,average='mean',detrend=detrend)

    f, Pxy = csd(calib,mirnovs.data[0][t_inds],fs=1/(mirnovs.time[1]-mirnovs.time[0]),
                        scaling='spectrum',nperseg=nperseg,noverlap=noverlap,average='mean',detrend=detrend)

    phase = np.angle(Pxy/Pxx,deg=True)
    phase[phase < -100] += 360

    plt.close('Transfer Function from CSD')
    fig,ax = plt.subplots(2,1,figsize=(6,4),sharex=True,num='Transfer Function from CSD',tight_layout=True)
    ax[0].plot(f*1e-6, np.abs(Pxy/Pxx)/3,label=mirnovs.names[0])
    ax[0].set_ylabel('Transfer Magnitude [T/s/A]')
    ax[0].legend(fontsize=8,loc='upper right')
    ax[0].grid()
    ax[1].plot(f*1e-6, phase)
    ax[1].set_ylabel('Transfer Phase [deg]')
    ax[1].set_xlabel('Frequency [MHz]')
    ax[1].grid()
    plt.show()
    

    fig.savefig('../output_plots/Cmod_Transfer_Function_CSD_%d.pdf'%shotno,dpi=300,transparent=True)
    # Build spectrogram
    from scipy import signal
    
    fs = 1/(time[1]-time[0])
    x = calib
    y = mirnovs.data[mirnov_ind][t_inds]
    N = len(x)

    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap,
          scale_to='magnitude', fft_mode='onesided2X',
           phase_shift=None)

    Sxy1 = SFT.spectrogram(y, x, detr=detrend, k_offset=nperseg//2,
            p0=0, p1=(N-noverlap) // SFT.hop)

    plt.close('Cross Spectrogram')
    plt.figure(num='Cross Spectrogram');
    plt.contourf(np.linspace(0,1,Sxy1.shape[1]),SFT.f*1e-6,np.abs(Sxy1),zorder=-5,\
                 levels=np.linspace(0,1,10),cmap='plasma');
    plt.colorbar(label='Cross Power [T/s * A]')
    plt.xlabel('Time [s]'); plt.ylabel('Frequency [MHz]')
    plt.gca().set_rasterization_zorder(-1)

    plt.show()
    # plt.savefig('../output_plots/Cmod_Cross_Spectrogram_%d.pdf'%shotno,dpi=300)

    print('Done')


def sandbox(sensor_name='BP2T_ABK'):
    hist = histfile('../data_output/floops_surface_C_MOD_LIM_m-n_14-11_f_7_FAR3D_NonLinear.hist')
    time = hist['time'][:-1]
    sensor_name = list(hist.keys())[1:] if sensor_name is None else sensor_name
    
    # Extract signals from hist
    signals = np.array([hist[name.upper()] for name in sensor_name])
    signals = np.diff(signals,axis=1)/(time[1]-time[0]) # Remove DC offset, convert to dB/s

    print('Done')
#######################################################################################
if __name__ == '__main__':
    # __prep_RLC_Transfer_Function()
    # sandbox(1150319903,tLim=[.5,1.65],input_channel=16,ACQ_board=2)

    # sandbox()
    # mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    sensor_set = 'C_MOD_ALL' # Example sensor names
    calibration_magnitude = 6.325 # Replace with actual calibration magnitude
    calibration_frequency_limits = (10, 1e6)  # Frequency range in Hz
    comparison_shot = 1150319902 # Example shot number for comparison
    synthDataFileNameInfo={'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
                                        'sensor_set':'C_MOD_ALL',
                                        'calibration_frequency_limits':(10,1e6),
                                        'I':4.5,'T':2e-2,'dt':1e-6,'m':[1],'n':[1],
                                        'save_ext_input':'_f-sweep_All-Mirnovs-Corrected-3D_Tiles'*True}
    plot_sensors =  'all'#['bp01_abk']
    needs_correction = ['12_abk', '1t_ghk', '3t_ghk', '_ef_bot', '_ef_top']
    input_channel = 16 # Channel for calibration signal
    ACQ_board = 2 # ACQ board for calibration signal
    tLim = [.5,1.65]
    B0_normalize = False
    compareSynthetic = True
    fLim = [0,1e6]
    yLim_mag = [[0,5],[0,4],[0,24]]#[0,21]
    yLim_phase = [-10,340]
    freq_domain_calculation = True # If True, use synthetic frequency domain data, otherwise time domain
    doSave='../output_plots/'*True
    save_Ext_plot='_newCoords_Tiles_Mirnov_Shields_3D_Norm_Phase'*True

    R = 5.3*0+0     # Ohms 
    L = 60e-6      # Henry
    C = 1.35e-9#780e-12     # Farads

    out = compareBode(shotno=comparison_shot,doSave=doSave,doPlot=True,
                synthDataFileNameInfo=synthDataFileNameInfo,plot_sensors=plot_sensors,
                input_channel=input_channel,tLim=tLim, ACQ_board=ACQ_board,
                needs_correction=needs_correction, B0_normalize=B0_normalize,
                compareSynthetic=compareSynthetic,fLim=fLim, R=R, L=L, C=C,
                freq_domain_calculation=freq_domain_calculation,yLim_mag=yLim_mag,
                yLim_phase=yLim_phase,save_Ext_plot=save_Ext_plot,) 
    
    transfer_mag, transfer_phase, f, sensors, synthetic_transfer_mag, \
        synthetic_transfer_phase,f_synth, calib, time, mirnovs, t_inds = out
    
    print('Done')
signal_spectrogram_C_Mod