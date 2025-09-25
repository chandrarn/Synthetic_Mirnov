# Emperical Bode Calibration

import get_Cmod_Data as gC
from header_Cmod import np, plt,sys,ShortTimeFFT, csd, hilbert
sys.path.append('../signal_generation/')
from header_signal_generation import gen_coupled_freq, histfile


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
    if doPlot: plot_transfer_function(transfer_mag, transfer_phase, f, sensors, shotno,doSave,\
                                      bad_sensor, synthetic_transfer_mag,synthetic_transfer_phase,f_synth,
                                      synthDataFileNameInfo, B0_normalize,yLim_mag=yLim_mag,
                                      yLim_phase=yLim_phase,save_ext=save_Ext_plot)

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
    mag_RLC, phase_RLC = __prep_RLC_Transfer_Function(R,L,C)

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

        # Magnitude signal needs to be in T/s, not just T
        transfer_mag.append(np.abs(probe_signals[:,sensor_ind]) / calibration_magnitude * mag_RLC(2*np.pi*freqs))
        transfer_phase.append(np.angle(probe_signals[:,sensor_ind],deg=True) + phase_RLC(2*np.pi*freqs)+360)

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
def __prep_RLC_Transfer_Function(R = 6, L = 60e-6, C = 760e-12, plot_R=False,R_0=0.7):

    # RLC Circuit transfer function:
    # Transfer function H(w) for series RLC (output across the capacitor)
    # RLC circuit parameters (eventually, correct for individual sensors)
    # R = 28     # Ohms 
    # L = 60e-6      # Henry
    # C = 760e-12#780e-12     # Farads

    def Z_R(w):
        out = R * np.ones_like(w)
        # Final factor is sqrt[resistivity/permeability]
        out[w/(2*np.pi)>=220e3] *= 1+(np.sqrt( w[w/(2*np.pi)>=220e3]/(2*np.pi))-np.sqrt(220e3))*0.0043 # Skin depth correction
        #out += R # 
        return out
    Z_L =  lambda w: (1j * w * L)
    Z_C =  lambda w:1 / (1j * w * C)
    Z_total =  lambda w: Z_R(w) + Z_L(w) + Z_C(w)

    H =  lambda w: Z_C(w) / Z_total(w)  # Voltage across capacitor / input voltage

    # Magnitude and phase response
    mag_RLC =  lambda w: np.abs(H(w))
    phase_RLC =  lambda w: np.angle(H(w), deg=True)

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








################################################################################################3
################################################################################################
def plot_transfer_function(transfer_mag, transfer_phase, f, sensors, shotno,doSave,bad_sensor,\
                           transfer_mag_synth=None,transfer_phase_synth=None,f_synth=None,
                           synthDataFileNameInfo=None,B0_normalize=False,yLim_mag=[0,10],\
                            yLim_phase=[60,460],save_ext=''):
    # Plot transfer function



    if len(sensors) == 1:
        plt.close('Transfer_Function_%d_%s%s'%(shotno,sensors[0],save_ext))
        fig_mag,ax_mag = plt.subplots(1,1, num='Transfer_Function_%d_%s%s'%(shotno,sensors[0],save_ext), tight_layout=True, sharex=True,sharey=True, figsize=(6,4))
        ax_phase=ax_mag.twinx()
        fig_phase = fig_mag
    else:
        plt.close('Transfer_Function_Magnitude_%d%s' % (shotno,save_ext))
        plt.close('Transfer_Function_Phase_%d%s' % (shotno,save_ext))
        fig_mag, ax_mag = plt.subplots(7,7, num='Transfer_Function_Magnitude_%d%s' % (shotno,save_ext), tight_layout=True, sharex=True,sharey=True, figsize=(9,8))
        fig_phase, ax_phase = plt.subplots(7,7, num='Transfer_Function_Phase_%d%s' % (shotno,save_ext), tight_layout=True, sharex=True,sharey=True, figsize=(9,8))
        ax_mag = ax_mag.flatten()
        ax_phase = ax_phase.flatten()
        fig_mag.delaxes(ax_mag[-2]);fig_mag.delaxes(ax_mag[-1])
        fig_phase.delaxes(ax_phase[-2]);fig_phase.delaxes(ax_phase[-1])
    if len(sensors) == 1:
        ax_mag = [ax_mag]
        ax_phase = [ax_phase]
    
    # Loop through each sensor
    for i, name in enumerate(sensors):
        if name[2:] in bad_sensor: continue

        ax_mag[i].plot(f * 1e-6, transfer_mag[i],\
                        label=f'{name}'+(' Magnitude' if len(sensors)==1 else ''))
        
        ax_phase[i].plot(f * 1e-6, transfer_phase[i], color='C1' if len(sensors)==1 else 'C0',\
                          label=f'{name}'+(' Phase' if len(sensors)==1 else ''), \
                            alpha=.6 if len(sensors)==1 else 1)
        
        # Plot synthetic transfer function for comparison
        if transfer_mag_synth is not None:
            
            ax_mag[i].plot(f_synth * 1e-6, transfer_mag_synth[i], '--', color='C0' if len(sensors)==1 else 'C1',\
                            label='Synth.' if i==0 else '',
                            alpha=.6)
            ax_phase[i].plot(f_synth* 1e-6, transfer_phase_synth[i], '--', color='C1',\
                                 label='Synth.' if i == 0 and len(sensors) != 1 else '', \
                                    alpha=.4 if len(sensors)==1 else .6)

        __clean_up_plot(ax_mag,ax_phase,sensors,i,yLim_mag,yLim_phase,B0_normalize)

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

######################################
def __clean_up_plot(ax_mag,ax_phase,sensors,i,yLim_mag,yLim_phase,B0_normalize):
    ax_mag[i].grid()
    ax_mag[i].set_ylim(yLim_mag)
    ax_phase[i].set_ylim(yLim_phase)
    if len(sensors) != 1: ax_phase[i].grid()
    if len(sensors) == 1:
        ax_mag[i].legend(fontsize=8, loc='upper left')
        ax_phase[i].legend(fontsize=8, loc='upper right')
        ax_mag[i].set_ylabel('Transfer Magnitude ' +'[T/T]' if B0_normalize else '[T/s/A]')
        ax_phase[i].set_ylabel('Transfer Phase [deg]')
        ax_mag[i].set_xlabel('Frequency [MHz]')
    else:
        ax_mag[i].legend(fontsize=8, loc='upper right',handlelength=0.5)
        ax_phase[i].legend(fontsize=8, loc='upper right',handlelength=0.5)
        ax_mag[i].set_xticks([0,.5,1])
        ax_phase[i].set_xticks([0,.5,1])
        if i == 7*3: 
            ax_mag[i].set_ylabel(r'$||H(\omega)||$ [T/s/A]')
            ax_phase[i].set_ylabel(r'$\angle H(\omega)$ [deg]')
        if i == 7*6+3 : 
            ax_mag[i].set_xlabel('Frequency [MHz]')
            ax_phase[i].set_xlabel('Frequency [MHz]')
            

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
def sandbox(shotno,tLim=[0,1],input_channel=16,ACQ_board=3,mirnov_channel='bp_ef_top'):
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
#######################################################################################
if __name__ == '__main__':
    # sandbox(1150319903,tLim=[.5,1.65],input_channel=16,ACQ_board=2)

    # mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    sensor_set = 'C_MOD_ALL' # Example sensor names
    calibration_magnitude = 6.325 # Replace with actual calibration magnitude
    calibration_frequency_limits = (10, 1e6)  # Frequency range in Hz
    comparison_shot = 1150319902 # Example shot number for comparison
    synthDataFileNameInfo={'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
                                        'sensor_set':'C_MOD_ALL',
                                        'calibration_frequency_limits':(10,1e6),
                                        'I':4.5,'T':2e-2,'dt':1e-6,'m':[1],'n':[1],
                                        'save_ext_input':'_f-sweep_All-Mirnovs-Corrected'*True}
    plot_sensors = 'all'#['bp01_abk']
    needs_correction = ['12_abk', '1t_ghk', '3t_ghk', '_ef_bot', '_ef_top']
    input_channel = 16 # Channel for calibration signal
    ACQ_board = 2 # ACQ board for calibration signal
    tLim = [.5,1.65]
    B0_normalize = False
    compareSynthetic = True
    fLim = [0,1e6]
    yLim_mag = [0,21]
    yLim_phase = [60,460]
    freq_domain_calculation = True # If True, use synthetic frequency domain data, otherwise time domain
    doSave='../output_plots/'*True
    save_Ext_plot='_newCoords_Tiles_Mirnov_Shields'*True

    R = 20     # Ohms 
    L = 60e-6      # Henry
    C = 780e-12#780e-12     # Farads

    out = compareBode(shotno=comparison_shot,doSave=doSave,doPlot=True,
                synthDataFileNameInfo=synthDataFileNameInfo,plot_sensors=plot_sensors,
                input_channel=input_channel,tLim=tLim, ACQ_board=ACQ_board,
                needs_correction=needs_correction, B0_normalize=B0_normalize,
                compareSynthetic=compareSynthetic,fLim=fLim, R=R, L=L, C=C,
                freq_domain_calculation=freq_domain_calculation,yLim_mag=yLim_mag,
                yLim_phase=yLim_phase,save_Ext_plot=save_Ext_plot,) 
    
    transfer_mag, transfer_phase, f, sensors, synthetic_transfer_mag, \
        synthetic_transfer_phase,f_synth, calib, time, mirnovs, t_inds = out
    #compareBode(shotno=1051202011,doSave='../output_plots/Cmod_1051202011',doPlot=True)
    
    # transfer_mag = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Mag_1151208900.nc')
    # transfer_phase = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Phase_1151208900.nc')
    # transfer_complex = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Complex_1151208900.nc')
    # plot_transfer_function(transfer_mag, transfer_phase,transfer_complex, shotno=1151208900)
    print('Done')



#########################################################################################
# def make_transfer_function(mirnov_Bode, calib_mag, calib_phase,\
#                             calib_freq,calib_complex,  shotno, mirnov, doSave=True):
#     # Amplitude component

#     transfer_mag = []
#     transfer_phase = []
#     transfer_complex = []
#     for name in mirnov_Bode.data_vars.keys():
#         if 'mag' in name:
#             # Get the frequency and magnitude
#             #freq = mirnov_Bode[name.replace('_mag', '_freq')].data
#             mag = mirnov_Bode[name].data
            
#             # Calculate the transfer function
#             transfer_mag.append( mag / calib_mag )
            
#             transfer_phase.append( ( calib_phase - calib_phase[0]) - \
#                                   ( mirnov_Bode[name.replace('_mag', '_phase')].data -\
#                                   mirnov_Bode[name.replace('_mag', '_phase')].data[0] ) )
    
#             # Complex tranfer function
#             try: # changes depending on if we calculated or loaded the Mirnov Bode plots
#                 complex_arr = mirnov_Bode[name.replace('_mag', '_hilbert')].data['r'] +\
#                   1j * mirnov_Bode[name.replace('_mag', '_hilbert')].data['i']
#             except: complex_arr = mirnov_Bode[name.replace('_mag', '_hilbert')].data

#             transfer_complex.append(complex_arr/ calib_complex)


#     transfer_mag = xr.DataArray(np.array(transfer_mag), dims=['sensor', 'freq'], \
#                 coords={'sensor': mirnov.names, 'freq': calib_freq},\
#                     attrs={'shotno': shotno, 'units': 'T/s/A'})
#     transfer_mag= transfer_mag.assign_coords(time=('freq', calib_freq))

#     transfer_phase = xr.DataArray(np.array(transfer_phase), dims=['sensor', 'freq'], \
#                 coords={'sensor': mirnov.names, 'freq': calib_freq},
#                 attrs={'shotno': shotno, 'units': 'radians'})
#     transfer_phase= transfer_phase.assign_coords(time=('freq', calib_freq))
    
#     transfer_complex = xr.DataArray(np.array(transfer_complex), dims=['sensor', 'freq'], \
#                 coords={'sensor': mirnov.names, 'freq': calib_freq},
#                 attrs={'shotno': shotno, 'units': 'complex amplitude'})
#     transfer_complex= transfer_complex.assign_coords(time=('freq', calib_freq))
    

#     # Save the transfer function
#     if doSave:
#         transfer_mag.to_netcdf(data_archive_path+'Cmod_Transfer_Mag_%d.nc'%shotno)
#         transfer_phase.to_netcdf(data_archive_path+'Cmod_Transfer_Phase_%d.nc'%shotno)
#         transfer_complex.to_netcdf(data_archive_path+'Cmod_Transfer_Complex_%d.nc'%shotno,auto_complex=True)

#     return transfer_mag, transfer_phase, transfer_complex
# ####################################################################
# def plot_transfer_function(combined_mag, combined_phase, transfer_complex,shotno,
#                             doSave='', doPlot=True,
#                            downsample=1000):
#     if doPlot:
#         plt.close('Transfer_Function_%s'%shotno)
#         fig, ax = plt.subplots(7,7, num='Transfer_Function',tight_layout=True,sharex=True,
#                                sharey=True,figsize=(10,10))
        
#         ax = ax.flatten()
#         for i, name in enumerate(combined_mag.sensor):
#             # Complex signal extraction
#             # complex_arr = transfer_complex.sel(sensor=name)[::downsample]
#             # complex_arr = complex_arr.data['r'] + 1j * complex_arr.data['i']
#             # mag = np.abs(complex_arr)
#             mag = np.abs(combined_mag.sel(sensor=name)[::downsample])
#             ax[i].plot(combined_mag.freq[::downsample]*1e-6, mag ,  label='%s'%name.data)
#             #ax[i].set_title(name)
            
#             ax[i].grid()
#             ax[i].legend(fontsize=8,handlelength=.5)
#             ax_ = ax[i].twinx()
#             # angle = np.anglenp.unwrap(np.angle(complex_arr,deg=True))
#             angle = combined_phase.sel(sensor=name)[::downsample] * 180 / np.pi
#             ax_.plot(combined_phase.freq[::downsample]*1e-6, angle, color='C1',
#                       label='Phase', alpha=.6)
            
#             ax_.set_ylim([-360,360])
#             #if i >= 42: ax[i].set_xlabel('Frequency [kHz]')
#             ax[i].set_xlim(combined_mag.freq[0]*1e-6, combined_mag.freq[-1]*1e-6)
#             #if i % 7 == 0: ax[i].set_ylabel('Transfer Magnitude [T/s/A]')
#             #if i % 7 == 6: ax[i].set_ylabel('Transfer Phase [deg]')
#             if i == 7 * 3: ax[i].set_ylabel('Transfer Magnitude [T/s/A]')
#             if i == 7*6 + 3: ax[i].set_xlabel('Frequency [MHz]')
#             if i == 7*3 + 6: ax_.set_ylabel('Transfer Phase [deg]')
#             if (i+1) % 7 != 0: ax_.set_yticklabels([])

#         #fig.text(0.5, 0.04,'Frequency [MHz]', ha='center')
#         #fig.text(0.04, 0.5, 'Transfer Magnitude [T/s/A]', va='center', rotation='vertical')
#         #fig.text(0.94, 0.5, 'Transfer Phase [deg]', va='center', rotation='vertical')
#         plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
#         if doSave: fig.savefig(doSave+'_Cmod_Transfer_Function_%d.pdf'%shotno,dpi=300)
#         plt.show()
    
#     return
#################################################################### 
######################################################################
# def bode_Mirnov(shotno, doPlot=['bp1t_abk','bp01_abk', 'bp1t_ghk'], doSave='',\
#                  block_reduce=15, tLim=[0,1]):
#     '''
#     Calculate the Bode plot for all the Mirnov signal
#     '''
#     try: 
#         raise SyntaxError
#         mirnov_bode_xr = xr.load_dataset(data_archive_path+'Cmod_Mirnov_Bode_%d.nc'%shotno)
#         mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']
#         print('Loaded Bode data from file')
#     except:

#         mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']

#         mirnov_Bode = {}
#         for channel_ind, name in enumerate(mirnovs.names):

#              # Time vector
#             time = mirnovs.time
#             t_inds = np.where((time >= tLim[0]) & (time <= tLim[1]))[0]
#             time = time[t_inds]

#             # Get the Hilbert transform of the signal
#             mirnov_hilbert = hilbert(mirnovs.data[channel_ind][t_inds])
            
#             # Calculate the magnitude and phase
#             mag = np.abs(mirnov_hilbert)
#             phase = np.unwrap(np.angle(mirnov_hilbert))
#             #phase -= phase[0]
            
#             # Calculate frequency
#             dt = 1e-6  # Assuming a sampling rate of 1 MHz
#             freq = np.diff(phase) / (2.0 * np.pi * dt)
            
#             time = downsample_signal(time[1:], block_reduce)
#             freq = downsample_signal(freq, block_reduce)
#             mag = downsample_signal(mag[1:], block_reduce)
#             phase = downsample_signal(phase[1:], block_reduce)
#             mirnov_hilbert = downsample_signal(mirnov_hilbert[1:], block_reduce)
           
#             # phase = phase[t_inds]
#             # mag = mag[t_inds]
#             # freq = freq[t_inds]
#             # mirnov_hilbert = mirnov_hilbert[t_inds]

     

#             mirnov_Bode[name] = {
#                 'time': time,
#                 'mag': mag,
#                 'phase': phase,
#                 'freq': freq,
#                 'hilbert': mirnov_hilbert
#             }
        
#             print(f'Calculated Bode for {name}')
#         mirnov_bode_xr = dict_to_xarray(mirnov_Bode, \
#                        filename=data_archive_path+'Cmod_Mirnov_Bode_%d.nc'%shotno)
    

#     if np.any(doPlot):
#         plt.close('Bode_Mirnov')
#         fig, ax = plt.subplots(len(doPlot),1, num='Bode_Mirnov',tight_layout=True,sharex=True)
#         for i, name in enumerate(doPlot):
#             ax[i].plot(mirnovs.time, mirnovs.data[mirnovs.names.index(name)], label=f'{name} Signal')
#             ax[i].plot(mirnov_bode_xr[name+'_time'], mirnov_bode_xr[name+'_mag'],alpha=.3, label=f'{name} Magnitude')
#             ax[i].set_ylabel(r'$\frac{d}{dt}$B$_\theta$ [T/s]')
#             ax[i].grid()
           
#             if i == len(doPlot) - 1:
#                 ax[i].set_xlabel('Time [s]')
#             ax[i].legend(loc='upper left', fontsize=8)
    
#     return mirnovs, mirnov_bode_xr
# ###################################################################
# ################################################################
# def bode_driver(shot, tLim=[0,1],doPlot=True,doSave=False, block_reduce=15):
#     try: 
#         #raise SyntaxError
#         data = np.load(data_archive_path+'Cmod_driver_signal_%d.npz'%shot)
#         calib_mag = data['calib_mag']
#         calib_freq = data['calib_freq']
#         time = data['time']
#         calib_phase = data['calib_phase']
#         calib = data['calib']
#         calib_hilbert = data['calib_hilbert']
#         print('Loaded driver signal from file')
#     except Exception as e:
#         print(e)
#         calib,time = __load_calib_signal(shot,tLim)

#         print('Driver Signal Length: %s'%len(calib))

#         #Filter
#         #calib = __doFilter(calib,time,10, None)
#         calib_hilbert = hilbert(calib)
#         calib_mag = np.abs(calib_hilbert)
#         calib_phase = np.unwrap(np.angle(calib_hilbert)) 
#         calib_phase -= calib_phase[0]  # Normalize phase to start at zero
#         calib_freq = (np.diff(calib_phase) /(2*np.pi)  * (1/(time[1]-time[0])))
#         calib_mag = calib_mag[1:] 

#         np.savez(data_archive_path+'Cmod_driver_signal_%d.npz'%shot,
#                  calib_mag=calib_mag,calib_freq=calib_freq, time=time,
#                    calib=calib, calib_phase=calib_phase,calib_hilbert=calib_hilbert)

#     # downsample
#     calib_mag = downsample_signal(calib_mag[1:], block_reduce)
#     calib_freq = downsample_signal(calib_freq, block_reduce)
#     calib_phase = downsample_signal(calib_phase[1:], block_reduce)
#     time_orig= time
#     time = downsample_signal(time[1:], block_reduce)
#     calib_hilbert = downsample_signal(calib_hilbert[1:], block_reduce)

#     if doPlot:
#         plt.close('Bode_Driver')
#         fig, ax = plt.subplots(2,1, num='Bode_Driver',tight_layout=True,sharex=True)
#         ax[0].plot(time_orig, calib, label='Driver Signal')
#         ax[0].plot(time, calib_mag, label='Driver Envelope', alpha=.3)
#         ax[1].plot(time, calib_freq*1e-3,label='Driver Frequency')
#         ax[1].set_xlabel('Time [s]')
#         ax[1].set_ylabel('Frequency [kHz]')
#         ax[0].set_ylabel('Magnitude [A]')
#         for i in range(2):
#             ax[i].grid()
#             ax[i].legend(fontsize=8)
#         if doSave: fig.savefig__load_calib_signal(doSave+'_Driver_Bode.pdf',dpi=300)
#         plt.show()
    
#     return time, calib_mag, calib_freq, calib_phase, calib_hilbert
# def dict_to_xarray(data_dict, filename="output.nc"):
#     """
#     Converts a dictionary of dictionaries of 1D arrays into an xarray Dataset and saves it to a NetCDF file.

#     Args:
#         data_dict (dict): A dictionary where:                     calibration_frequency_limits, comparison_shot,doSave)

#             - Keys are top-level variable names (e.g., sensor names).
#             - Values are dictionaries where:
#                 - Keys are data component names (e.g., 'time', 'mag', 'phase', 'freq').
#                 - Values are 1D NumPy arrays.
#         filename (str, optional): The name of the output NetCDF file. Defaults to "output.nc".
#     """

#     # Determine the dimensions based on the first array in the first dictionary
#     first_key = next(iter(data_dict))
#     first_array_key = next(iter(data_dict[first_key]))
#     time_dim = len(data_dict[first_key][first_array_key])

#     # Create a dictionary to hold xarray DataArrays
#     data_vars = {}

#     # Iterate through the dictionary and create DataArrays
#     for top_level_key, inner_dict in data_dict.items():
#         for component_key, array_data in inner_dict.items():
#             # Create a unique name for the DataArray
#             data_array_name = f"{top_level_key}_{component_key}"

#             # Create the DataArray
#             data_vars[data_array_name] = (["time"], array_data)

#     # Create the xarray Dataset
#     ds = xr.Dataset(
#         data_vars=data_vars,
#         coords={"time": np.arange(time_dim)}  # Create a default time coordinate
#     )

#     # Save the Dataset to a NetCDF file
#     ds.to_netcdf(filename, auto_complex=True)
#     print(f"Saved xarray dataset to: {filename}")
    
#     return ds
# def downsample_signal(signal, factor=15):
#     """
#     Downsamples a 1D signal by averaging in blocks.

#     Args:
#         signal (np.ndarray): The input signal (1D NumPy array).
#         factor (int): The downsampling factor (block size).

#     Returns:
#         np.ndarray: The downsampled signal.
#     """
#     if factor <= 0:
#         raise ValueError("Downsampling factor must be positive.")

#     # Calculate the number of full blocks
#     num_blocks = len(signal) // factor

#     # Reshape the signal into blocks
#     reshaped_signal = signal[:num_blocks * factor].reshape(num_blocks, factor)

#     # Average each block
#     downsampled_signal = np.mean(reshaped_signal, axis=1)

#     return downsampled_signal


