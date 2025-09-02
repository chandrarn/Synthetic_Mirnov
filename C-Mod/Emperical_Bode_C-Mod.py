# Emperical Bode Calibration

import get_Cmod_Data as gC
from header_Cmod import np, plt, xr, hilbert, data_archive_path, __doFilter


def compareBode(shotno=1151208900,doSave='',doPlot=True,tLim=[.05, .997]):

    # Get driver signal
    time, calib_mag, calib_freq, calib_phase, calib_complex = \
        bode_driver(shotno,tLim=tLim)


    # Get Magnetics signals
    mirnov, mirnov_Bode = bode_Mirnov(shotno,tLim=tLim)

    # Generate transfer function
    transfer_mag, transfer_phase, transfer_complex =\
        make_transfer_function(mirnov_Bode, calib_mag, calib_phase, \
                               calib_freq,calib_complex, shotno, mirnov)

    plot_transfer_function(transfer_mag, transfer_phase, transfer_complex,shotno)
    print('Finsihed')
###################################################################
def make_transfer_function(mirnov_Bode, calib_mag, calib_phase,\
                            calib_freq,calib_complex,  shotno, mirnov, doSave=True):
    # Amplitude component

    transfer_mag = []
    transfer_phase = []
    transfer_complex = []
    for name in mirnov_Bode.data_vars.keys():
        if 'mag' in name:
            # Get the frequency and magnitude
            #freq = mirnov_Bode[name.replace('_mag', '_freq')].data
            mag = mirnov_Bode[name].data
            
            # Calculate the transfer function
            transfer_mag.append( mag / calib_mag )
            
            transfer_phase.append( ( calib_phase - calib_phase[0]) - \
                                  ( mirnov_Bode[name.replace('_mag', '_phase')].data -\
                                  mirnov_Bode[name.replace('_mag', '_phase')].data[0] ) )
    
            # Complex tranfer function
            try: # changes depending on if we calculated or loaded the Mirnov Bode plots
                complex_arr = mirnov_Bode[name.replace('_mag', '_hilbert')].data['r'] +\
                  1j * mirnov_Bode[name.replace('_mag', '_hilbert')].data['i']
            except: complex_arr = mirnov_Bode[name.replace('_mag', '_hilbert')].data

            transfer_complex.append(complex_arr/ calib_complex)


    transfer_mag = xr.DataArray(np.array(transfer_mag), dims=['sensor', 'freq'], \
                coords={'sensor': mirnov.names, 'freq': calib_freq},\
                    attrs={'shotno': shotno, 'units': 'T/s/A'})
    transfer_mag= transfer_mag.assign_coords(time=('freq', calib_freq))

    transfer_phase = xr.DataArray(np.array(transfer_phase), dims=['sensor', 'freq'], \
                coords={'sensor': mirnov.names, 'freq': calib_freq},
                attrs={'shotno': shotno, 'units': 'radians'})
    transfer_phase= transfer_phase.assign_coords(time=('freq', calib_freq))
    
    transfer_complex = xr.DataArray(np.array(transfer_complex), dims=['sensor', 'freq'], \
                coords={'sensor': mirnov.names, 'freq': calib_freq},
                attrs={'shotno': shotno, 'units': 'complex amplitude'})
    transfer_complex= transfer_complex.assign_coords(time=('freq', calib_freq))
    

    # Save the transfer function
    if doSave:
        transfer_mag.to_netcdf(data_archive_path+'Cmod_Transfer_Mag_%d.nc'%shotno)
        transfer_phase.to_netcdf(data_archive_path+'Cmod_Transfer_Phase_%d.nc'%shotno)
        transfer_complex.to_netcdf(data_archive_path+'Cmod_Transfer_Complex_%d.nc'%shotno,auto_complex=True)

    return transfer_mag, transfer_phase, transfer_complex
####################################################################
def plot_transfer_function(combined_mag, combined_phase, transfer_complex,shotno,
                            doSave='', doPlot=True,
                           downsample=1000):
    if doPlot:
        plt.close('Transfer_Function_%s'%shotno)
        fig, ax = plt.subplots(7,7, num='Transfer_Function',tight_layout=True,sharex=True,
                               sharey=True,figsize=(10,10))
        
        ax = ax.flatten()
        for i, name in enumerate(combined_mag.sensor):
            # Complex signal extraction
            # complex_arr = transfer_complex.sel(sensor=name)[::downsample]
            # complex_arr = complex_arr.data['r'] + 1j * complex_arr.data['i']
            # mag = np.abs(complex_arr)
            mag = np.abs(combined_mag.sel(sensor=name)[::downsample])
            ax[i].plot(combined_mag.freq[::downsample]*1e-6, mag ,  label='%s'%name.data)
            #ax[i].set_title(name)
            
            ax[i].grid()
            ax[i].legend(fontsize=8,handlelength=.5)
            ax_ = ax[i].twinx()
            # angle = np.anglenp.unwrap(np.angle(complex_arr,deg=True))
            angle = combined_phase.sel(sensor=name)[::downsample] * 180 / np.pi
            ax_.plot(combined_phase.freq[::downsample]*1e-6, angle, color='C1',
                      label='Phase', alpha=.6)
            
            ax_.set_ylim([-360,360])
            #if i >= 42: ax[i].set_xlabel('Frequency [kHz]')
            ax[i].set_xlim(combined_mag.freq[0]*1e-6, combined_mag.freq[-1]*1e-6)
            #if i % 7 == 0: ax[i].set_ylabel('Transfer Magnitude [T/s/A]')
            #if i % 7 == 6: ax[i].set_ylabel('Transfer Phase [deg]')
            if i == 7 * 3: ax[i].set_ylabel('Transfer Magnitude [T/s/A]')
            if i == 7*6 + 3: ax[i].set_xlabel('Frequency [MHz]')
            if i == 7*3 + 6: ax_.set_ylabel('Transfer Phase [deg]')
            if (i+1) % 7 != 0: ax_.set_yticklabels([])

        #fig.text(0.5, 0.04,'Frequency [MHz]', ha='center')
        #fig.text(0.04, 0.5, 'Transfer Magnitude [T/s/A]', va='center', rotation='vertical')
        #fig.text(0.94, 0.5, 'Transfer Phase [deg]', va='center', rotation='vertical')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
        
        if doSave: fig.savefig(doSave+'_Cmod_Transfer_Function_%d.pdf'%shotno,dpi=300)
        plt.show()
    
    return
#################################################################### 
######################################################################
def bode_Mirnov(shotno, doPlot=['bp1t_abk','bp01_abk', 'bp1t_ghk'], doSave='',\
                 block_reduce=15, tLim=[0,1]):
    '''
    Calculate the Bode plot for all the Mirnov signal
    '''
    try: 
        raise SyntaxError
        mirnov_bode_xr = xr.load_dataset(data_archive_path+'Cmod_Mirnov_Bode_%d.nc'%shotno)
        mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']
        print('Loaded Bode data from file')
    except:

        mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']

        mirnov_Bode = {}
        for channel_ind, name in enumerate(mirnovs.names):

             # Time vector
            time = mirnovs.time
            t_inds = np.where((time >= tLim[0]) & (time <= tLim[1]))[0]
            time = time[t_inds]

            # Get the Hilbert transform of the signal
            mirnov_hilbert = hilbert(mirnovs.data[channel_ind][t_inds])
            
            # Calculate the magnitude and phase
            mag = np.abs(mirnov_hilbert)
            phase = np.unwrap(np.angle(mirnov_hilbert))
            #phase -= phase[0]
            
            # Calculate frequency
            dt = 1e-6  # Assuming a sampling rate of 1 MHz
            freq = np.diff(phase) / (2.0 * np.pi * dt)
            
            time = downsample_signal(time[1:], block_reduce)
            freq = downsample_signal(freq, block_reduce)
            mag = downsample_signal(mag[1:], block_reduce)
            phase = downsample_signal(phase[1:], block_reduce)
            mirnov_hilbert = downsample_signal(mirnov_hilbert[1:], block_reduce)
           
            # phase = phase[t_inds]
            # mag = mag[t_inds]
            # freq = freq[t_inds]
            # mirnov_hilbert = mirnov_hilbert[t_inds]

     

            mirnov_Bode[name] = {
                'time': time,
                'mag': mag,
                'phase': phase,
                'freq': freq,
                'hilbert': mirnov_hilbert
            }
        
            print(f'Calculated Bode for {name}')
        mirnov_bode_xr = dict_to_xarray(mirnov_Bode, \
                       filename=data_archive_path+'Cmod_Mirnov_Bode_%d.nc'%shotno)
    

    if np.any(doPlot):
        plt.close('Bode_Mirnov')
        fig, ax = plt.subplots(len(doPlot),1, num='Bode_Mirnov',tight_layout=True,sharex=True)
        for i, name in enumerate(doPlot):
            ax[i].plot(mirnovs.time, mirnovs.data[mirnovs.names.index(name)], label=f'{name} Signal')
            ax[i].plot(mirnov_bode_xr[name+'_time'], mirnov_bode_xr[name+'_mag'],alpha=.3, label=f'{name} Magnitude')
            ax[i].set_ylabel(r'$\frac{d}{dt}$B$_\theta$ [T/s]')
            ax[i].grid()
           
            if i == len(doPlot) - 1:
                ax[i].set_xlabel('Time [s]')
            ax[i].legend(loc='upper left', fontsize=8)
    
    return mirnovs, mirnov_bode_xr
###################################################################
################################################################
def bode_driver(shot, tLim=[0,1],doPlot=True,doSave=False, block_reduce=15):
    try: 
        #raise SyntaxError
        data = np.load(data_archive_path+'Cmod_driver_signal_%d.npz'%shot)
        calib_mag = data['calib_mag']
        calib_freq = data['calib_freq']
        time = data['time']
        calib_phase = data['calib_phase']
        calib = data['calib']
        calib_hilbert = data['calib_hilbert']
        print('Loaded driver signal from file')
    except Exception as e:
        print(e)
        calib,time = __load_calib_signal(shot,tLim)

        print('Driver Signal Length: %s'%len(calib))

        #Filter
        #calib = __doFilter(calib,time,10, None)
        calib_hilbert = hilbert(calib)
        calib_mag = np.abs(calib_hilbert)
        calib_phase = np.unwrap(np.angle(calib_hilbert)) 
        calib_phase -= calib_phase[0]  # Normalize phase to start at zero
        calib_freq = (np.diff(calib_phase) /(2*np.pi)  * (1/(time[1]-time[0])))
        calib_mag = calib_mag[1:] 

        np.savez(data_archive_path+'Cmod_driver_signal_%d.npz'%shot,
                 calib_mag=calib_mag,calib_freq=calib_freq, time=time,
                   calib=calib, calib_phase=calib_phase,calib_hilbert=calib_hilbert)

    # downsample
    calib_mag = downsample_signal(calib_mag[1:], block_reduce)
    calib_freq = downsample_signal(calib_freq, block_reduce)
    calib_phase = downsample_signal(calib_phase[1:], block_reduce)
    time_orig= time
    time = downsample_signal(time[1:], block_reduce)
    calib_hilbert = downsample_signal(calib_hilbert[1:], block_reduce)

    if doPlot:
        plt.close('Bode_Driver')
        fig, ax = plt.subplots(2,1, num='Bode_Driver',tight_layout=True,sharex=True)
        ax[0].plot(time_orig, calib, label='Driver Signal')
        ax[0].plot(time, calib_mag, label='Driver Envelope', alpha=.3)
        ax[1].plot(time, calib_freq*1e-3,label='Driver Frequency')
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Frequency [kHz]')
        ax[0].set_ylabel('Magnitude [A]')
        for i in range(2):
            ax[i].grid()
            ax[i].legend(fontsize=8)
        if doSave: fig.savefig(doSave+'_Driver_Bode.pdf',dpi=300)
        plt.show()
    
    return time, calib_mag, calib_freq, calib_phase, calib_hilbert

# Load calibration signal
def __load_calib_signal(shot, tLim=[0,1]):
        conn=gC.openTree(shot,treeName='MAGNETICS')
        time=conn.get(r'dim_of(\MAGNETICS::TOP.ACTIVE_MHD.DATA_ACQ.CPCI.ACQ_216_3.INPUT_16)').data()
      
        calib=conn.get(r'\MAGNETICS::TOP.ACTIVE_MHD.DATA_ACQ.CPCI.ACQ_216_3.INPUT_16').data()
        # Trim signal
        t_inds = np.where((time >= tLim[0]) & (time <= tLim[1]))[0]
        time = time[t_inds]
        calib = calib[t_inds]* 10 
        return calib, time
###################################3###################################
def dict_to_xarray(data_dict, filename="output.nc"):
    """
    Converts a dictionary of dictionaries of 1D arrays into an xarray Dataset and saves it to a NetCDF file.

    Args:
        data_dict (dict): A dictionary where:                     calibration_frequency_limits, comparison_shot,doSave)

            - Keys are top-level variable names (e.g., sensor names).
            - Values are dictionaries where:
                - Keys are data component names (e.g., 'time', 'mag', 'phase', 'freq').
                - Values are 1D NumPy arrays.
        filename (str, optional): The name of the output NetCDF file. Defaults to "output.nc".
    """

    # Determine the dimensions based on the first array in the first dictionary
    first_key = next(iter(data_dict))
    first_array_key = next(iter(data_dict[first_key]))
    time_dim = len(data_dict[first_key][first_array_key])

    # Create a dictionary to hold xarray DataArrays
    data_vars = {}

    # Iterate through the dictionary and create DataArrays
    for top_level_key, inner_dict in data_dict.items():
        for component_key, array_data in inner_dict.items():
            # Create a unique name for the DataArray
            data_array_name = f"{top_level_key}_{component_key}"

            # Create the DataArray
            data_vars[data_array_name] = (["time"], array_data)

    # Create the xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": np.arange(time_dim)}  # Create a default time coordinate
    )

    # Save the Dataset to a NetCDF file
    ds.to_netcdf(filename, auto_complex=True)
    print(f"Saved xarray dataset to: {filename}")
    
    return ds
def downsample_signal(signal, factor=15):
    """
    Downsamples a 1D signal by averaging in blocks.

    Args:
        signal (np.ndarray): The input signal (1D NumPy array).
        factor (int): The downsampling factor (block size).

    Returns:
        np.ndarray: The downsampled signal.
    """
    if factor <= 0:
        raise ValueError("Downsampling factor must be positive.")

    # Calculate the number of full blocks
    num_blocks = len(signal) // factor

    # Reshape the signal into blocks
    reshaped_signal = signal[:num_blocks * factor].reshape(num_blocks, factor)

    # Average each block
    downsampled_signal = np.mean(reshaped_signal, axis=1)

    return downsampled_signal



################################################################################3
def sandbox(shotno,tLim=[0,1]):
    # Test cross-correlation method

    # Load driver signal
    calib,time = __load_calib_signal(shotno,tLim)

    # Load Mirnov signals
    mirnovs  =  gC.__loadData(shotno,pullData=['bp_k'],params={'tLim':[0,1]})['bp_k']



    print('Done')


#######################################################################################
if __name__ == '__main__':
    sandbox(1151208901,tLim=[0,1])

    compareBode(shotno=1151208901,doSave='../output_plots/',doPlot=True)
    #compareBode(shotno=1051202011,doSave='../output_plots/Cmod_1051202011',doPlot=True)
    
    # transfer_mag = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Mag_1151208900.nc')
    # transfer_phase = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Phase_1151208900.nc')
    # transfer_complex = xr.open_dataarray(data_archive_path+'Cmod_Transfer_Complex_1151208900.nc')
    # plot_transfer_function(transfer_mag, transfer_phase,transfer_complex, shotno=1151208900)
    
    print('Done')



