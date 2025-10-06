# Batch script to run arbitrary number of ThinCurr simulations, with arbitrary mode number, frequency, etc
# Automatically runs spectrogram calculation, and saves real/imag along with mode numbers, and frequency vs time, in an xarray dataset.
# Break up task into: get modes/frequencies, run simulations, calculate spectrograms, save results.


# Import necessary libraries

from header_signal_generation import plt, np, sys, xr, Fraction, os
sys.path.append('../C-Mod/')
from Synthetic_Mirnov import gen_synthetic_Mirnov
from spectrogram_Cmod import signal_spectrogram_C_Mod

from gen_mode_evolutions import gen_mode_params, gen_mode_params_for_training


#######################################################################3

def batch_run_synthetic_spectrogram(output_directory='',
                                    ThinCurr_params={'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
                                        'sensor_set':'Synth-C_MOD_BP_T',
                                        'save_ext':'','doNoise':False},
                                    Mode_params={'dt':1e-6,'T':10e-3,'periods':3},
                                    spectrogram_params={'pad':230,'fft_window':230,'block_reduce':(230,0)},
                                    save_Ext='',max_modes=5,max_m=15,max_n=5,
                                    doSave=False,doPlot=False,training_shots=1):
    """
    Batch run synthetic spectrogram generation for given parameters.
    
    Parameters:
    - output_directory: Directory to save output files.
    - ThinCurr_params: Dictionary containing mesh file and sensor set.
    - save_Ext: Extension for saved files.
    - archiveExt: Archive extension for saving simulation parameters.
    - doSave: Boolean to indicate if results should be saved.
    - doPlot: Boolean to indicate if plots should be generated.
    - doNoise: Boolean to add noise to signals.
    - pad: Padding for the spectrogram.
    - fft_window: Window size for FFT.
    - f_lim: Frequency limits for the spectrogram.
    - sensor_names: List of sensor names to use in the simulation.
    
    Returns:
    - None
    """

    # Generate frequencies and mode numbers
    per_shot_mode_params =  gen_mode_params_for_training(training_shots=training_shots,params=Mode_params,\
                                doPlot=doPlot,save_ext=save_Ext+'_')

    # For each mode, run the simulation
    for mode_param in per_shot_mode_params:
        print(f"Running simulation for mode: {mode_param['m']}/{mode_param['n']} at frequency {mode_param['f']} Hz")
        
        # Generate synthetic Mirnov signals
        # # Need sensor list output for sensor names
        gen_synthetic_Mirnov(
            mesh_file=ThinCurr_params['mesh_file'],
            sensor_set=ThinCurr_params['sensor_set'],
            params=mode_param,
            save_ext=ThinCurr_params['save_ext'],
            doSave=doSave,
            archiveExt='training_data/',
            doPlot=doPlot,
            plotOnly=False,
            wind_in=ThinCurr_params['wind_in'],
            eta = ThinCurr_params['eta'],
            file_geqdsk=mode_param['file_geqdsk'],
            cmod_shot=ThinCurr_params['cmod_shot']
        )
        
        # Calculate spectrogram
        # Needs to return complex valued spectrogram for each sensor, shape (n_sensors, n_freq, n_time)
        diag,signals,time,out_spect,out_spect_all_cplx, freq, sensor_name = signal_spectrogram_C_Mod(
            shotno=None,  # No shot number for synthetic data
            params=mode_param,
            sensor_name=None,
            sensor_set=ThinCurr_params['sensor_set'],
            pad=spectrogram_params['pad'],
            fft_window=spectrogram_params['fft_window'],
            doSave=doSave,
            save_Ext=save_Ext,
            doPlot=doPlot,
            mesh_file = ThinCurr_params['mesh_file'],
            archiveExt='training_data/',
            tLim=[0,Mode_params['T']],
            block_reduce=spectrogram_params['block_reduce'],
            plot_reduce=(1,1),filament=True, use_rolling_fft=True,
            f_lim=None,cLim=None,tScale=1e3,doSave_data=True # Set frequency limits and color limits
        )
        
        # Save results in an xarray dataset
        if doSave:
            xarray_file = save_xarray_results(output_directory, mode_param, time, freq, out_spect_all_cplx,ThinCurr_params,save_Ext,sensor_name)

            convert_spectrogram_to_training_data(xarray_file, timepoint_index=20,doSave=doSave,doPlot=doPlot)
#######################################################################################
######################################################################################
def save_xarray_results(output_directory, mode_param, time, freq, out_spect,ThinCurr_params,save_Ext,sensor_names):
    # Save results in an xarray dataset with dimensions time, frequency, sensor name

    # Create a filename based on mode parameters, and how many files are already in the directory
    current_files = len(os.listdir(output_directory) )

    # In general, these are all lists 
    m = mode_param['m']
    n = mode_param['n']
    f = mode_param['f_Hz'] # Need frequency in Hz, instead of phase advance
    I = mode_param['I'] 
    if type(f) is float: f_out = '%d'%f*1e-3
    else: f_out = '_f-Custom'
    if type(m) is not list: mn_out = '%d-%d'%(m,n)  
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])

    # Generate filename
    fName = f'spectrogram_mn_{mn_out}_f_{f_out}_{ThinCurr_params["sensor_set"]}_{ThinCurr_params['mesh_file']}_{save_Ext}_Count_{current_files+1}.nc'

    f_save = os.path.join(output_directory, fName)

     # Create a dictionary to hold the DataArrays for each sensor
    data_vars = {}
    for i, sensor_name in enumerate(sensor_names):
        data_vars[sensor_name + '_real'] = (['frequency', 'time'], np.abs(out_spect[i]))
        data_vars[sensor_name + '_imag'] = (['frequency', 'time'], np.angle(out_spect[i]))

    # Add time dependent frequency, amplitude, as extra data_var
    time_inds_downsample = np.linspace(0, len(f[0])-1, len(time), dtype=int) # Downsample to match
    for ind,mode_data in enumerate(f):
            data_vars['F_Mode_%d'%ind] = (['time'], mode_data[time_inds_downsample])
            data_vars['I_Mode_%d'%ind] = (['time'], I[ind][time_inds_downsample])

    # Create a dictionary to hold the coordinates
    coords = {'frequency': freq, 'time': time}


    # Create the xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={'sampling_frequency': 1 / (time[1] - time[0]),
               'mode_m': m, # This can be a list, contains information on number of modes
               'mode_n': n,
               'sensor_set': ThinCurr_params['sensor_set'],
               'mesh_file': ThinCurr_params['mesh_file'],}
    )

    # Save the Dataset to a NetCDF file
    filename = f"{output_directory}/Spectrogram_{mn_out}_f{f_out}{save_Ext}.nc"
    ds.to_netcdf(filename)
    print(f"Saved xarray dataset to: {filename}")

    return filename

#####################################################################################
#####################################################################################
# Function to generate mode parameters for synthetic spectrogram generation
# Stored as separate script for clarity and reusability

#####################################################################################
# Function to convert spectrogram output into training data format
def convert_spectrogram_to_training_data(xarray_f_name, timepoint_index=20,\
                cLim=None,ylim=[0,500], doSave='',doPlot=True):
    """
    Convert spectrogram output into training data format.
    
    - For a given timepoint: concatenate all the real, imaginary components for each sensor
      into two 2D matrices of size frequency vs number of sensors.
    
    Returns:
    - None (displays plots)
    """

    # Load in xarray dataset with spectrogram data
    ds = xr.open_dataset(xarray_f_name)

    # Get sensor names (excluding 'frequency' and 'time' which are coordinates)
    sensor_names = [var for var in ds.data_vars if "Mode" not in var]
    num_sensors = len(sensor_names) // 2  # Divide by 2 since we have _real and _imag for each sensor

    # Initialize matrices to hold the real and imaginary components
    spect_real = np.zeros((len(ds['frequency']), num_sensors))
    spect_imag = np.zeros((len(ds['frequency']), num_sensors))

    # Populate the matrices
    for i in range(num_sensors):
        sensor_name = sensor_names[i*2][:-5] # Remove _real or _imag
        spect_real[:, i] = ds[sensor_name + '_real'].isel(time=timepoint_index).values
        spect_imag[:, i] = ds[sensor_name + '_imag'].isel(time=timepoint_index).values


    if doPlot:__plot_training_matricies(spect_real, spect_imag, ds, timepoint_index, sensor_names,\
                               num_sensors,doSave=doSave,cLim=cLim,ylim=ylim)
############################################33
def __plot_training_matricies(spect_real, spect_imag, ds, timepoint_index, sensor_names,\
                               num_sensors,cLim=None, ylim=None,doSave=''):
    # Plot the matrices
    fig, ax = plt.subplots(1, 2, figsize=(12, 6),sharex=True, sharey=True,tight_layout=True)

    # Set color limits if provided
    if cLim is not None:
        vmin, vmax = cLim
    else:
        vmin = None
        vmax = None

    im0 = ax[0].imshow(spect_real, aspect='auto', origin='lower',vmin=vmin, vmax=vmax,
            extent=[0, num_sensors, ds['frequency'].min()*1e-3, ds['frequency'].max()*1e-3])
    fig.colorbar(im0,ax=ax[0], label=r'$||\frac{d}{dt}B_\theta||$ [T/s]')
    ax[0].set_xlabel('Sensor Index')
    ax[0].set_ylabel('Frequency [kHz], t=%3.3f ms'%(ds['time'][timepoint_index]*1e3))
    ax[0].set_xticklabels([sensor_names[i*2][:-5] for i in np.arange(0,num_sensors,10,dtype=int)],\
                        rotation=90)
    #ax[0].title(f'Real Component (Time Index {timepoint_index})')

    if ylim is not None:
        ax[0].set_ylim(ylim)
    
    im1 = ax[1].imshow(spect_imag/np.pi, aspect='auto', origin='lower',
            extent=[0, num_sensors, ds['frequency'].min()*1e-3, ds['frequency'].max()*1e-3])
    fig.colorbar(im1,ax=ax[1], label=r'$\angle\left[\frac{d}{dt}B_\theta\right]$ [$\pi$-radians]')
    ax[1].set_xlabel('Sensor Index')
    #ax[1].set_ylabel('Frequency')
    ax[1].set_xticklabels([sensor_names[i*2][:-5] for i in np.arange(0,num_sensors,10,dtype=int)],\
                        rotation=90)
    # ax[1].title(f'Imaginary Component (Time Index {timepoint_index})')

    if doSave:
        save_path = os.path.join(doSave, f"Training_Matrices_TimeIndex_{timepoint_index}.pdf")
        plt.savefig(save_path)
        print(f"Saved training matrices plot to: {save_path}")

    plt.show()
##################################################################3
if __name__ == '__main__':
    # Example usage
    output_directory = '../data_output/synthetic_spectrograms/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # ThinCurr Eta calculation
    # Bulk resistivities
    Mo = 53.4e-9 # Ohm * m at 20c
    SS = 690e-9 # Ohm * m at 20c
    w_tile_lim = 1.5e-2  # Tile limiter thickness
    w_tile_arm = 1.5e-2 *1 # Tile extention thickness
    w_vv = 3e-2 # Vacuum vessel thickness
    w_ss = 1e-2  # Support structure thickness
    w_shield = 0.43e-3 
    # Surface resistivity: eta/thickness
    # Assume that limiter support structures are 0.6-1.5cm SS, tiles are 1.5cm thick Mo, VV is 3cm thick SS 
    # For more accuracy, could break up filaments into different eta values based on position
    eta = f'{SS/w_ss}, {Mo/w_tile_lim}, {SS/w_ss}, {Mo/w_tile_lim}, {SS/w_vv}, {SS/w_ss}, {Mo/w_tile_arm}, {SS/w_shield}' 

    ThinCurr_params = {
        'mesh_file': 'C_Mod_ThinCurr_Combined-homology.h5',
        'sensor_set': 'C_MOD_ALL',
        'cmod_shot' : 1051202011,
        'save_ext': '',
        'doNoise': False,
        'wind_in' : 'phi',
        'file_geqdsk' : 'g1051202011.1000',
        'eta' : eta
    }

    Mode_params = {'dt':1e-7,'T':1e-3,'periods':2,'n_pts':100,'m_pts':60,'R':None,'r':None,\
                   'noise_envelope':0.00,'n_threads':12,'max_modes':1,'max_m':1,'max_n':1} 

    spectrogram_params = {'pad':230,'fft_window':230,'block_reduce':(230,10)}

    save_Ext = '_Synth_low-n'
    doSave = '../output_plots/'
    doPlot = True
    training_shots = 1

    batch_run_synthetic_spectrogram(
        output_directory=output_directory,
        ThinCurr_params=ThinCurr_params,
        Mode_params=Mode_params,
        spectrogram_params=spectrogram_params,
        save_Ext=save_Ext,
        doSave=doSave,
        doPlot=doPlot,
        training_shots=training_shots
    )


    print('Done batch running synthetic spectrograms')