# Batch script to run arbitrary number of ThinCurr simulations, with arbitrary mode number, frequency, etc
# Automatically runs spectrogram calculation, and saves real/imag along with mode numbers, and frequency vs time, in an xarray dataset.
# Break up task into: get modes/frequencies, run simulations, calculate spectrograms, save results.


# Import necessary libraries

from header_signal_generation import plt, np, sys, xr, Fraction, os
#sys.path.append('../C-Mod/')
from Synthetic_Mirnov import gen_synthetic_Mirnov


from header_signal_analysis import doFFT



#######################################################################3

def batch_run_synthetic_spectrogram(output_directory='',
                                    ThinCurr_params={'mesh_file':'C_Mod_ThinCurr_Combined-homology.h5',
                                        'sensor_set':'Synth-C_MOD_BP_T',
                                        'save_ext':'','doNoise':False},
                                    Mode_params={},
                                    spectrogra_params={'pad':1900,'fft_window':1500},
                                    save_Ext='',
                                    doSave=True,doPlot=False,training_shots=1):
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
    mode_params = gen_mode_params(training_shots=training_shots)


    # For each mode, run the simulation
    for mode_param in mode_params:
        print(f"Running simulation for mode: {mode_param['m']}/{mode_param['n']} at frequency {mode_param['f']} Hz")
        
        # Generate synthetic Mirnov signals
        gen_synthetic_Mirnov(
            mesh_file=ThinCurr_params['mesh_file'],
            sensor_set=ThinCurr_params['sensor_set'],
            params=mode_param,
            save_ext=ThinCurr_params['save_ext'],
            doSave=doSave,
            archiveExt='training_data/',
            doPlot=doPlot,
            plotOnly=False
        )
        
        # Calculate spectrogram
        # Needs to return complex valued spectrogram for each sensor, shape (n_sensors, n_freq, n_time)
        time, freq, out_spect, all_spects = gen_single_spect_multimode(
            params=[mode_param],
            sensor_names=[ThinCurr_params['sensor_set']],
            sensor_set=ThinCurr_params['sensor_set'],
            pad=spectrogra_params['pad'],
            fft_window=spectrogra_params['fft_window'],
            doSave=doSave,
            filament=True,
            save_Ext=save_Ext
        )
        
        # Save results in an xarray dataset
        if doSave:
            save_xarray_results(output_directory, mode_param, time, freq, out_spect,ThinCurr_params,save_Ext)

#######################################################################################
######################################################################################
def save_xarray_results(output_directory, mode_param, time, freq, out_spect,ThinCurr_params,save_Ext):
    # Save results in an xarray dataset with dimensions time, frequency, sensor name

    # Create a filename based on mode parameters, and how many files are already in the directory
    current_files = len(os.listdir(output_directory) )

    # In general, these are all lists 
    m = mode_param['m']
    n = mode_param['n']
    f = mode_param['f']
    i = mode_param['i'] 
    if type(f) is float: f_out = '%d'%f*1e-3
    else: f_out = '-'.join([str(np.mean(f_)*1e-3) for f_ in f])
    if type(m) is not list: mn_out = '%d-%d'%(m,n)  
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])

    # Generate filename
    fName = f'spectrogram_mn_{mn_out}_f_{f_out}_{ThinCurr_params["sensor_set"]}_{ThinCurr_params['mesh_file']}_{save_Ext}_Count_{current_files+1}.nc'

    f_save = os.path.join(output_directory, fName)

     # Create a dictionary to hold the DataArrays for each sensor
    data_vars = {}
    for i, sensor_name in enumerate(sensor_names):
        data_vars[sensor_name] = (['time', 'frequency'], out_spect[i])

    # Create a dictionary to hold the coordinates
    coords = {'frequency': freq, 'time': time}


    # Create the xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={'sampling_frequency': 1 / (time[1] - time[0]),
               'mode_m': m,
               'mode_n': n,
               'mode_f': f,
               'mode_i': i,
               'sensor_set': ThinCurr_params['sensor_set'],
               'mesh_file': ThinCurr_params['mesh_file'],}
    )

    # Save the Dataset to a NetCDF file
    filename = f"{output_directory}/Spectrogram_{mn_out}_f{f_out}{save_Ext}.nc"
    ds.to_netcdf(filename)
    print(f"Saved xarray dataset to: {filename}")


#####################################################################################
#####################################################################################
# Function to generate mode parameters for synthetic spectrogram generation
def gen_mode_params(training_shots=1):
    """
    Generate mode parameters for synthetic spectrogram generation.
    
    Parameters:
    - training_shots: Number of training shots to generate modes for.
    
    Returns:
    - List of dictionaries containing mode parameters.
    """
    pass

# Function to convert spectrogram output into training data format
def convert_spectrogram_to_training_data(out_spect, time, freq, sensor_names):
    """
    Convert spectrogram output into training data format.
    
    - For every timepoint: concatenate all the real, imaginary components for each sensor
    - For every timepoint: note the frequencies of all the modes as the category labels, 
    - For every timepoint: note the mode numbers as regression labels
    
    Returns:
    - xarray dataset?
    """

    