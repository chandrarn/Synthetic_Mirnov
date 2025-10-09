#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from header_synthetic_mirnov_generation import np, plt, pyvista, ThinCurr,\
    Mirnov, save_sensors, OFT_env, mu0, histfile, geqdsk, xr

from prep_ThinCurr_input import gen_coil_currs, gen_filament_coords, calc_filament_coords_field_lines,\
      gen_OFT_sensors_file, gen_OFT_filement_and_eta_file


from run_ThinCurr_model import get_mesh, run_frequency_scan, makePlots, correct_frequency_response


def thincurr_fft(
    probe_details: xr.Dataset,
    mesh_model: str,
    eqdsk: geqdsk,
    freq: float,
    mode: dict,
    eta: list[float],
    working_files_directory: str = 'input_data/',
    save_Ext: str = '',
    debug: bool = True,
    n_threads: int = 0,
    sensor_freq_response: dict = {},
    doPlot: bool = False,
    doSave: bool = False
) -> xr.Dataset:
    


    """
    Calculate the real and imaginary components of the magnetic probe signals using ThinCurr.
    Assumes that a valid conducting structure mesh is provided.


    Args:
        probe_details: Dataset containing probe geometry in X,Y,Z coordinate, orientation in theta, phi, 
        mesh_model: Path to the vessel model file for ThinCurr
        eqdsk: Equilibrium field data from a gEQDSK file
        freq: Frequency of the mode to simulate
        mode: dictionary containing m, n and m_pts and n_pts to set the filament resolution
        debug: If True, print debug information
        n_threads: Number of threads to use for ThinCurr calculations. If 0, use all available threads.
        sensor_freq_response: Dictionary with sensor names as keys and frequency response functions as values,
            if empty, assume that the sensors have a flat response
        doPlot: If True, generate debug plots of the mesh, filaments, and sensors
    Returns:
        xr.Dataset: Dataset containing the simulated probe signals in [T/s].
    """


    ######################################################################################
    # Prepare filament currents, locations

    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(mode)

    # Get starting coorindates for fillaments
    theta,phi=gen_filament_coords(mode)

    filament_coords = calc_filament_coords_field_lines(mode,eqdsky,doDebug=debug)

    # Put filamanets in OFT file format
    gen_OFT_filement_and_eta_file(working_files_directory,mode,filament_coords, eta)



    ######################################################################################
    # Generate sensors in OFT format
    sensors=gen_OFT_sensors_file(probe_details, working_files_directory, debug=debug)



    ######################################################################################
    # Prepare ThinCurr Model
        # Get finite element Mesh
    tw_mesh, sensor_obj, Mc = \
        get_mesh(mesh_model,working_files_directory,mode,sensors, debug=debug)
    



    ######################################################################################
    # Calculate frequency response
    sensors_bode = run_frequency_scan(tw_mesh,n_threads,sensors,mesh_model,sensor_obj,\
                                      working_files_directory,save_Ext)
    
    # Correct for sensor frequency response
    correct_frequency_response(sensors_bode, sensor_freq_response, freq,doSave)

    ######################################################################################
    # Optional debug output
    # Plot mesh, filaments, sensors, and currents
    scale= makePlots(tw_mesh,mode,coil_currs,sensors,working_files_directory,
                                save_Ext,Mc,filament_coords,eqdsk,sensors,
                                plot_B_surf=True,debug=debug,clim_J=None,doPlot=doPlot)
    

    ######################################################################################
        

    return sensors_bode


###############################################################################
###############################################################################
###############################################################################
# Example code to generate a response in the C-Mod mesh
if __name__ == '__main__':
    # Load example probe details
    probe_details = xr.load_dataset('input_data/mirnov_probes_CMod.nc')

    # Load example equilibrium
    file_geqdsk = 'input_data/gEQDSK_CMod_200kA_2.0T'
    eqdsk = geqdsk(file_geqdsk)

    # Define mode to simulate
    mode = {'m': 4, 'n': 2, 'm_pts': 20, 'n_pts': 40}

    # Resistivity of the conducting structure in Ohm-m
    eta = [1e-6]

    # Path to the mesh model file for ThinCurr
    mesh_model = 'input_data/CMod_vessel_ThinCurr_mesh.xml'

    # Frequency to simulate
    freq = 1000  # Hz

    # Sensor frequency response functions (example)
    sensor_freq_response = {
        'Mirnov1': lambda f: 1 / (1 + 1j * f / 1000),
        'Mirnov2': lambda f: 1 / (1 + 1j * f / 2000),
        # Add more sensors as needed
    }

    # Run ThinCurr FFT simulation
    sensors_bode = thincurr_fft(
        probe_details,
        mesh_model,
        eqdsk,
        freq,
        mode,
        eta,
        working_files_directory='input_data/',
        save_Ext='_CMod',
        debug=True,
        n_threads=4,
        sensor_freq_response=sensor_freq_response
    )

    # Print results
    for i, sensor_name in enumerate(sensors_bode['sensor_names'].values):
        print(f'Sensor: {sensor_name}, Real: {sensors_bode[sensor_name].real:.5e}, Imag: {sensors_bode[sensor_name].imag:.5e}')