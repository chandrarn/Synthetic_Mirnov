#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from header_synthetic_mirnov_generation import np, plt, pyvista, ThinCurr,\
    Mirnov, save_sensors, OFT_env, mu0, histfile, geqdsk, xr

from prep_ThinCurr_input import gen_coil_currs, gen_filament_coords, calc_filament_coords_field_lines,\
      gen_OFT_sensors_file, gen_OFT_filement_and_eta_file


from run_ThinCurr_model import get_mesh, run_frequency_scan, makePlots


def thincurr_fft(
    probe_details: xr.Dataset,
    mesh_model: str,
    eqdsk: geqdsk,
    freq: float,
    mode: tuple[int, int],
    eta: list[float],
    working_files_directory: str = 'input_data/',
    save_Ext: str = '',
    debug: bool = True,
) -> xr.Dataset:
    


    """
    Calculate the real and imaginary components of the magnetic probe signals using ThinCurr.
    Assumes that a valid conducting structure mesh is provided.


    Args:
        probe_details: Dataset containing probe geometry in X,Y,Z coordinate, orientation in theta, phi, 
            as well as an internal dictionary with sensor frequency response information, for each named sensor
        mesh_model: Path to the vessel model file for ThinCurr
        eqdsk: Equilibrium field data from a gEQDSK file
        freq: Frequency of the mode to simulate
        mode: Tuple of (m, n) for the mode to simulate.

    Returns:
        xr.Dataset: Dataset containing the simulated probe signals.
    """


    ######################################################################################
    # Prepare filament currents, locations

    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(mode)

    # Get starting coorindates for fillaments
    theta,phi=gen_filament_coords(mode)

    filament_coords = calc_filament_coords_field_lines(mode,eqdsk,doDebug=debug)

    # Put filamanets in OFT file format
    gen_OFT_filement_and_eta_file(working_files_directory,mode,filament_coords, eta)



    ######################################################################################
    # Generate sensors in OFT format
    sensors=gen_OFT_sensors_file(probe_details, working_files_directory, debug=debug)



    ######################################################################################
    # Prepare ThinCurr Model
        # Get finite element Mesh
    tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv = \
        get_mesh(mesh_model,working_files_directory,mode,sensors, debug=debug)
    



    ######################################################################################
    # Calculate frequency response
    sensors_bode = run_frequency_scan(tw_mesh,mode,sensors,mesh_model,sensor_obj,\
                                      working_files_directory,save_Ext)
    


    ######################################################################################
    # Optional debug output
    # Plot mesh, filaments, sensors, and currents
    scale= makePlots(tw_mesh,mode,coil_currs,sensors,working_files_directory,
                                save_Ext,Mc, L_inv,filament_coords,eqdsk,sensors,
                                plot_B_surf=True,debug=debug,clim_J=None)
    

    return sensors_bode


###############################################################################
###############################################################################
###############################################################################
# Example code to generate a response in the C-Mod mesh
if __name__ == '__main__':
    pass