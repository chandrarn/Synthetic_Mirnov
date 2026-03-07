# Test case for ThinCurr frequency scan compared to C-Mod copper plasma calibration


from header_signal_generation import (
    ThinCurr,
    OFT_env,
    np,
)


def do_Synthetic_Frequency_Scan(
    mesh_file="C_Mod_ThinCurr_Combined-homology.h5",
    xml_filename="oft_in.xml",
    filament_params={
        "R": 0.8,
        "Z": 0,
    },
    doSave="../output_plot/C-Mod_Bode/",
    sensor_set="C_MOD_ALL",
    cmod_shot=1151208900,
    eta="1.8E-5, 1.8E-5, 3.6E-5",
    freqs=np.logspace(1, 1e6, 10),
    debug=True,
    n_threads=12,
):
    # Load the mesh and create the sensor set
    myOFT = OFT_env(nthreads=4)
    tw_torus = ThinCurr(myOFT)
    tw_torus.setup_model(mesh_file=mesh_file, xml_filename="oft_in.xml")
    tw_torus.setup_io()

    tw_torus.compute_Lmat()
    tw_torus.compute_Rmat()


#####################################################3
