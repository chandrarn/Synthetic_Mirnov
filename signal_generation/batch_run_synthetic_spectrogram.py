# Batch script to run arbitrary number of ThinCurr simulations, with arbitrary mode number, frequency, etc
# Automatically runs spectrogram calculation, and saves real/imag along with mode numbers, and frequency vs time, in an xarray dataset.
# Break up task into: get modes/frequencies, run simulations, calculate spectrograms, save results.


# Import necessary libraries

from header_signal_generation import (
    plt,
    np,
    sys,
    xr,
    os,
    OFT_env,
    working_directory,
)

sys.path.append("../C-Mod/")
from Synthetic_Mirnov import gen_synthetic_Mirnov
from spectrogram_Cmod import signal_spectrogram_C_Mod

from gen_mode_evolutions import gen_mode_params_for_training

import re
#######################################################################3


def batch_run_synthetic_spectrogram(
    output_directory="",
    ThinCurr_params={
        "mesh_file": "C_Mod_ThinCurr_Combined-homology.h5",
        "sensor_set": "Synth-C_MOD_BP_T",
        "save_ext": "",
        "doNoise": False,
    },
    Mode_params={"dt": 1e-6, "T": 10e-3, "periods": 3},
    spectrogram_params={"pad": 230, "fft_window": 230, "block_reduce": (230, 0)},
    save_Ext="",
    max_modes=5,
    max_m=15,
    max_n=5,
    doSave=False,
    doPlot=False,
    training_shots=1,
    doPerturbation=True,
    justLoadGeqdsk=False,
    gEQDSK_files_dir="input_data/gEQDSK_files/",
    one_of_each_mn=False,
    prescribed_mn_pairs=None,
    doSave_Training_Data=False,
):
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

    effective_training_shots = (
        len(prescribed_mn_pairs) if prescribed_mn_pairs is not None else training_shots
    )

    # Generate frequencies and mode numbers
    per_shot_mode_params = gen_mode_params_for_training(
        training_shots=effective_training_shots,
        params=Mode_params,
        doPlot=doPlot * False,
        save_ext=save_Ext + "_",
        doPerturbation=doPerturbation,
        justLoadGeqdsk=justLoadGeqdsk,
        gEQDSK_files_dir=gEQDSK_files_dir,
        one_of_each_mn=one_of_each_mn,
        prescribed_mn_pairs=prescribed_mn_pairs,
    )

    # Initialize OFT environment
    oft_env = OFT_env(nthreads=ThinCurr_params["n_threads"], abort_callback=True)
    # For each mode, run the simulation
    for mode_param in per_shot_mode_params:
        print(
            f"Running simulation for mode: {mode_param['m']}/{mode_param['n']} at average frequency "
            + f"{np.mean(mode_param['f']):0.2f} Hz"
        )

        # try:
        # Generate synthetic Mirnov signals
        # # # Need sensor list output for sensor names
        gen_synthetic_Mirnov(
            mesh_file=ThinCurr_params["mesh_file"],
            sensor_set=ThinCurr_params["sensor_set"],
            params=mode_param,
            save_ext=ThinCurr_params["save_ext"],
            doSave=doSave,
            archiveExt="training_data/",
            doPlot=doPlot * False,
            plotOnly=False,
            wind_in=ThinCurr_params["wind_in"],
            eta=ThinCurr_params["eta"],
            file_geqdsk=mode_param["file_geqdsk"],
            cmod_shot=ThinCurr_params["cmod_shot"],
            oft_env=oft_env,
            debug=True,
        )

        # Calculate spectrogram
        # Needs to return complex valued spectrogram for each sensor, shape (n_sensors, n_freq, n_time)
        diag, signals, time, out_spect, out_spect_all_cplx, freq, sensor_name = (
            signal_spectrogram_C_Mod(
                shotno=None,  # No shot number for synthetic data
                params=mode_param,
                sensor_name=None,
                sensor_set=ThinCurr_params["sensor_set"],
                pad=spectrogram_params["pad"],
                fft_window=spectrogram_params["fft_window"],
                doSave=doSave,
                save_Ext=save_Ext,
                doPlot=doPlot,
                mesh_file=ThinCurr_params["mesh_file"],
                archiveExt="training_data/",
                tLim=[0, Mode_params["T"]],
                block_reduce=spectrogram_params["block_reduce"],
                plot_reduce=(1, 1),
                filament=True,
                use_rolling_fft=True,
                f_lim=[0, 150],
                cLim=None,
                tScale=1e3,
                doSave_data=True,  # Set frequency limits and color limits
            )
        )

        # Save results in an xarray dataset
        if doSave:
            xarray_file = save_xarray_results(
                output_directory,
                mode_param,
                time,
                freq,
                out_spect_all_cplx,
                ThinCurr_params,
                save_Ext,
                sensor_name,
            )

            if doSave_Training_Data:
                convert_spectrogram_to_training_data(
                    xarray_file, timepoint_index=2, doSave=doSave, doPlot=doPlot
                )
        # except Exception as e:
        #     print(f"Error occurred for mode {mode_param['m']}/{mode_param['n']} at frequency {mode_param['f']} Hz: {e}")
        #     continue


#######################################################################################
######################################################################################
def save_xarray_results(
    output_directory,
    mode_param,
    time,
    freq,
    out_spect,
    ThinCurr_params,
    save_Ext,
    sensor_names,
):
    # Save results in an xarray dataset with dimensions time, frequency, sensor name

    def _sanitize_var_name(name):
        safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name))
        if safe and safe[0].isdigit():
            safe = f"sensor_{safe}"
        return safe

    def _safe_attr_value(value):
        if np.isscalar(value) or isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value)
            if arr.dtype.kind in {"i", "u", "f"}:
                return arr.astype(np.float64).tolist()
            return str(value)
        return str(value)

    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # Create a filename based on mode parameters, and how many files are already in the directory
    current_files = len(os.listdir(output_directory))

    # In general, these are all lists
    m = mode_param["m"]
    n = mode_param["n"]
    f = mode_param["f_Hz"]  # Need frequency in Hz, instead of phase advance
    I_0 = mode_param["I"]
    if np.isscalar(f):
        f_out = f"{float(f) * 1e-3:1.3f}kHz"
    else:
        f_arr = np.asarray(f, dtype=np.float64)
        f_out = f"{np.min(f_arr) * 1e-3:1.3f}-{np.max(f_arr) * 1e-3:1.3f}kHz"
    if type(m) is not list:
        mn_out = "%d-%d" % (m, n)
    else:
        mn_out = (
            "-".join([str(m_) for m_ in m]) + "---" + "-".join([str(n_) for n_ in n])
        )

    # Generate filename
    fName = f"spectrogram_mn_{mn_out}_f_{f_out}_{ThinCurr_params['sensor_set']}_{ThinCurr_params['mesh_file'][:-2]}_{save_Ext}_Count_{current_files + 1}.nc"

    f_save = os.path.join(output_directory, fName)

    # Create a dictionary to hold the DataArrays for each sensor
    data_vars = {}
    freq = np.asarray(freq, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    n_freq = len(freq)
    n_time = len(time)

    for i, sensor_name in enumerate(sensor_names):
        raw_spec = np.asarray(out_spect[i])
        if raw_spec.shape == (n_time, n_freq):
            raw_spec = raw_spec.T
        elif raw_spec.shape != (n_freq, n_time):
            raise ValueError(
                f"Unexpected spectrogram shape {raw_spec.shape} for sensor {sensor_name}. "
                f"Expected {(n_freq, n_time)} or {(n_time, n_freq)}."
            )

        sensor_key = _sanitize_var_name(sensor_name)
        data_vars[sensor_key + "_real"] = (
            ["frequency", "time"],
            np.abs(raw_spec).astype(np.float32, copy=False),
        )
        data_vars[sensor_key + "_imag"] = (
            ["frequency", "time"],
            np.angle(raw_spec).astype(np.float32, copy=False),
        )

    # Add time dependent frequency, amplitude, as extra data_var
    if np.isscalar(f):
        f_modes = [np.full(n_time, float(f), dtype=np.float64)]
    else:
        f_modes = [np.asarray(mode_data, dtype=np.float64) for mode_data in f]

    if np.isscalar(I_0):
        i_modes = [np.full(n_time, float(I_0), dtype=np.float64)]
    else:
        i_modes = [np.asarray(mode_current, dtype=np.float64) for mode_current in I_0]

    if len(f_modes) != len(i_modes):
        raise ValueError(
            f"Mode frequency/current count mismatch: {len(f_modes)} vs {len(i_modes)}"
        )

    for ind, (mode_arr, current_arr) in enumerate(zip(f_modes, i_modes)):
        mode_inds = np.linspace(0, len(mode_arr) - 1, n_time, dtype=int)
        current_inds = np.linspace(0, len(current_arr) - 1, n_time, dtype=int)
        data_vars["F_Mode_%d" % ind] = (
            ["time"],
            mode_arr[mode_inds].astype(np.float32, copy=False),
        )
        data_vars["I_Mode_%d" % ind] = (
            ["time"],
            current_arr[current_inds].astype(np.float32, copy=False),
        )

    # Create a dictionary to hold the coordinates
    coords = {"frequency": freq, "time": time}

    # Create the xarray Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "sampling_frequency": np.float64(1 / (time[1] - time[0])),
            "mode_m": _safe_attr_value(
                m
            ),  # This can be a list, contains information on number of modes
            "mode_n": _safe_attr_value(n),
            "sensor_set": _safe_attr_value(ThinCurr_params["sensor_set"]),
            "mesh_file": _safe_attr_value(ThinCurr_params["mesh_file"]),
        },
    )

    # Save the Dataset to a NetCDF file
    filename = f_save

    resolved_filename = os.path.realpath(filename)
    force_scipy = str(os.environ.get("SYNTH_MIRNOV_FORCE_SCIPY", "0")).lower() in {
        "1",
        "true",
        "yes",
    }
    nfs_home_mount = resolved_filename.startswith("/mnt/home/")

    ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4_CLASSIC", mode="w")

    if (force_scipy or nfs_home_mount) * False:
        reason = "env override" if force_scipy else "NFS mount (/mnt/home)"
        print(f"Using scipy backend directly due to {reason}")
        ds.to_netcdf(filename, engine="scipy", format="NETCDF3_64BIT", mode="w")
        print("Saved with scipy backend")
        print(f"Saved xarray dataset to: {filename}")
        return filename

    netcdf4_failed = False
    try:
        ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4_CLASSIC", mode="w")
        print("Saved with netcdf4 backend")
    except Exception as exc:
        netcdf4_failed = True
        print(f"netcdf4 write failed ({exc}); retrying with HDF5 file locking disabled")
        old_locking = os.environ.get("HDF5_USE_FILE_LOCKING")
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        try:
            ds.to_netcdf(filename, engine="netcdf4", format="NETCDF4_CLASSIC", mode="w")
            netcdf4_failed = False
        except Exception as retry_exc:
            print(f"netcdf4 retry failed ({retry_exc}); trying h5netcdf backend")
        finally:
            if old_locking is None:
                os.environ.pop("HDF5_USE_FILE_LOCKING", None)
            else:
                os.environ["HDF5_USE_FILE_LOCKING"] = old_locking

    if netcdf4_failed:
        try:
            ds.to_netcdf(filename, engine="h5netcdf", mode="w")
            netcdf4_failed = False
            print("Saved with h5netcdf backend")
        except Exception as h5_exc:
            print(
                f"h5netcdf write failed ({h5_exc}); falling back to scipy/NETCDF3_64BIT"
            )

    if netcdf4_failed:
        ds.to_netcdf(filename, engine="scipy", format="NETCDF3_64BIT", mode="w")
        print("Saved with scipy backend")
    print(f"Saved xarray dataset to: {filename}")

    return filename


#####################################################################################
#####################################################################################
# Function to generate mode parameters for synthetic spectrogram generation
# Stored as separate script for clarity and reusability


#####################################################################################
# Function to convert spectrogram output into training data format
def convert_spectrogram_to_training_data(
    xarray_f_name, timepoint_index=2, cLim=None, ylim=[0, 500], doSave="", doPlot=True
):
    """
    Convert spectrogram output into training data format.

    - For a given timepoint: concatenate all the real, imaginary components for each sensor
      into two 2D matrices of size frequency vs number of sensors.

    Returns:
    - None (displays plots)
    """

    # Load in xarray dataset with spectrogram data
    with xr.open_dataset(xarray_f_name) as ds:
        # Get sensor names (excluding 'frequency' and 'time' which are coordinates)
        sensor_names = [var for var in ds.data_vars if "Mode" not in str(var)]
        num_sensors = (
            len(sensor_names) // 2
        )  # Divide by 2 since we have _real and _imag for each sensor

        # Initialize matrices to hold the real and imaginary components
        spect_real = np.zeros((len(ds["frequency"]), num_sensors))
        spect_imag = np.zeros((len(ds["frequency"]), num_sensors))

        # Populate the matrices
        for i in range(num_sensors):
            sensor_name = str(sensor_names[i * 2])[:-5]  # Remove _real or _imag
            spect_real[:, i] = (
                ds[sensor_name + "_real"].isel(time=timepoint_index).values
            )
            spect_imag[:, i] = (
                ds[sensor_name + "_imag"].isel(time=timepoint_index).values
            )

        if doPlot:
            __plot_training_matricies(
                spect_real,
                spect_imag,
                ds,
                timepoint_index,
                sensor_names,
                num_sensors,
                doSave=doSave,
                cLim=cLim,
                ylim=ylim,
            )


############################################33
def __plot_training_matricies(
    spect_real,
    spect_imag,
    ds,
    timepoint_index,
    sensor_names,
    num_sensors,
    cLim=None,
    ylim=None,
    doSave="",
):
    # Plot the matrices
    fig, ax = plt.subplots(
        1, 2, figsize=(12, 6), sharex=True, sharey=True, tight_layout=True
    )

    # Set color limits if provided
    if cLim is not None:
        vmin, vmax = cLim
    else:
        vmin = None
        vmax = None

    im0 = ax[0].imshow(
        spect_real,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=[
            0,
            num_sensors,
            ds["frequency"].min() * 1e-3,
            ds["frequency"].max() * 1e-3,
        ],
    )
    fig.colorbar(im0, ax=ax[0], label=r"$||\frac{d}{dt}B_\theta||$ [T/s]")
    ax[0].set_xlabel("Sensor Index")
    ax[0].set_ylabel(
        "Frequency [kHz], t=%3.3f ms" % (ds["time"][timepoint_index] * 1e3)
    )
    ax[0].set_xticks(np.arange(0, num_sensors, 10))
    ax[0].set_xticklabels(
        [sensor_names[i * 2][:-5] for i in np.arange(0, num_sensors, 10, dtype=int)],
        rotation=90,
    )
    # ax[0].title(f'Real Component (Time Index {timepoint_index})')

    if ylim is not None:
        ax[0].set_ylim(ylim)

    im1 = ax[1].imshow(
        spect_imag / np.pi,
        aspect="auto",
        origin="lower",
        extent=[
            0,
            num_sensors,
            ds["frequency"].min() * 1e-3,
            ds["frequency"].max() * 1e-3,
        ],
    )
    fig.colorbar(
        im1,
        ax=ax[1],
        label=r"$\angle\left[\frac{d}{dt}B_\theta\right]$ [$\pi$-radians]",
    )
    ax[1].set_xlabel("Sensor Index")
    # ax[1].set_ylabel('Frequency')
    ax[1].set_xticklabels(
        [sensor_names[i * 2][:-5] for i in np.arange(0, num_sensors, 10, dtype=int)],
        rotation=90,
    )
    # ax[1].title(f'Imaginary Component (Time Index {timepoint_index})')

    if doSave:
        save_path = os.path.join(
            doSave, f"Training_Matrices_TimeIndex_{timepoint_index}.pdf"
        )
        plt.savefig(save_path)
        print(f"Saved training matrices plot to: {save_path}")

    plt.show()


##################################################################3
if __name__ == "__main__":
    # Example usage
    # output_directory = '../data_output/synthetic_spectrograms/low_m-n_testing/new_Mirnov_set/'
    output_directory = (
        working_directory + "../data_output/synthetic_spectrograms/SPARC/"
    )
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # ThinCurr Eta calculation
    # Bulk resistivities
    Mo = 53.4e-9  # Ohm * m at 20c
    SS = 690e-9  # Ohm * m at 20c
    w_tile_lim = 1.5e-2  # Tile limiter thickness
    w_tile_arm = 1.5e-2 * 1  # Tile extention thickness
    w_vv = 3e-2  # Vacuum vessel thickness
    w_ss = 1e-2  # Support structure thickness
    w_shield = 0.43e-3
    # Surface resistivity: eta/thickness
    # Assume that limiter support structures are 0.6-1.5cm SS, tiles are 1.5cm thick Mo, VV is 3cm thick SS
    # For more accuracy, could break up filaments into different eta values based on position
    eta = f"{SS / w_ss}, {Mo / w_tile_lim}, {SS / w_ss}, {Mo / w_tile_lim}, {SS / w_vv}, {SS / w_ss}, {Mo / w_tile_arm}, {SS / w_shield}"
    # eta = [SS/w_ss, Mo/w_tile_lim, SS/w_ss, Mo/w_tile_lim, SS/w_vv, SS/w_ss, Mo/w_tile_arm, SS/w_shield]

    # Check for job-specific mesh file from SLURM script
    # mesh_file = os.environ.get('MESH_FILE_FOR_JOB', 'C_Mod_ThinCurr_Combined-homology.h5')
    # mesh_file = 'C_Mod_ThinCurr_Combined-homology.h5'
    # sensor_set = 'C_MOD_ALL'

    # SPARC side
    mesh_file = "SPARC_vv_prtmrv_noext.h5"
    # mesh_file ='SPARC_mirnov_plugwest_v2-homology.h5'
    # mesh_file = "vacuum_mesh.h5"
    # sensor_set = "SPARC_BP_MRNV"
    sensor_set = "SPARC_MRNV"

    # This is a guess for now
    eta = "1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5" + ", 2E-5, 2E-5"

    print("USING MESH: ", mesh_file)
    ThinCurr_params = {
        "mesh_file": mesh_file,
        "sensor_set": sensor_set,
        "cmod_shot": 1160930034,  # 1051202011,
        "save_ext": "",
        "doNoise": False,
        "wind_in": "theta",
        "file_geqdsk": "g1051202011.1000",
        "eta": eta,
        "n_threads": 50,
    }

    Mode_params = {
        "dt": 1e-7,
        "T": 1e-3,
        "periods": 2,
        "n_pts": 60,
        "m_pts": 60,
        "R": None,
        "r": None,
        "noise_envelope": 0.01,
        "max_modes": 1,
        "max_m": 16,
        "max_n": np.arange(2, 7, dtype=int),
        "n_threads": 50,
    }

    # spectrogram_params = {'pad':230,'fft_window':230,'block_reduce':(230,10)}

    spectrogram_params = {"pad": 0, "fft_window": None, "block_reduce": (800, 10)}
    save_Ext = "_Synth_low-n_New_Helicity"
    doSave = "../output_plots/low_m-n_spectrograms" * True
    doPlot = False
    training_shots = 33
    doPerturbation = True
    justLoadGeqdsk = True
    gEQDSK_files_dir = "input_data/gEQDSK_files/SPARC/"
    one_of_each_mn = True
    prescribed_mn_pairs = [
        (8, 8),
        (9, 9),
        (14, 9),
        (15, 10),
        (17, 11),
        (18, 12),
        (20, 13),
        (21, 14),
        (23, 15),
        (10, 5),
        (12, 6),
        (14, 7),
        (16, 8),
        (18, 9),
        (20, 10),
        (22, 11),
        (24, 12),
        (26, 13),
        (28, 14),
        (30, 15),
        (9, 3),
        (12, 4),
        (15, 5),
        (18, 6),
        (21, 7),
        (24, 8),
        (27, 9),
        (30, 10),
        (33, 11),
        (36, 12),
        (39, 13),
        (42, 14),
        (45, 15),
    ]

    batch_run_synthetic_spectrogram(
        output_directory=output_directory,
        ThinCurr_params=ThinCurr_params,
        Mode_params=Mode_params,
        spectrogram_params=spectrogram_params,
        save_Ext=save_Ext,
        doSave=doSave,
        doPlot=doPlot,
        training_shots=training_shots,
        doPerturbation=doPerturbation,
        justLoadGeqdsk=justLoadGeqdsk,
        gEQDSK_files_dir=gEQDSK_files_dir,
        one_of_each_mn=one_of_each_mn,
        prescribed_mn_pairs=prescribed_mn_pairs,
    )

    print("Done batch running synthetic spectrograms")
