#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:04:59 2024
    header file for synthetic mirnov data
@author: rian
"""

import struct
import sys
import importlib.util
import os
from os import getcwd
import subprocess

# Specific module load order to circumvent HDF5 issues with xarray
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import xarray as xr
import h5py
    
from freeqdsk import geqdsk
import cv2
from scipy.interpolate import make_smoothing_spline, BSpline
from scipy.special import factorial
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

from fractions import Fraction
import json
from socket import gethostname

from Time_dep_Freq import I_KM, F_KM, I_AE, F_AE, F_AE_plot,F_KM_plot, gen_coupled_freq,debug_mode_frequency_plot


#####################3
# Add paths
working_directory = os.environ.get('SCRIPT_DIR', getcwd()+'/')
print('Working Directory: ', working_directory)
sys.path.append(working_directory+'../signal_analysis/')
sys.path.append(working_directory+'../C-Mod/')
from mirnov_Probe_Geometry import Mirnov_Geometry as Mirnov_Geometry_C_Mod



try:
    import vtk # potentially important for LaTex rendering in PyVista
    import pyvista
except: pass
# MDS load may not work on all machines
try:import MDSplus as mds
except:
    try:import mdsthin as mds
    except: mds = None


# sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release_sched_mit_psfc_r8/python/')
#sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Intel_Compiled/python/') # This one
# sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Updated/src/python/')
# Updated OFT:
# Updated binary
#sys.path.append('/home/rianc/Downloads/OpenFUSIONToolkit_v1.0 (2).0-beta6-Ubuntu_22_04-GNU-x86_64/OpenFUSIONToolkit_v1.0.0-beta6-Linux-GNU-x86_64/python/')
# Updated git repo source:
# sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit/build_release/python/')

sys.path.append(os.getenv('OFT_ROOTPATH'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors,flux_loop
from OpenFUSIONToolkit.util import mu0
from OpenFUSIONToolkit.io import histfile
from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh,\
    build_torus_bnorm_grid, ThinCurr_periodic_toroid


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc,cm
from matplotlib.colors import Normalize
plt.ion()
#pyvista.set_jupyter_backend('static') # Comment to enable interactive PyVista plots
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
import matplotlib;
try:
    matplotlib.use('TkAgg') # Use TkAgg backend for plotting
    plt.ion()
except:pass # TkAgg can't be assigned in headless operations

################################################################################################
####################
# Sbatch environment and error handling
import shutil
import time
import errno
import ctypes

server = (gethostname()[:4] == 'orcd') or (gethostname()[:4]=='node')
# Save files in job specific folders if possible [necessary for multi-node operations]
j_id =  os.environ.get('SLURM_WORKING_FOLDER',os.getcwd()+'/')
working_directory = os.environ.get('SCRIPT_DIR', j_id)
#from rolling_spectrogram import rolling_spectrogram

###########################
def _flush_process_stdio():
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass

####################################################
def _copy_file_with_retries(src, dst, max_retries=8, remove_src=False, debug=True):
    """Robust file copy helper for NFS/SLURM environments.

    Uses copy-to-temp + atomic replace to avoid partial destination files
    and retries transient filesystem errors (including ESTALE).
    """

    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
    tmp_dst = dst_abs + ".tmp"

    transient_errnos = {errno.ESTALE, errno.EIO, errno.EBUSY}

    for attempt in range(max_retries):
        try:
            with open(src_abs, "rb") as f_src, open(tmp_dst, "wb") as f_dst:
                shutil.copyfileobj(f_src, f_dst, length=1024 * 1024)
                f_dst.flush()
                os.fsync(f_dst.fileno())
            os.replace(tmp_dst, dst_abs)

            if remove_src:
                try:
                    os.unlink(src_abs)
                except OSError as exc:
                    if exc.errno not in {errno.ENOENT, errno.ESTALE}:
                        raise
            return dst_abs
        except OSError as exc:
            try:
                if os.path.exists(tmp_dst):
                    os.unlink(tmp_dst)
            except OSError:
                pass

            should_retry = (
                exc.errno in transient_errnos
                or (exc.errno == errno.ENOENT and attempt < max_retries - 1)
            )
            if should_retry and attempt < max_retries - 1:
                if debug:
                    print(
                        f"Transient file error copying {src_abs} -> {dst_abs}: {exc}. "
                        + f"Retrying ({attempt + 1}/{max_retries})...",
                        flush=True,
                    )
                time.sleep(2 ** min(attempt, 4))
                continue
            raise

    raise RuntimeError(
        f"Failed to copy {src_abs} -> {dst_abs} after {max_retries} attempts"
    )

#####################################################################
################################################################################################
def NFS_Estale_aware_copy(_src, _dst, max_retries=8, remove_src=False, debug=False):
    # Stage output history into input_data/ with retry logic for NFS ESTALE.
    try:
        new_file_path = _copy_file_with_retries(
            _src, _dst, max_retries=8, remove_src=True, debug=debug
        )
    except OSError as exc:
        if exc.errno == errno.ESTALE:
            print(
                f"Persistent ESTALE when staging {_src} -> {_dst}; falling back to source path",
                flush=True,
            )
            new_file_path = _src
        else:
            raise
    return new_file_path