#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Dec 17 12:04:59 2024
    header file for synthetic mirnov data
@author: rian
"""

import sys
import os


# MDS load may not work on all machines
try:
    import MDSplus as mds
except Exception:
    try:
        import mdsthin as mds
    except Exception:
        mds = None


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

try:
    matplotlib.use("TkAgg")  # Use TkAgg backend for plotting
    plt.ion()
except Exception:
    pass  # TkAgg can't be assigned in headless operations


# print(getcwd())
# sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release_sched_mit_psfc_r8/python/')
# sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Intel_Compiled/python/') # This one
# sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Updated/src/python/')
# Updated OFT:
# Updated binary

# sys.path.append('/home/rianc/Downloads/OpenFUSIONToolkit_v1.0 (2).0-beta6-Ubuntu_22_04-GNU-x86_64/OpenFUSIONToolkit_v1.0.0-beta6-Linux-GNU-x86_64/python/')
# Updated git repo source:
# sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit/build_release/python/')

# Specific module load order to circumvent HDF5 issues with xarray

from socket import gethostname

server = (gethostname()[:4] == "orcd") or (gethostname()[:4] == "node")

# Save files in job specific folders if possible [necessary for multi-node operations]
j_id = os.environ.get("SLURM_WORKING_FOLDER", os.getcwd() + "/")
working_directory = os.environ.get("SCRIPT_DIR", j_id)


# from rolling_spectrogram import rolling_spectrogram

# pyvista.set_jupyter_backend('static') # Comment to enable interactive PyVista plots
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markeredgewidth"] = 2
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("font", **{"size": 11})
rc("text", usetex=True)
plt.ion()

#####################3
# Add paths
print("Working Directory: ", working_directory)
sys.path.append(working_directory + "../signal_analysis/")
sys.path.append(working_directory + "../C-Mod/")
