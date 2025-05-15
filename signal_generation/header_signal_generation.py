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
import h5py
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc,cm
from matplotlib.colors import Normalize
import pyvista
# MDS load may not work on all machines
try:import MDSplus as mds
except:import mdsthin as mds
#pyvista.set_jupyter_backend('static') # Comment to enable interactive PyVista plots
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)


sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release/python/')
sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit_Intel_Compiled/python/')
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors,flux_loop
from OpenFUSIONToolkit.util import build_XDMF, mu0
from OpenFUSIONToolkit.io import histfile
from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh, build_torus_bnorm_grid, build_periodic_mesh, write_periodic_mesh

    
from freeqdsk import geqdsk
import cv2
from scipy.interpolate import make_smoothing_spline
from scipy.special import factorial
from scipy.ndimage import gaussian_filter1d
from fractions import Fraction
import json
from socket import gethostname
server = (gethostname()[:4] == 'orcd') or (gethostname()[:4]=='node')

from Time_dep_Freq import I_KM, F_KM, I_AE, F_AE, F_AE_plot,F_KM_plot

#from rolling_spectrogram import rolling_spectrogram

#####################3
# Add paths
sys.path.append('../signal_analysis/')
sys.path.append('../C-Mod/')
from mirnov_Probe_Geometry import Mirnov_Geometry as Mirnov_Geometry_C_Mod

