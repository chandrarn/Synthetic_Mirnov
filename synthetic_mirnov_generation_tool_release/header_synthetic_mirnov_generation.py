# General libraries
import numpy as np
import xarray as xr
import json
import os
import sys

# Plasma filament generation
from freeqdsk import geqdsk
import cv2
from fractions import Fraction

# TARS Equilibrium field tool for filament generation
tars_python_path = os.getenv('TARS_ROOTPATH')
if tars_python_path is not None:
    sys.path.append(tars_python_path)
else: raise ValueError('Please set TARS_ROOTPATH environment variable to point to OpenFUSIONToolkit build directory')
from tars.filaments import EquilibriumFilament, TraceType
from tars.magnetic_field import EquilibriumField


# For debug plots
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc,cm
from matplotlib.colors import Normalize
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
import matplotlib;matplotlib.use('TkAgg') # Use TkAgg backend for plotting
plt.ion()
import pyvista


# ThinCurr and OFT imports
thincurr_python_path = os.getenv('OFT_ROOTPATH')
if thincurr_python_path is not None:
    sys.path.append(thincurr_python_path)
else: raise ValueError('Please set OFT_ROOTPATH environment variable to point to OpenFUSIONToolkit build directory')
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors,flux_loop
from OpenFUSIONToolkit.util import mu0
from OpenFUSIONToolkit.io import histfile
from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh,\
    build_torus_bnorm_grid, ThinCurr_periodic_toroid, build_periodic_mesh,\
    write_periodic_mesh

# Bulk resistivities
Mo = 53.4e-9 # Ohm * m at 20c
SS = 690e-9 # Ohm * m at 20c
w_tile_lim = 1.5e-2  # Tile limiter thickness
w_tile_arm = 1.5e-2 *1 # Tile extention thickness
w_vv = 3e-2 # Vacuum vessel thickness
w_ss = 1e-2  # Support structure thickness
w_shield = 0.43e-3 

