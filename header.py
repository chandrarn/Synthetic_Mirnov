#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:04:59 2024
    header file for synthetic mirnov data
@author: rian
"""

import struct
import sys
import os
import h5py
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc,cm
import pyvista
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
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors,flux_loop
from OpenFUSIONToolkit.util import build_XDMF, mu0
from OpenFUSIONToolkit.io import histfile

from freeqdsk import geqdsk
import cv2
from scipy.interpolate import make_smoothing_spline
from scipy.special import factorial
import json
