#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:22:03 2025
    Header file for C-Mod data access/plotting
@author: rianc
"""


import numpy as np
import matplotlib.pyplot as plt
import mdsthin as mds # Needs to be separately installed through pip
from matplotlib.colors import Normalize
from matplotlib import rc,cm
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)

import sys
sys.path.append('../signal_analysis/')

from rolling_spectrogram import rolling_spectrogram
