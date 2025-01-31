#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 15:03:23 2025
 gen torus b-norm surface for freq/time dep signal evolution
 
@author: rian
"""

import struct
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2


import sys;sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release/python/')
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh, build_torus_bnorm_grid, build_periodic_mesh, write_periodic_mesh
from OpenFUSIONToolkit.util import build_XDMF

def create_circular_bnorm(filename,R0,Z0,a,n,m,npts=200):
    theta_vals = np.linspace(0.0,2*np.pi,npts,endpoint=False)
    with open(filename,'w+') as fid:
        fid.write('{0} {1}\n'.format(npts,n))
        for theta in theta_vals:
            fid.write('{0} {1} {2} {3}\n'.format(
                R0+a*np.cos(theta),
                Z0+a*np.sin(theta),
                np.cos(m*theta),
                np.sin(m*theta)
            ))
# Create n=2, m=3 mode
create_circular_bnorm('tCurr_mode.dat',1.0,0.0,0.4,2,3)

ntheta = 40
nphi = 80
r_grid, bnorm, nfp = build_torus_bnorm_grid('tCurr_mode.dat',ntheta,nphi,resample_type='theta',use_spline=False)
lc_mesh, r_mesh, tnodeset, pnodesets, r_map = build_periodic_mesh(r_grid,nfp)