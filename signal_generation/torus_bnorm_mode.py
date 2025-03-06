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

write_periodic_mesh('thincurr_mode.h5', r_mesh, lc_mesh+1, np.ones((lc_mesh.shape[0],)),
                    tnodeset, pnodesets, pmap=r_map, nfp=nfp, include_closures=True)


tw_mode = ThinCurr(nthreads=4)
tw_mode.setup_model(mesh_file='thincurr_mode.h5')
tw_mode.setup_io(basepath='plasma/')

tw_mode.compute_Lmat()
# Condense model to single mode period (necessary for mesh periodicity reasons)
if nfp > 1:
    nelems_new = tw_mode.Lmat.shape[0]-nfp+1
    Lmat_new = np.zeros((nelems_new,nelems_new))
    Lmat_new[:-1,:-1] = tw_mode.Lmat[:-nfp,:-nfp]
    Lmat_new[:-1,-1] = tw_mode.Lmat[:-nfp,-nfp:].sum(axis=1)
    Lmat_new[-1,:-1] = tw_mode.Lmat[-nfp:,:-nfp].sum(axis=0)
    Lmat_new[-1,-1] = tw_mode.Lmat[-nfp:,-nfp:].sum(axis=None)
else:
    Lmat_new = tw_mode.Lmat
# Get inverse
Linv = np.linalg.inv(Lmat_new)

bnorm_flat = bnorm.reshape((2,bnorm.shape[1]*bnorm.shape[2]))
# Get surface flux from normal field (e.g. B-field X mesh vertex area va)
flux_flat = bnorm_flat.copy()
flux_flat[0,r_map] = tw_mode.scale_va(bnorm_flat[0,r_map])
flux_flat[1,r_map] = tw_mode.scale_va(bnorm_flat[1,r_map])
if nfp > 1:
    tw_mode.save_scalar(bnorm_flat[0,r_map],'Bn_c')
    tw_mode.save_scalar(bnorm_flat[1,r_map],'Bn_s')
    output = np.zeros((2,nelems_new+nfp-1))
    for j in range(2):
        output[j,:nelems_new] = np.dot(Linv,np.r_[flux_flat[j,1:-bnorm.shape[2]],0.0,0.0])
        output[j,-nfp+1:] = output[j,-nfp]
else:
    tw_mode.save_scalar(bnorm_flat[0,:],'Bn_c')
    tw_mode.save_scalar(bnorm_flat[1,:],'Bn_s')
    output = np.zeros((2,tw_mode.Lmat.shape[0]))
    for j in range(2):
        output[j,:] = np.dot(Linv,np.r_[flux_flat[j,1:],0.0,0.0])

print(r_mesh.shape,lc_mesh.shape,tw_mode.Lmat.shape)
print(output.shape,bnorm.shape,tw_mode.np)
tw_mode.save_current(output[0,:],'Jc')
tw_mode.save_current(output[1,:],'Js')
print('Preparing XMDF')
tw_mode.build_XDMF()
'''
fig, ax = plt.subplots(2,2)
if nfp > 1:
    ax[0,0].contour(np.r_[0.0,output[0,:-nfp-1]].reshape((nphi-1,ntheta)).transpose(),10)
    ax[0,1].contour(np.r_[0.0,output[1,:-nfp-1]].reshape((nphi-1,ntheta)).transpose(),10)
else:
    ax[0,0].contour(np.r_[0.0,output[0,:-nfp-1]].reshape((nphi,ntheta)).transpose(),10)
    ax[0,1].contour(np.r_[0.0,output[1,:-nfp-1]].reshape((nphi,ntheta)).transpose(),10)
ax[1,0].contour(bnorm[0,:,:].transpose(),10)
_ = ax[1,1].contour(bnorm[1,:,:].transpose(),10)
'''
with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
    h5_file.create_dataset('thincurr/driver', data=output, dtype='f8')