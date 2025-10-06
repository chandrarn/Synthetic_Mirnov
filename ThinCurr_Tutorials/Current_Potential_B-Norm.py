# ThinCurr tutorial for periodic mesh testing
# Rian C. O' Brien 2024
# Run from command line: python3 Synthetic_Mirnov.py
# or from IDE

# Load libraries
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
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"
# Updated git repo source:
sys.path.append('/home/rianc/Documents/OpenFUSIONToolkit/build_release/python/')
# thincurr_python_path = os.getenv('OFT_ROOTPATH')
# if thincurr_python_path is not None:
#     sys.path.append(os.path.join(thincurr_python_path,'python'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.meshing import build_torus_bnorm_grid, ThinCurr_periodic_toroid

# Build B-norm
# def create_circular_bnorm(filename,R0,Z0,a,n,m,npts=200):
#     theta_vals = np.linspace(0.0,2*np.pi,npts,endpoint=False)
#     with open(filename,'w+') as fid:
#         fid.write('{0} {1}\n'.format(npts,n))
#         for theta in theta_vals:
#             fid.write('{0} {1} {2} {3}\n'.format(
#                 R0+a*np.cos(theta),
#                 Z0+a*np.sin(theta),
#                 np.cos(m*theta),
#                 np.sin(m*theta)
#             ))
# # Create n=2, m=3 mode
# create_circular_bnorm('tCurr_mode.dat',1.0,0.0,0.4,3,4)

# Generate Mesh
ntheta = 57#46#23
nphi = 30
r_grid, bnorm, nfp = build_torus_bnorm_grid('../signal_generation/input_data/tCurr_mode.dat',ntheta,nphi,resample_type='theta',use_spline=False)

plasma_mode = ThinCurr_periodic_toroid(r_grid,nfp,ntheta,nphi)

# dat = np.loadtxt('../signal_generation/input_data/tCurr_mode.dat',skiprows=1)
# plt.figure();plt.plot(dat[:,0],dat[:,1],'*');plt.grid();#plt.show()

# fig = plt.figure(figsize=(12,5))
# plasma_mode.plot_mesh(fig)
# plt.show()

plasma_mode.write_to_file('thincurr_mode.h5')


myOFT = OFT_env(nthreads=8)
tw_mode = ThinCurr(myOFT)
tw_mode.setup_model(mesh_file='thincurr_mode.h5')
tw_mode.setup_io(basepath='plasma/')

# Compute self-inductance for equivalent surface current
tw_mode.compute_Lmat()
Lmat_new = plasma_mode.condense_matrix(tw_mode.Lmat)
# Get inverse
Linv = np.linalg.inv(Lmat_new)

# Compute equivalent current
bnorm_flat = bnorm.reshape((2,bnorm.shape[1]*bnorm.shape[2]))
# Get surface flux from normal field
flux_flat = bnorm_flat.copy()
 
flux_flat[0,plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[0,plasma_mode.r_map])
flux_flat[1,plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[1,plasma_mode.r_map])
tw_mode.save_scalar(bnorm_flat[0,plasma_mode.r_map],'Bn_c')
tw_mode.save_scalar(bnorm_flat[1,plasma_mode.r_map],'Bn_s')
output_full = np.zeros((2,tw_mode.nelems))
output_unique = np.zeros((2,Linv.shape[0]))
for j in range(2):
    output_unique[j,:] = np.dot(Linv,plasma_mode.nodes_to_unique(flux_flat[j,:]))
    output_full[j,:] = plasma_mode.expand_vector(output_unique[j,:])
 
tw_mode.save_current(output_full[0,:],'Jc')
tw_mode.save_current(output_full[1,:],'Js')
_ = tw_mode.build_XDMF()

fig, ax = plt.subplots(2,2)
ax[0,0].contour(plasma_mode.unique_to_nodes_2D(output_unique[0,:]),10)
ax[0,1].contour(plasma_mode.unique_to_nodes_2D(output_unique[1,:]),10)
ax[1,0].contour(bnorm[0,:,:].transpose(),10)
_ = ax[1,1].contour(bnorm[1,:,:].transpose(),10)
plt.show()


with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
    h5_file.create_dataset('thincurr/driver', data=output_full, dtype='f8')


print('Done')