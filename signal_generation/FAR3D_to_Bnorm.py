#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:19:50 2025
    FAR-3D to b-norm for effective surface current calculation
    Note: the original eigenvector is in normalized-Psi, we 
    need to first resample it on the original R,Z equilibrium grid

    This assumes that B_r, B_th don't meaningfully transform in direction, magnitude
    The Merezhkin correction is used to "fake" flux expansion on the LFS. 
    This does not account for poloidally asymmetric changes in magnitude
@author: rian
"""
import numpy as np
from scipy.interpolate import griddata
import xarray as xr
try:from M3DC1_to_Bnorm import calculate_normal_vector, save_Bnorm
except:pass
from freeqdsk import geqdsk
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
#####################################################################################
def convert_FAR3D_to_Bnorm(br_file='br_0000',bth_file='bth_0000',\
    m=[9,10,11,12],eqdsk_file='g1051202011.1000',debug=True,lambda_merezhkin=0.4):
    # Pull in FAR3D output eigenvectors, (sin/cos), convery to equilibrium geometry, convery to B-Norm
    
     B_r_RZ_downsamp, B_th_RZ_downsamp, R_downsamp , Z_downsamp  = map_psi_to_equilibrium(br_file, bth_file, m, eqdsk_file,debug, lambda_merezhkin)

    # Restructure r,theta for the cos/sin basis functions into a single B field vector

###################################################################################
def plot_B(br_file='br_0000',bth_file='bth_0000',m=[9,10,11,12],\
           eqdsk_file='g1051202011.1000',data_directory='../M3D-C1_Data/',plot_directory='../output_plots/'):
    # Plot FAR3D Eigenfunction output _without_ accounting for equilibrium shaping

    # Load in Data
    dat_r = np.loadtxt('../M3D-C1_Data/'+br_file,skiprows=1)
    dat_th = np.loadtxt('../M3D-C1_Data/'+bth_file,skiprows=1)
    
    psi_norm = dat_r[:,0]
    dat_r = dat_r[:,1:]
    dat_th = dat_th[:,1:]
    
    # Evolve eigenvector around in a circle, with the appropriate m#
    theta = np.linspace(0,2*np.pi,100)
    def eigenfunc(theta,B,m):
        out = np.zeros((len(B),))
        for ind,m_ in enumerate(m): out += B[:,2*ind]*np.cos(m_*theta) + B[:,2*ind+1]*np.sin(m_*theta)
        return out
    
    r_grid = np.array([eigenfunc(t,dat_r,m) for t in theta]).T
    th_grid = np.array([eigenfunc(t,dat_th,m) for t in theta]).T
    


    plt.close('FAR3D_Eigenfunction_Original')
    fig,ax = plt.subplots(1,2,tight_layout=True,num='FAR3D_Eigenfunction_Original',subplot_kw={'projection': 'polar'},figsize=(4,4))

    #ax[0].contourf(theta,psi_normalized,r_grid)
    #ax[1].contourf(theta,psi_normalized,th_grid)
    ax[0].contourf(theta,psi_norm,r_grid)
    ax[1].contourf(theta,psi_norm,th_grid)
    
    ax[1].set_xlabel(r'B$_\theta(\psi_N)$')
    ax[0].set_xlabel(r'B$_r(\psi_N)$')
    for i in range(2):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([None,0.25,None,None,1])
    
    fig.savefig(plot_directory+'FAR3D_Eigenfunction_Original.pdf',transparent=True)
    plt.show()
########################################################################################
# Get Psi grid, r,z from gEqdsk file
def get_psi_grid(eqdsk_file):
    with open(eqdsk_file,'r') as f: eqdsk_obj = geqdsk.read(f)
    psi = eqdsk_obj['psi']
    psi_normalized = (psi - eqdsk_obj['simagx']) / (eqdsk_obj['sibdry'] - eqdsk_obj['simagx'])

    # get R,Z
    r_min = eqdsk_obj['rleft'] - eqdsk_obj['rmagx']
    r_max = r_min + eqdsk_obj['rdim']
    R = np.linspace(r_min,r_max, eqdsk_obj['nx'])
    #
    z_min = eqdsk_obj['zmid'] - eqdsk_obj['zdim']/2
    z_max = z_min + eqdsk_obj['zdim']
    Z = np.linspace(z_min,z_max, eqdsk_obj['ny']) - eqdsk_obj['zmagx']


    return psi_normalized, R, Z, eqdsk_obj['rmagx'], eqdsk_obj['zmagx']
####################3
def gen_psi_boundary_contour(psi_normalized, R, Z,rmagx,zmagx,psi_LCFS=0.95):
    # Generate contour of the plasma boundary at a given psi_LCFS

    # Contour detection for the desired rational surface radius
    contour,hierarchy=cv2.findContours(np.array(psi_normalized<psi_LCFS,dtype=np.uint8),
                                    cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # Algorithm will also find non-closed contours (e.g. around the magnets)
    try: contour = np.squeeze(contour) # Check if only the main contour was found
    except:
        a_avg=[]#track average minor radial distance to contour
        for s in contour:
            s=np.squeeze(s) # remove extra dimensions
            a_avg.append(np.mean( (R[s[1]]-rmagx)**2+(Z[s[0]]-zmagx)**2))
        # Select contour closest on average to the magnetic center [the main contour]
        contour = np.squeeze( contour[np.argmin(a_avg)] )

    # Fill in indicies inside the contour
    mask = np.zeros(psi_normalized.shape,dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
    inside_inds = np.argwhere(mask==1)

    return inside_inds, contour
############################################################
def interpolate_B(theta_inside,R_inside,Z_inside,dat_B,indices,m,phase=0):
    # Evaluate the FAR3D eigenfunction _at the Psi, theta points corresponding to R,Z_
    def eigenfunc(theta,B,B_indices,m):
        out = np.zeros((len(B_indices),))
        for ind,m_ in enumerate(m): out += B[B_indices,2*ind]*np.cos(m_*theta+phase) + B[B_indices,2*ind+1]*np.sin(m_*theta+phase)
        return out
    B_RZ = eigenfunc(theta_inside,dat_B,indices,m)

    # Downsample the grid from the gEQDSK file
    r_downsamp = np.linspace(R_inside.min(),R_inside.max(),100)
    z_downsamp = np.linspace(Z_inside.min(),Z_inside.max(),100)
    R_downsamp, Z_downsamp = np.meshgrid(r_downsamp, z_downsamp)

    # Resample the FAR3D mode, now at the correct RZ locations, onto the gEQDSK grid
    B_RZ_downsamp = griddata((R_inside, Z_inside), B_RZ, (R_downsamp, Z_downsamp), method='cubic')

    return R_downsamp, Z_downsamp ,B_RZ_downsamp
############

##############################################
def map_psi_to_equilibrium(br_file='br_0000',bth_file='bth_0000',\
    m=[9,10,11,12],eqdsk_file='g1051202011.1000',debug=True,lambda_merezhkin=0.4,\
        data_directory='../M3D-C1_Data/',plot_directory='../output_plots/'):
    
    # Load in FAR3D data
    dat_r = np.loadtxt(data_directory+br_file,skiprows=1)
    dat_th = np.loadtxt(data_directory+bth_file,skiprows=1)
    
    psi_norm = dat_r[:,0]
    dat_r = dat_r[:,1:]
    dat_th = dat_th[:,1:]
    
    # Load in equilibrium data
    psi_normalized,R,Z, rmagx,zmagx = get_psi_grid(data_directory+eqdsk_file)

    # Get plasma boundary contour
    inside_inds,_ = gen_psi_boundary_contour(psi_normalized, R, Z,rmagx,zmagx,psi_LCFS=0.95)

    # Regrid R,Z, calculate theta from the magnetic axis
    R_grid, Z_grid = np.meshgrid(R,Z,indexing='ij')
    theta = np.atan2(Z_grid,R_grid) % (2*np.pi) # theta in [0,2pi], atan2 originally in [-pi,pi]
    # If desired, we can use the Merezhkin formulation to approximate LFS/HFS asymmetry
    theta = theta - lambda_merezhkin*np.sin(theta)

    # Extract only the points inside the contour
    # These are the R,Z points from the gEQDSK file which we want to evaluate the FAR3D mode over
    psi_norm_inside = psi_normalized[inside_inds[:,0],inside_inds[:,1]]
    R_inside = R_grid[inside_inds[:,0],inside_inds[:,1]]
    Z_inside = Z_grid[inside_inds[:,0],inside_inds[:,1]]
    theta_inside = theta[inside_inds[:,0],inside_inds[:,1]]
    #if debug: plt.plot(R_inside+rmagx,Z_inside+zmagx,'*',label='Psi-Norm Points',alpha=.2)

    ## Simply evaluate the Br, Bth functions at the apropriate psi,theta points, cooresponding to R,Z
    # Compute absolute differences for all test values at once
    diff = np.abs(psi_norm[None, :] - psi_norm_inside[:, None])  # shape: (M, N)
    # Find the index of the minimum difference for each test value
    indices = np.argmin(diff, axis=1)  # shape: (M,)

    # Interpolate the FAR3D Mode on the equilibrium Psi grid
    R_downsamp,Z_downsamp, B_r_RZ_downsamp_cos = interpolate_B(theta_inside,R_inside,Z_inside,dat_r[:,0:],indices,m)
    _,_, B_r_RZ_downsamp_sin = interpolate_B(theta_inside,R_inside,Z_inside,dat_r[:,0:],indices,m,np.pi/2)
    _,_, B_th_RZ_downsamp_cos = interpolate_B(theta_inside,R_inside,Z_inside,dat_th[:,0:],indices,m)
    _,_, B_th_RZ_downsamp_sin = interpolate_B(theta_inside,R_inside,Z_inside,dat_th[:,0:],indices,m,np.pi/2)

    ###########################
    # Plot the results
    if debug:
        plt.close('FAR3D_Eigenfunction')
        fig,ax = plt.subplots(1,2,tight_layout=True,num='FAR3D_Eigenfunction',figsize=(4,4),sharex=True,sharey=True)
        ax[0].contourf(R_downsamp+rmagx,Z_downsamp+zmagx,B_r_RZ_downsamp_cos,levels=30)
        ax[1].contourf(R_downsamp+rmagx,Z_downsamp+zmagx,B_th_RZ_downsamp_cos,levels=30)
        ax[0].set_title(r'B$_r(\psi_N)$')
        ax[1].set_title(r'B$_\theta(\psi_N)$')
        ax[0].set_ylabel(r'Z [m]')
        for i in range(2):ax[i].set_xlabel(r'R [m]')
        fig.savefig(plot_directory+'FAR3D_Eigenfunction_Equilibrium.pdf',transparent=True)
        plt.show()

    print('halt')

    return B_r_RZ_downsamp_cos,B_r_RZ_downsamp_sin, B_th_RZ_downsamp_cos,B_th_RZ_downsamp_sin,\
          R_downsamp + rmagx, Z_downsamp + zmagx
    
if __name__ == '__main__':
    plot_B()
    map_psi_to_equilibrium()
    print('Finished')

