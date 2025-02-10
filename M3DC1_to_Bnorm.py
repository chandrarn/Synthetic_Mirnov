#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:49:17 2025
     Helper functino for translating betweeing psi/b at one phase and B-norm
     # Steps: pull B, psi
     Pull indicies of boundary for B
     caluclate normal vector: tangency vector on LCFS X toroidal vector
     Take dot product to exgtract b-norm
     save B-norm, coordinates
     Run ThinCurr helper function to create mode
     
     Note: Might only work for linear M3D-C1 runs (matrix dimension issues)
     - Initialize thincurr at nonzer current
     NEed time dep version of bnorm otherwise faking it with two basies
@author: rian
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'size':11})
rc('text', usetex=True)
import xarray as xr
from sys import path;path.append('/orcd/home/002/rianc/')
try:import C1py
except:pass
import cv2 # necessary for contour detection

def convert_to_Bnorm(C1file_name,n,npts):
    
    # Pull fields and coordinates from .h5 file
    psi,B1,B2,R,Z = get_fields_from_C1(C1file_name,False)
    
    # Calculate B-normal for two slices
    B1_norm= calculate_normal_vector(psi,R,Z,B1)[0]
    B2_norm,_,contour=calculate_normal_vector(psi,R,Z,B2)
    
    # Save in ThinCurr format
    save_Bnorm(C1file_name,contour,B1_norm,B2_norm,n,npts)
    
def get_fields_from_C1(filename,saveNetCDF=True):
    # Get psi grid, B grid, R,Z coords
    
    # Get Psi
    psi = C1py.read_field('psi', slice=-1, filename=filename, points=200,
                         rrange=None, zrange=None, iequil=1)
    
    
    b_field = C1py.read_field('bfield', slice=[0,1], filename=filename, points=200,
                        rrange=None, zrange=None, iequil=None,idiff=True)
    b_field2 = C1py.read_field('bfield', slice=[0,1], filename=filename, points=200,
                        rrange=None, zrange=None, iequil=None,phi=90,idiff=True)
 
    R=b_field.coords['R'].values;Z=b_field.coords['Z']

    if saveNetCDF:
        psi.to_netcdf('psi.nc')
        b_field.to_netcdf('bfield_1.nc')
        b_field2.to_netcdf('b_field2.nc')
    
    return psi.data,b_field.data,b_field2.data,psi.coords['R'].values,\
        psi.coords['Z'].values
        
    
def calculate_normal_vector(psi_,R,Z,B,rmagx,zmagx,psi_LCFS,samp_pts=200,doPlot=False,ax=None,fig=None):
    # Get LCFS coordinates
    psi=np.array(psi_.T>=psi_LCFS*1.05,dtype=np.uint8)
    contour,hierarchy=cv2.findContours(psi,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #return contour
    # Algorithm will find non-closed contours (e.g. around the magnets)
    try: contour = np.squeeze(contour) # Check if only the main contour was found
    except:
        a_avg=[]#track average minor radial distance to contour
        for s in contour:
            s=np.squeeze(s).T # remove extra dimensions
            a_avg.append(np.mean( (R[s[1]]-rmagx)**2+(Z[s[0]]-zmagx)**2))
        # Select contour closested on average to the magnetic center
        
        #print(a_avg)
        contour = np.squeeze( contour[np.argmin(a_avg)] )
    
    # Calculate tangency vector
    deriv = np.zeros(contour.shape)
    b_norm = []
    all_norm=[]
    for ind,index in enumerate(contour):
        ind_forward = (ind+1) if (ind+1) < samp_pts else 0
        dr = (R[contour[ind_forward,0]]-R[contour[ind-1,0]])/2
        dz = (Z[contour[ind_forward,1]]-Z[contour[ind-1,1]])/2
        
        fn_norm = lambda vec: np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        tang = (dr,0,dz)
        tang /= fn_norm(tang)
        tor_vec = (0,1,0) # Vec in toroidal direction always orthogonal to LCFS tangency vec
        
        fn_cross = lambda v1, v2: (v1[1]*v2[2]-v1[2]*v2[1],
                                   v1[2]*v2[0]-v1[0]*v2[2],
                                   v1[0]*v2[1]-v1[1]*v2[0])
        norm = fn_cross(tang,tor_vec)
        
        fn_dot = lambda v1, v2: v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        #b_norm.append(fn_dot(B[:,index[0],index[1]],norm))
        b_norm.append(fn_dot(B[:,index[1],index[0]],norm))
        all_norm.append(norm)
    
    if doPlot:
        if ax==None:
            fig,ax=plt.subplots(1,3,tight_layout=True,figsize=(6,3.2))
        norm = colors.Normalize(vmin=min(b_norm),vmax=max(b_norm))
        #ax.contour(R,Z,psi_.T,[1],hold='on',origin='lower')
        vmin=np.min(B);vmax=np.max(B)
        extent=[R[0],R[-1],Z[0],Z[-1]]
        for j in range(3):
            ax[j].imshow(B[j].T,origin='lower',extent=extent,vmin=vmin,vmax=vmax)
            for i in range(len(contour)):
                ax[j].plot(R[contour[i,0]],Z[contour[i,1]],'*',
                        c=cm.get_cmap('plasma')(norm(b_norm[i]) ),alpha=.1)
                ax[j].plot([R[contour[i,0]],R[contour[i,0]]+all_norm[i][0]*.05],
                        [Z[contour[i,1]],Z[contour[i,1]]+all_norm[i][2]*.05],'k-',
                        zorder=5)
            ax[j].set_xlabel('R [m]')
        ax[0].set_ylabel('Z [m]')
        ax[0].set_title(r'B$_\mathrm{r}$')
        ax[1].set_title(r'B$_\phi$')
        ax[2].set_title(r'B$_\mathrm{z}$')
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('plasma')),
                     label=r'$\tilde{\mathrm{B}}_\perp$ [?]',ax=ax[2])
        plt.show()
    return b_norm,all_norm, contour
        
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

##########################################
def save_Bnorm(C1file_name,contour,B1_norm,B2_norm,n,npts):
    with open(C1file_name[:-2]+'_tCurr_mode.dat','w+') as f:
        f.write('{0} {1}\n'.format(npts,n))
        for ind,z in contour:
            f.write('{0} {1} {2} {3}\n').format(contour[ind,0],
                         z,B1_norm[ind],B2_norm[ind])        
##########################################
def __plot_time_series(t_points=5):
    fig,ax=plt.subplots(2,2,tight_layout=True)
    psi=xr.open_dataset('psi.nc')
    psi_=psi.to_array().data.squeeze()
    R=psi.coords['R'].values
    Z=psi.coords['Z'].values
    for i in np.arange(1,5):
        b_field=xr.open_dataset('B_Field_%d.nc'%i)
        B=b_field.to_array().data.squeeze()
        calculate_normal_vector(psi_,R,Z,B,samp_pts=200,doPlot=True,
                                ax=ax[np.unravel_index(i-1,(2,2))],fig=fig)
############################
def __plot_time_series_im(t_points=5,B_index=0):
    label=['R','Phi','Z']
    plt.close('B_%s'%label[B_index])
    fig,ax=plt.subplots(2,2,tight_layout=True,sharex=True,sharey=True,
                        num='B_%s'%label[B_index])
    psi=xr.open_dataset('psi.nc')
    psi_=psi.to_array().data.squeeze()
    R=psi.coords['R'].values
    Z=psi.coords['Z'].values
    extent=[R[0],R[-1],Z[0],Z[-1]]
    for i in np.arange(1,5):
        b_field=xr.open_dataset('B_Field_%d.nc'%i)
        B=b_field.to_array().data.squeeze()
        ax[np.unravel_index(i-1,(2,2))].imshow(B[B_index].T,origin='lower',extent=extent)
        ax[np.unravel_index(i-1,(2,2))].contour(psi_.T,[1],hold='on',
                                                origin='lower',extent=extent,alpha=.7)
        norm = colors.Normalize(vmin=np.min(B[B_index]),vmax=np.max(B[B_index]))
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('viridis')),
                     label='$\mathrm{B_%s}$ [?]'%label[B_index],ax=ax[np.unravel_index(i-1,(2,2))])
    for i in range(2):
        ax[1,i].set_xlabel('R [m]')
        ax[i,0].set_ylabel('Z [m]')
    plt.show()
