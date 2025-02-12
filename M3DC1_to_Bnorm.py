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
try:
    import C1py
    import fio_py
except:pass
import cv2 # necessary for contour detection
from scipy.special import factorial
from scipy.interpolate import make_smoothing_spline
from socket import gethostname

def convert_to_Bnorm(C1file_name,n,npts):
    
    # Pull fields and coordinates from .h5 file
    if gethostname()[:4]=='orcd': # if running on Engaging
        psi,B1,B2,R,Z,rmagx,zmagx,PsiLCFS = get_fields_from_C1(C1file_name,False)
    else:
        psi,B1,B2,R,Z,rmagx,zmagx,PsiLCFS = return_local()
    
    #return psi,B1,B2,R,Z,rmagx,zmagx,PsiLCFS
    # Calculate B-normal for two slices
    B1_norm, R_contour, Z_contour= calculate_normal_vector(psi,R,Z,B1,
                                                           rmagx,zmagx,PsiLCFS)[:3]
    
    B2_norm=calculate_normal_vector(psi,R,Z,B2,rmagx,zmagx,PsiLCFS)[0]
    
    #return psi, B1,B2, B1_norm, B2_norm, R_contour, Z_contour
    # Save in ThinCurr format
    save_Bnorm('C1' if gethostname()[:4]=='orcd' else '',
               R_contour,Z_contour,B1_norm,B2_norm,n,len(R_contour))
    
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

    # C1 file data
    isrc = fio_py.open_source(fio_py.FIO_M3DC1_SOURCE,filename)
    psi_lcfs = fio_py.eval_series(fio_py.get_series(isrc, fio_py.FIO_LCFS_PSI), 0.)
    rmagx = fio_py.eval_series(fio_py.get_series(isrc, fio_py.FIO_MAGAXIS_R),0)
    zmagx = fio_py.eval_series(fio_py.get_series(isrc, fio_py.FIO_MAGAXIS_Z),0)
    if saveNetCDF:
        psi.to_netcdf('psi.nc')
        b_field.to_netcdf('bfield_1.nc')
        b_field2.to_netcdf('b_field2.nc')
    
    return psi.data,b_field.data,b_field2.data,psi.coords['R'].values,\
        psi.coords['Z'].values, rmagx, zmagx, psi_lcfs
        
def return_local():
    psi = xr.load_dataarray('psi.nc')
    b_field_2=xr.load_dataarray('b_field2.nc')
    b_field_1=xr.load_dataarray('bfield_1.nc')
    
    R=b_field_1.coords['R'].values;Z=b_field_1.coords['Z'].values
    psi=psi.data
    b_field_1=b_field_1.data
    b_field_2=b_field_2.data
    
    rmagx=1.88;zmagx=0;psi_LCFS=0.18402695598855184
    return psi, b_field_1, b_field_2, R, Z, rmagx, zmagx, psi_LCFS
def __debug_local_bnorm(doSave=''):
    
    psi,B1,B2,R,Z,rmagx,zmagx,PsiLCFS = return_local()
    #return psi,R,Z,b_field
    b_norm, R_sig, Z_sig,all_norm, contour, all_dr, B  = \
        calculate_normal_vector(psi, R, Z, B1, 
                rmagx, zmagx, PsiLCFS,doPlot=True,doSave=doSave)
    return b_norm, all_norm, contour, all_dr, B, R_sig, Z_sig
def calculate_normal_vector(psi_,R,Z,B,rmagx,zmagx,psi_LCFS,samp_pts=200,
                    doPlot=True,ax=None,fig=None,doSave=''):
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
    all_tang=[]
    all_dr=[]
    R_sig = R[contour[:,0]]
    Z_sig = Z[contour[:,1]]
    
    # Spline smoothing
    R_sig = make_smoothing_spline(np.arange(len(R_sig)), R_sig)(np.arange(len(R_sig)))
    Z_sig = make_smoothing_spline(np.arange(len(Z_sig)), Z_sig)(np.arange(len(Z_sig)))
    for ind,index in enumerate(contour):
        ind_forward = (ind+1) if (ind+1) < samp_pts else 0
        
        # dr = (R[contour[ind_forward,0]]-R[contour[ind-1,0]])/2
        # dz = (Z[contour[ind_forward,1]]-Z[contour[ind-1,1]])/2
        
        diff_op,stencil=__finDiff(R_sig,ind,5,1,True)
        dr = np.dot(R_sig[stencil],diff_op)[0]
        diff_op,stencil=__finDiff(Z_sig,ind,5,1,True)
        dz = np.dot(Z_sig[stencil],diff_op)[0]
                  
        all_dr.append(dr)
        
        fn_norm = lambda vec: np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        tang = (dr,0,dz)
        tang /= fn_norm(tang)
        all_tang.append(tang)
        tor_vec = (0,1,0) # Vec in toroidal direction always orthogonal to LCFS tangency vec
        
        fn_cross = lambda v1, v2: (v1[1]*v2[2]-v1[2]*v2[1],
                                   v1[2]*v2[0]-v1[0]*v2[2],
                                   v1[0]*v2[1]-v1[1]*v2[0])
        norm = fn_cross(tang,tor_vec)
        
        fn_dot = lambda v1, v2: v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        #b_norm.append(fn_dot(B[:,index[0],index[1]],norm))
        b_norm.append(fn_dot(B[:,index[0],index[1]],norm))
        all_norm.append(norm)
    
    b_norm = np.array(b_norm)
    
    if doPlot:  __b_norm_debug_plots(ax,b_norm,B,R,Z,R_sig,Z_sig,all_tang,
                                     all_norm,contour,rmagx,zmagx,doSave)
       
    return b_norm, R_sig, Z_sig,all_norm, contour, all_dr, B

def __b_norm_debug_plots(ax,b_norm,B,R,Z,R_sig,Z_sig,all_tang,all_norm,contour,
                         rmagx,zmagx,doSave,countindex=False):
     
     if ax==None:
         plt.close('B_Norm_Extraction')
         fig,ax=plt.subplots(1,3,tight_layout=True,figsize=(8,3.),
                             num='B_Norm_Extraction')
     norm = colors.Normalize(vmin=min(b_norm),vmax=max(b_norm))
     #ax.contour(R,Z,psi_.T,[1],hold='on',origin='lower')
     vmin=np.min(B);vmax=np.max(B)
     norm_2=colors.Normalize(vmin=vmin,vmax=vmax)
     extent=[R[0],R[-1],Z[0],Z[-1]]
     plotNorm=True
     
     for j in range(3):
         ax[j].imshow(B[j].T,origin='lower',extent=extent,vmin=vmin,vmax=vmax)
         for i in range(len(contour)):
             ax[j].plot(R_sig[i],Z_sig[i],'*',
                     c=cm.get_cmap('plasma')(norm(b_norm[i]) ),alpha=.6)
             vec=all_norm[i] if plotNorm else all_tang[i]
             ax[j].plot([R_sig[i],R_sig[i]+vec[0]*.05],
                     [Z_sig[i],Z_sig[i]+vec[2]*.05],'k-',
                     zorder=5,alpha=.1)
         ax[j].set_xlabel('R [m]')
     ax[0].set_ylabel('Z [m]')
     ax[0].set_title(r'B$_\mathrm{r}$')
     ax[1].set_title(r'B$_\phi$')
     ax[2].set_title(r'B$_\mathrm{z}$')
     fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('plasma')),
              label=r'$||\tilde{\mathrm{B}}_{\hat{n}}||$ [T]',ax=ax[2])
     fig.colorbar(cm.ScalarMappable(norm=norm_2,cmap=cm.get_cmap('viridis')),
              label=r'$||\tilde{\mathrm{B}}_j||$ [T]',ax=ax[1])
     if doSave:fig.savefig(doSave+'B_Contour_Comparison.pdf',transparent=True)
     ############################################
     plt.close('Extraction_Verify')
     fig,ax=plt.subplots(1,3,tight_layout=True,sharex=True,sharey=True,
                         num='Extraction_Verify',figsize=(8,3))
     all_norm = np.array(all_norm)
     v1=np.abs(min(b_norm))*1.5;v2=np.abs(max(b_norm))*1.5
     ylim = [-v1,v1] if v1>v2 else [-v2,v2]
     
     theta = np.arctan2(Z_sig-zmagx,R_sig-rmagx)/np.pi
     argbreak = np.argmax(np.diff(theta))+1
     arginds = np.concatenate((np.arange(argbreak,len(b_norm)),np.arange(0,argbreak)))
     theta = theta[arginds]
     if not countindex:
         b_norm = b_norm[arginds]
         contour = contour[arginds]
         all_norm = all_norm[arginds]         
     xrange = np.arange(len(b_norm)) if countindex else theta
     for i in range(3):
         ax[i].plot(xrange,B[i,contour[:,0],contour[:,1]],label='$B_j$')
         
         ax[i].plot(xrange,b_norm,label=r'$||B_{\hat{n}}||$')
     ax_tmp=[]
     for i in range(3):
         ax_tmp.append(ax[i].twinx())
         ax_tmp[-1].plot(xrange,all_norm[:,i],c=cm.get_cmap('tab10')(2),label=r'$\hat{n}_j$')
         ax_tmp[-1].set_ylim([-1,1])
         if i !=2: ax_tmp[-1].set_yticklabels([])
     ax_tmp[-2].legend(fontsize=8,loc='upper right')
     ax_tmp[-1].set_ylabel(r'$||\hat{n}_j||$ [norm]')
    
     for i in range(3):
         ax[i].grid()
         ax[i].set_xlabel('Contour Index \#' if countindex else r'$\theta$ [$\pi$-rad]')
     ax[1].legend(fontsize=8,loc='upper left')
     ax[0].set_ylabel(r'B$_r$ [T]')
     ax[1].set_ylabel(r'B$_\phi$ [T]')
     ax[2].set_ylabel(r'B$_z$ [T]')
     ax[0].set_ylim(ylim)
     if doSave:fig.savefig(doSave+'B_Vector_Comparison.pdf',transparent=True)
 
 
     plt.show()     
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
def save_Bnorm(C1file_name,contour_R,contour_Z,B1_norm,B2_norm,n,npts):
    with open(C1file_name+'_tCurr_mode.dat','w+') as f:
        f.write('{0} {1}\n'.format(npts,n))
        for ind,r in enumerate(contour_R):
            f.write('{0} {1} {2} {3}\n'.format(r,contour_Z[ind],
                         B1_norm[ind],B2_norm[ind]) )        
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
############################
def __finDiff(signal,ind,order,deriv,periodic=False): # Finite difference stencil
    # Autodetermine derivative order based on length of availible data
    
    if periodic: 
        s = np.arange(-order,order+1) # always true for periodic case
        s_return = np.arange(-order,(order+1) if len(signal)-ind >= (order+1) else
                               len(signal)-ind) +ind
        # If we're close to the end of signal vector
        if len(signal)-ind < (order+1): 
            s_return=np.append(s_return,np.arange(0,order+1-(len(signal)-ind)))
        
    else: 
        s=np.arange(-order if ind >=order else -ind,
                (order+1) if len(signal)-ind>=(order+1) else len(signal)-ind)
        s_return = s+ind
    # Build with automatic s generator: input: order, derivative
    if not len(s)>deriv:raise SyntaxError("Insufficient Points for Derivative")
    S_mat=np.zeros((len(s),len(s)))
    for i in range(len(s)):
        S_mat[i]=s**i
    d_vec=np.zeros((len(s),1))
    d_vec[deriv]=factorial(deriv)
    try:
        return np.matmul(np.linalg.inv(S_mat),d_vec),s_return
    except:
        print(s,s_return,ind,d_vec,S_mat)
        raise SyntaxError
############################
if __name__=='__main__':__debug_local_bnorm()
