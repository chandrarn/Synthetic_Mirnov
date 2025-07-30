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
from scipy.interpolate import griddata, make_smoothing_spline
import xarray as xr
try:from M3DC1_to_Bnorm import save_Bnorm, __finDiff
except:pass
from socket import gethostname
from freeqdsk import geqdsk
import cv2
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib import rc,cm
matplotlib.use('TkAgg')

#####################################################################################
def convert_FAR3D_to_Bnorm(br_file='br_0000',bth_file='bth_0000',psi_contour_select=0.8,\
    m=[9,10,11,12],n=10,eqdsk_file='g1051202011.1000',debug=True,lambda_merezhkin=0.4,doSave='../output_plots/'):
    # Pull in FAR3D output eigenvectors, (sin/cos), convery to equilibrium geometry, convery to B-Norm
    
    B_r_RZ_downsamp_cos,B_r_RZ_downsamp_sin, B_th_RZ_downsamp_cos,B_th_RZ_downsamp_sin, R_downsamp ,\
          Z_downsamp,psi_normalized,psi_norm_downsamp  = \
        map_psi_to_equilibrium(br_file, bth_file, m, eqdsk_file,debug, lambda_merezhkin)

    
    # B_theta is orthogonal to B_norm, we need to calculate B_r(x,y,phi) * r_n_hat (x,y,z): B_norm = n_norm * r_n_hat * ||B_r||
    b_norm_cos, R_sig, Z_sig = calc_n_hat_b_norm_coords_geqdsk(eqdsk_file,R_downsamp,Z_downsamp,max(m),\
                                        n,psi_norm_downsamp,B_r_RZ_downsamp_cos,psi_contour_select,doSave=doSave)
    b_norm_sin, _, _ = calc_n_hat_b_norm_coords_geqdsk(eqdsk_file,R_downsamp,Z_downsamp,max(m),n,\
                                                       psi_norm_downsamp,B_r_RZ_downsamp_sin,psi_contour_select,doSave=doSave)

    # # Save B norm in ThinCurr format
    save_Bnorm('FAR3D' if (gethostname()[:4]=='orcd' or gethostname()[:4]=='node') \
            else '',  R_sig,Z_sig,b_norm_cos,b_norm_sin,n,len(Z_sig))
    print('Finished conversion of FAR3D eigenfunctions %s to ThinCurr effective surface current'%br_file)
#####################################################################################
def calc_n_hat_b_norm_coords_geqdsk(file_geqdsk,R,Z,m,n,psi_norm_downsamp,B_r,\
                                    psi_contour_select,doSave,debug=True,save_ext=''):
    # Calculate B-normal for the FAR3D eigenmode, on some select contour of Psi, in machine coordinates

    # Using geqdsk equilibrium to locate flux surfaces
    # Load eqdsk
    with open('input_data/'+file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
    
    # get q(psi(r,z))
    psi_lin = np.linspace(eqdsk.simagx,eqdsk.sibdry,eqdsk.nx)
    # Convert psi_linear to psi_norm, for extrapolation into psi_norm_downsample
    psi_lin = (psi_lin-eqdsk.simagx)/(eqdsk.sibdry-eqdsk.simagx)
    p = np.polyfit(psi_lin, eqdsk.qpsi,10)
    fn_q = lambda psi: np.polyval(p,psi)
    q_rz = fn_q(psi_norm_downsamp) # q(r,z)
    
    

    # Contour detection for the desired rational surface radius
    # contour,hierarchy=cv2.findContours(np.array(q_rz<m/n,dtype=np.uint8),
    #                                 cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contour,_=cv2.findContours(np.array(psi_norm_downsamp<psi_contour_select,dtype=np.uint8),
                                cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    # Algorithm will also find non-closed contours (e.g. around the magnets)
    try: contour = np.squeeze(contour) # Check if only the main contour was found
    except:
        a_avg=[]#track average minor radial distance to contour
        for s in contour:
            s=np.squeeze(s) # remove extra dimensions
            a_avg.append(np.mean( (R[s[0]]-eqdsk.rmagx)**2+(Z[s[1]]-eqdsk.zmagx)**2))
        # Select contour closest on average to the magnetic center [the main contour]
        contour = np.squeeze( contour[np.argmin(a_avg)] )


    # Calculate tangency vector
    b_norm = []
    all_norm=[]
    all_tang=[]
    all_dr=[]
    all_r_hat = []
    all_dot_prod = []
    # Indexing is flipped
    contour = contour[:,[1,0]]
    R_sig = R[contour[:,0],contour[:,1]] 
    Z_sig = Z[contour[:,0],contour[:,1]]

    # Spline smoothing (necessary for derivative stability)
    R_sig = make_smoothing_spline(np.arange(len(R_sig)), R_sig)(np.arange(len(R_sig)))
    Z_sig = make_smoothing_spline(np.arange(len(Z_sig)), Z_sig)(np.arange(len(Z_sig)))
    for ind,index in enumerate(contour):
        
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
        
        fn_dot = lambda v1, v2: v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
        fn_cross = lambda v1, v2: (v1[1]*v2[2]-v1[2]*v2[1],
                                   v1[2]*v2[0]-v1[0]*v2[2],
                                   v1[0]*v2[1]-v1[1]*v2[0])
        norm = fn_cross(tang,tor_vec)
        all_norm.append(norm)
       
        # Calculate r-hat (r,phi,z) from magnetic axis, at contour point
        r_hat = np.array([R_sig[ind]-eqdsk.rmagx,0,Z_sig[ind]-eqdsk.zmagx])
        r_hat /= fn_norm(r_hat)# Normalize (-1 is because unit normal is flipped, I think)
        all_r_hat.append(r_hat)
        # Take dot product with norm, scale by B_r
        b_norm.append(fn_dot(r_hat,norm)*B_r[index[0],index[1]]*1e4)
        all_dot_prod.append(fn_dot(r_hat,norm))
        #all_norm.append(b_norm)
    
            
    b_norm = np.array(b_norm)

    if debug: __b_norm_debug(b_norm,R_sig,Z_sig,B_r,R,Z,contour,eqdsk.rmagx,eqdsk.zmagx,q_rz,all_r_hat,all_norm,all_tang,all_dot_prod,
                             save_ext=save_ext,doSave=doSave)
    
    return b_norm, R_sig, Z_sig
###################################################################################
def __b_norm_debug(b_norm,R_sig,Z_sig,B_r,R,Z,contour,rmagx,zmagx,q_rz,all_r_hat,all_norm,all_tang,\
                   all_dot_prod,save_ext,doSave='',doPlot_norm=False):
    plt.close('B_Norm_Extraction%s'%save_ext)
    fig,ax=plt.subplots(1,2,tight_layout=True,figsize=(5.5,3),
                            num='B_Norm_Extraction%s'%save_ext)

    extent=[min(R_sig),max(R_sig),min(Z_sig),max(Z_sig)]
    vmin=np.nanmin(B_r);vmax=np.nanmax(B_r)
    #ax.imshow(B_r.T,origin='lower',extent=extent,vmin=vmin,vmax=vmax)
    for i in range(2):ax[i].contourf(R,Z,B_r,zorder=-5)
    ax[0].contour(R,Z,q_rz,cmap='plasma')
    norm_q = Normalize(vmin=np.nanmin(q_rz),vmax=np.nanmax(q_rz))
    fig.colorbar(cm.ScalarMappable(norm=norm_q,cmap=cm.get_cmap('plasma')),
        label=r'q(r,z)',ax=ax[0])
    
    if not doPlot_norm:norm = Normalize(vmin=vmin,vmax=vmax)
    else: norm = Normalize(vmin = min(all_dot_prod),vmax=max(all_dot_prod))
    for i in range(len(contour)):
            if not doPlot_norm:  ax[1].plot(R_sig[i],Z_sig[i],'*',
                     c=cm.get_cmap('plasma')(norm(b_norm[i]) ),alpha=.6)
            else:
                ax[1].plot(R_sig[i],Z_sig[i],'*',
                            c=cm.get_cmap('plasma')(norm(all_dot_prod[i]) ),alpha=.6)
                if not (i % 4):
                    ax[1].plot([R_sig[i],R_sig[i]+all_r_hat[i][0]*.05],[Z_sig[i],Z_sig[i]+all_r_hat[i][2]*.05],label=r'$\hat{r}$' if i==0 else None,lw=1)
                    ax[1].plot([R_sig[i],R_sig[i]+all_norm[i][0]*.05],[Z_sig[i],Z_sig[i]+all_norm[i][2]*.05],c='k',label=r'$\hat{n}$' if i==0 else None,lw=1)
                    ax[1].plot([R_sig[i],R_sig[i]+all_tang[i][0]*.05],[Z_sig[i],Z_sig[i]+all_tang[i][2]*.05],c='r',label=r'$\frac{d}{dr,z}\hat{r}$' if i==0 else None,lw=1)

            
    ax[0].set_ylabel('Z [m]')
    for i in range(2):
        ax[i].set_rasterization_zorder(-1)
        ax[i].set_xlabel('R [m]')
    #ax[1].set_title(r'B$_\mathrm{r}$')
    if doPlot_norm: ax[1].legend(fontsize=8,handlelength=1)
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('plasma')),
        label=r'$\hat{r}\cdot\hat{n}$' if doPlot_norm else r'$||\tilde{\mathrm{B}}_{\hat{n}}||$ [G]',ax=ax[1])
    
    
    if doSave: fig.savefig(doSave+fig.canvas.manager.get_window_title()+('_vecs_' if doPlot_norm else '')+'.pdf',transparent=True)
########################################################################################
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
def interpolate_B(theta_inside,R_inside,Z_inside,dat_B,indices,m,psi_norm_inside=None,phase=0):
    # Evaluate the FAR3D eigenfunction _at the Psi, theta points corresponding to R,Z
    def eigenfunc(theta,B,B_indices,m):
        out = np.zeros((len(B_indices),))
        for ind,m_ in enumerate(m): out += B[B_indices,ind]*np.cos(m_*(theta+0*np.pi/2)+phase) + B[B_indices,ind+len(m)]*np.sin(m_*(theta+0*np.pi/2)+phase)
        return out
    B_RZ = eigenfunc(theta_inside,dat_B,indices,m)

    # Downsample the grid from the gEQDSK file
    r_downsamp = np.linspace(R_inside.min(),R_inside.max(),100)
    z_downsamp = np.linspace(Z_inside.min(),Z_inside.max(),100)
    R_downsamp, Z_downsamp = np.meshgrid(r_downsamp, z_downsamp)

    # Resample the FAR3D mode, now at the correct RZ locations, onto the gEQDSK grid
    B_RZ_downsamp = griddata((R_inside, Z_inside), B_RZ, (R_downsamp, Z_downsamp), method='cubic')
    # For calculation of rational surface location (for B-norm), we also need to resample the normalized psi grid
    if np.any(psi_norm_inside):psi_norm_downsamp = griddata((R_inside, Z_inside), psi_norm_inside, (R_downsamp, Z_downsamp), method='cubic')
    # Need to resample Phi at RZ points 

    return R_downsamp, Z_downsamp, psi_norm_downsamp if np.any(psi_norm_inside) else None ,B_RZ_downsamp

##############################################
def map_psi_to_equilibrium(br_file='br_0000',bth_file='bth_0000',\
    m=[9,10,11,12],eqdsk_file='g1051202011.1000',debug=True,lambda_merezhkin=0.4,\
        data_directory='../M3D-C1_Data/',plot_directory='../output_plots/'):
    # Generate equivalent FAR3D eigenfunction for Br, Bth, interpolated on the equilibrium flux surface grid in (r,z)

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
    R_downsamp,Z_downsamp,psi_norm_downsamp, B_r_RZ_downsamp_cos = interpolate_B(theta_inside,R_inside,Z_inside,dat_r[:,0:],indices,m,psi_norm_inside)
    _,_,_, B_r_RZ_downsamp_sin = interpolate_B(theta_inside,R_inside,Z_inside,dat_r[:,0:],indices,m,phase=np.pi/2)
    _,_,_, B_th_RZ_downsamp_cos = interpolate_B(theta_inside,R_inside,Z_inside,dat_th[:,0:],indices,m)
    _,_,_, B_th_RZ_downsamp_sin = interpolate_B(theta_inside,R_inside,Z_inside,dat_th[:,0:],indices,m,phase=np.pi/2)

    ###########################
    # Plot the results
    if debug:
        plt.close('FAR3D_Eigenfunction')
        fig,ax = plt.subplots(1,2,tight_layout=True,num='FAR3D_Eigenfunction',figsize=(5,4),sharex=True,sharey=True)

        # Get normalization
        vmin=np.nanmin(B_r_RZ_downsamp_cos)
        if vmin >np.nanmin(B_th_RZ_downsamp_cos): vmin = np.nanmin(B_th_RZ_downsamp_cos)
        vmax=np.nanmax(B_r_RZ_downsamp_cos)
        if vmax <np.nanmax(B_th_RZ_downsamp_cos): vmax = np.nanmax(B_th_RZ_downsamp_cos)
        ax[0].contourf(R_downsamp+rmagx,Z_downsamp+zmagx,B_r_RZ_downsamp_cos,levels=30,vmin=vmin,vmax=vmax,zorder=-5)
        ax[1].contourf(R_downsamp+rmagx,Z_downsamp+zmagx,B_th_RZ_downsamp_cos,levels=30,vmin=vmin,vmax=vmax,zorder=-5)
        ax[0].set_title(r'B$_r(\psi_N)$')
        ax[1].set_title(r'B$_\theta(\psi_N)$')
        ax[0].set_ylabel(r'Z [m]')
        for i in range(2):
            ax[i].set_xlabel(r'R [m]')
            ax[i].set_rasterization_zorder(-1)

        norm = Normalize(vmin=vmin,vmax=vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('viridis')),
             label=r'$||\tilde{\mathrm{B}}_{j}||$ [T]',ax=ax[1])
        if plot_directory: fig.savefig(plot_directory+'FAR3D_Eigenfunction_Equilibrium.pdf',transparent=True)
        plt.show()

    print('halt')

    return B_r_RZ_downsamp_cos,B_r_RZ_downsamp_sin, B_th_RZ_downsamp_cos,B_th_RZ_downsamp_sin,\
          R_downsamp + rmagx, Z_downsamp + zmagx, psi_normalized, psi_norm_downsamp
###################################################################################
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
        for ind,m_ in enumerate(m): out += B[:,ind]*np.cos(m_*theta) + B[:,ind+len(m)]*np.sin(m_*theta)
        return out
    
    r_grid = np.array([eigenfunc(t,dat_r,m) for t in theta]).T
    th_grid = np.array([eigenfunc(t,dat_th,m) for t in theta]).T
    


    plt.close('FAR3D_Eigenfunction_Original')
    fig,ax = plt.subplots(1,2,tight_layout=True,num='FAR3D_Eigenfunction_Original',subplot_kw={'projection': 'polar'},figsize=(4,4))

    #ax[0].contourf(theta,psi_normalized,r_grid)
    #ax[1].contourf(theta,psi_normalized,th_grid)
    ax[0].contourf(theta,psi_norm,r_grid,zorder=-5)
    ax[1].contourf(theta,psi_norm,th_grid,zorder=-5)
    
    ax[1].set_xlabel(r'B$_\theta(\psi_N)$')
    ax[0].set_xlabel(r'B$_r(\psi_N)$')
    for i in range(2):
        ax[i].set_rasterization_zorder(-1)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([None,0.25,None,None,1])
    
    if plot_directory: fig.savefig(plot_directory+'FAR3D_Eigenfunction_Original.pdf',transparent=True)
    plt.show()
###############################################################################
###############################################################################
if __name__ == '__main__':
    #plot_B(plot_directory='../output_plots/')
    #map_psi_to_equilibrium(plot_directory='../output_plots/')
    # convert_FAR3D_to_Bnorm(doSave='../output_plots/')
    print('Finished')

