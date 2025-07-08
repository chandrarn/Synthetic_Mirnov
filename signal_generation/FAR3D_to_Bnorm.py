#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:19:50 2025
    FAR-3D to b-norm
@author: rian
"""
import numpy as np
import xarray as xr
from M3DC1_to_Bnorm import calculate_normal_vector
import matplotlib.pyplot as plt

def convert_FAR3D_to_Bnorm(br_file,b_th_file,col):
    
    dat = np.loadtxt('M3D-C1_Data/br_0000',skiprows=1)
    psi_norm = dat[:,0]
    B1 = dat[:,col]
    B2 = dat[:,col+1]
    
    # Psi normal conversion
    

def psi_norm_conversion(psi_norm,psiC1_file,psi_LCFS):
    # generate radial vector to map Psi-norm to R
    psi = xr.load_dataarray(psiC1_file)
    R = psi.coords['R'].data
    Z = psi.coords['Z'].data
    psi = psi.data
    
    psi_max = np.max(psi)
    
    psi_mid = psi[:,100]
    psi_n = np.sqrt( (psi_max-psi_mid)/(psi_max-psi_LCFS))
    

def plot_B(br_file='br_0000',bth_file='bth_0000',m=[9,10,11,12]):
    dat_r = np.loadtxt('../M3D-C1_Data/'+br_file,skiprows=1)
    dat_th = np.loadtxt('../M3D-C1_Data/'+bth_file,skiprows=1)
    
    psi_norm = dat_r[:,0]
    dat_r = dat_r[:,1:]
    dat_th = dat_th[:,1:]
    
    theta = np.linspace(0,2*np.pi,100)
    
    def eigenfunc(theta,B,m):
        out = np.zeros((len(B),))
        for ind,m_ in enumerate(m): out += B[:,2*ind]*np.cos(m_*theta) + B[:,2*ind+1]*np.sin(m_*theta)
        return out
    
    r_grid = np.array([eigenfunc(t,dat_r,m) for t in theta]).T
    th_grid = np.array([eigenfunc(t,dat_th,m) for t in theta]).T
    
    plt.close('FAR3D_Eigenfunction')
    fig,ax = plt.subplots(2,1,subplot_kw={'projection':'polar'},tight_layout=True,
                          num='FAR3D_Eigenfunction',figsize=(2,4))
    ax[0].contourf(theta,psi_norm,r_grid)
    ax[1].contourf(theta,psi_norm,th_grid)
    
    ax[1].set_ylabel(r'B$_\theta(\psi_N)$')
    ax[0].set_ylabel(r'B$_r(\psi_N)$')
    for i in range(2):
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([None,0.25,None,None,1])
    
    fig.savefig('../output_plots/FAR3D_Eigenfunction.pdf',transparent=True)
    plt.show()
    
if __name__ == '__main__':plot_B()