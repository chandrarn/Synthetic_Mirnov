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
    