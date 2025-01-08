#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 16:37:30 2024
    Generate grid from eqdsk file
    Need: How does timestep map to filament points
@author: rian
"""
import numpy as np
from scipy import interpolate

def gen_grid_Eqdsk(eqdsk_file='/home/rian/Documents/reconstructions/hbt_115600/run_3/Psitri.eqdsk',\
                   m=2,n=1):
    # Create list of r,z points for q=m/n, to use for filament generation
    eqdsk_obj = read_eqdsk(eqdsk_file)
    
    # Psi index for m/n
    # index in fraction of uniform psi grid
    q_mn_psi=np.argmin((eqdsk_obj['qpsi']-m/n)**2)/eqdsk_obj['nr'] 
    
    # Psi_rz in normalized values
    psi_norm = (eqdsk_obj['psirz']-eqdsk_obj['psiax'])/(eqdsk_obj['psilcf']-eqdsk_obj['psiax'])
    
    r0,z0=np.unravel_index(np.argmax(eqdsk_obj['psirz']),eqdsk_obj['psirz'].shape)
    
    inds_lower = np.argmin((psi_norm[:,:z0]-q_mn_psi)**2,axis=1)
    inds_upper = np.argmin((psi_norm[:,z0:]-q_mn_psi)**2,axis=1)+z0
    
    # Mask r>q_mn
    ind_min = np.argwhere(psi_norm[:r0,z0]>q_mn_psi)[-1][0]
    ind_max = np.argwhere(psi_norm[r0:,z0]>q_mn_psi)[0][0]+r0
    return psi_norm,inds_lower,inds_upper,ind_min,ind_max

def read_eqdsk(filename):

    def read_1d(fid, j, n):
        output = np.zeros((n,))
        for i in range(n):
            if j == 0:
                line = fid.readline()
            output[i] = line[j:j+16]
            j += 16
            if j == 16*5:
                j = 0
        return output, j

    def read_2d(fid, j, n, m):
        output = np.zeros((m, n))
        for k in range(n):
            for i in range(m):
                if j == 0:
                    line = fid.readline()
                output[i, k] = line[j:j+16]
                j += 16
                if j == 16*5:
                    j = 0
        return output, j
    # Read-in data
    eqdsk_obj = {}
    with open(filename, 'r') as fid:
        # Get sizes
        line = fid.readline()
        split_line = line.split()
        eqdsk_obj['nz'] = int(split_line[-1])
        eqdsk_obj['nr'] = int(split_line[-2])
        # Read header content
        line_keys = [['rdim',  'zdim',  'raxis',  'rleft',  'zmid'],
                     ['raxis', 'zaxis', 'psiax', 'psilcf', 'bcentr'],
                     ['itor',  'skip',  'skip',   'skip',   'skip'],
                     ['skip',  'skip',  'skip',   'skip',   'skip']]
        for i in range(4):
            line = fid.readline()
            for j in range(5):
                if line_keys[i][j] == 'skip':
                    continue
                line_seg = line[j*16:(j+1)*16]
                eqdsk_obj[line_keys[i][j]] = float(line_seg)
        # Read flux profiles
        j = 0
        keys = ['fpol', 'pres', 'ffprime', 'pprime']
        for key in keys:
            eqdsk_obj[key], j = read_1d(fid, j, eqdsk_obj['nr'])
        # Read PSI grid
        eqdsk_obj['psirz'], j = read_2d(fid, j, eqdsk_obj['nz'], eqdsk_obj['nr'])
        eqdsk_obj['psirz'] = np.transpose(eqdsk_obj['psirz'])
        # Read q-profile
        eqdsk_obj['qpsi'], j = read_1d(fid, j, eqdsk_obj['nr'])
        # Skip line (data already present)
        line = fid.readline()
        split_line = line.split()
        eqdsk_obj['nbbbs'] = int(split_line[-1])
        eqdsk_obj['nlimit'] = int(split_line[-2])
        # Read outer flux surface
        eqdsk_obj['rzout'], j = read_2d(fid, j, eqdsk_obj['nbbbs'], 2)
        # Read limiting corners
        eqdsk_obj['rzlim'], j = read_2d(fid, j, eqdsk_obj['nlimit'], 2)
    return eqdsk_obj

