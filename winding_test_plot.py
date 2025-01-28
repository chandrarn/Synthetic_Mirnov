#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 18:38:44 2025

@author: rian
"""

# Test plotting for filament generator

from geqdsk_filament_generator import calc_filament_coords_geqdsk
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from fractions import Fraction


def main():
    params={'m':12,'n':10,'r':.25,'R':1,'n_pts':100,'m_pts':50,\
    'f':1e3,'dt':1e-4,'periods':1,'n_threads':4,'I':10}
    
    
        
    coords = gen_filament_coords(params)
    
    
    # Plotting
    r = params['r']
    
    r_theta = lambda theta: r
    
    
    r=params['r'];R=params['R'];m=params['m'];n=params['n']
    
    theta = np.linspace(0,2*np.pi,params['n_pts'])
    
    m_theta_start = np.linspace(0,2*np.pi,params['m_pts'])
    
    coords = []
    ratio = Fraction(m,n)
    m_local=ratio.numerator;n_local=ratio.denominator
    for theta_m in m_theta_start:
        coords.append([])
        for t in theta:
            coords[-1].append([])
            r_x = r*np.cos(m_local*t+theta_m)
            
            coords[-1][-1].append((R+r_x)*np.cos(n_local*t))
            coords[-1][-1].append((R+r_x)*np.sin(n_local*t))
            coords[-1][-1].append(r*np.sin(m_local*t+theta_m))
            
    coords = np.array(coords)
    
    # Old way
    theta, phi = gen_filament_coords(params)
    # theta = theta % (2*np.pi)
    # phi= phi % (2*np.pi)
    coords_old = calc_filament_coords_geqdsk(None,theta,phi,params)
    #return coords
    # Make plot
    plt.close('test')
    fig,ax=plt.subplots(1,2,num='test',tight_layout=True,subplot_kw={'projection':'3d'})
    
    other_lines=.2
    for ind, line in enumerate(coords):
        color = cm.get_cmap('plasma')((np.cos(m_theta_start[ind]*m)+1)/2)
        color_old = cm.get_cmap('plasma')((np.cos(theta[ind]*m)+1)/2)
        ax[0].plot(*line[0].T,'*',ms=10,c=color)
        ax[0].plot(*line.T,c=color,alpha=1 if ind==(len(coords)-1) else other_lines,lw=10 if ind==0 else 1)
        
        ax[1].plot(*np.array(coords_old[ind]).T[0],'*',ms=10,c=color_old)
        ax[1].plot(*coords_old[ind],c=color_old,
                   alpha=1 if ind==(len(coords_old)-1) else other_lines,lw=10 if ind==0 else 1)
        
    for i in range(2):ax[i].grid()
    plt.show()
    
    return coords,coords_old, theta, phi


########################
def gen_filament_coords(params):
    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate phi, theta coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point
    ratio = Fraction(m,n)
    m_local=ratio.numerator;n_local=ratio.denominator
    return np.linspace(0,2*np.pi/m_local*n_local,m_pts,endpoint=True),\
        np.linspace(0,m_local*2*np.pi,n_pts,endpoint=True)
##########################
if __name__=='__main__':main()