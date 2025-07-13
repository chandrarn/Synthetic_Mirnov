#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:26:44 2025
     Given some m/n and a gEqdsk file, or a cylindrical R,a, calculate the geomgetric
     coordnates for some number of mode filaments
@author: rian
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from freeqdsk import geqdsk
import cv2
from fractions import Fraction
from os import getcwd
plt.ion()

########################
def gen_filament_coords(params):
    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate theta,phi coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point
    theta_out=[]; phi_out=[]
    if type(m_pts) is int: m_pts = [m_pts]
    if type(n_pts) is int: n_pts = [n_pts]*len(m_pts)
    for ind_m, m_ in enumerate(m if type(m) is list else [m]):
        n_ = n[ind_m] if type(n) is list else n
        ratio = Fraction(m_,n_)
        m_local=ratio.numerator;n_local=ratio.denominator
        theta_out.append(np.linspace(0,2*np.pi/m_local*n_local,m_pts[ind_m],endpoint=True))
        phi_out.append(np.linspace(0,m_local*2*np.pi,n_pts[ind_m],endpoint=True))
    return theta_out, phi_out
        
########################
def calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params,debug=False,fil=0,
                                R_lim=[1.5,2.3],Z_lim=[-.4,.4]):
    # For a set of theta, phi mode coordinates, for an m/n mode, return the x,y,z
    # coordinates of the filaments, assigning the radial coordainte based on
    # either a GEQDSK file or a fixed minor radial parameter
    
    # Loop over the number of desired filament surfaces
    # If m, n is not a list, assume single mode
    # Outputs coordinates in the form of a list of lists, where each sublist
    # corresponds to one m/n pair
    coords=[]
    for ind_m, m in enumerate(params['m'] if type(params['m']) is list else [params['m']]):
        coords.append([]) # Initialize empty list for this m/n pair
        n=params['n'][ind_m] if type(params['n']) is list else params['n']
        R=params['R'][ind_m] if type(params['R']) is list else params['R']
        a=params['r'][ind_m] if type(params['r']) is list else params['r']
        
        if file_geqdsk is None: # Assigning circiular flux surface, fixed radius at a
            r_theta = lambda theta: np.array([a]*len(theta)) # running circular approximation
            zmagx=0;rmagx=R
            theta_r = np.linspace(0,2*np.pi,100)
            R_eq,Z_eq = np.linspace(*R_lim,100), np.linspace(*Z_lim,100)
            
            contour = np.array([np.argmin(np.abs(R_eq-a*np.cos(t)-rmagx) ) \
                                for t in theta_r])
            contour = np.vstack((contour,[np.argmin(np.abs(Z_eq-a*np.sin(t)-zmagx).squeeze() ) \
                                        for t in theta_r]) )
            contour=contour[::-1].T # flip for consistency with gEqdsk contour detection
            
            r_norm = [a]*100 # mode exists at a constant radius
            
        else: # Using geqdsk equilibrium to locate flux surfaces
            # Load eqdsk
            print(getcwd())
            with open('input_data/'+file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
            
            # get q(psi(r,z))
            psi_eqdsk = eqdsk.psi
            R_eq=np.linspace(eqdsk.rleft,eqdsk.rleft+eqdsk.rdim,len(psi_eqdsk))
            Z_eq=np.linspace(eqdsk.zmid-eqdsk.zdim/2,eqdsk.zmid+eqdsk.zdim/2,len(psi_eqdsk))
            psi_lin = np.linspace(eqdsk.simagx,eqdsk.sibdry,eqdsk.nx)
            p = np.polyfit(psi_lin, eqdsk.qpsi,10)
            fn_q = lambda psi: np.polyval(p,psi)
            q_rz = fn_q(psi_eqdsk) # q(r,z)
            
            

            # Contour detection for the desired rational surface radius
            contour,hierarchy=cv2.findContours(np.array(q_rz<m/n,dtype=np.uint8),
                                            cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
            # Algorithm will also find non-closed contours (e.g. around the magnets)
            try: contour = np.squeeze(contour) # Check if only the main contour was found
            except:
                a_avg=[]#track average minor radial distance to contour
                for s in contour:
                    s=np.squeeze(s) # remove extra dimensions
                    a_avg.append(np.mean( (R_eq[s[1]]-eqdsk.rmagx)**2+(Z_eq[s[0]]-eqdsk.zmagx)**2))
                # Select contour closest on average to the magnetic center [the main contour]
                contour = np.squeeze( contour[np.argmin(a_avg)] )
        
            # Calculate r(theta) to stay on the surface as we wind around
            r_norm=np.sqrt((R_eq[contour[:,1]]-eqdsk.rmagx)**2+(Z_eq[contour[:,0]]-eqdsk.zmagx)**2)
            theta_r=np.arctan2(Z_eq[contour[:,0]]-eqdsk.zmagx,R_eq[contour[:,1]]-eqdsk.rmagx) % (2*np.pi)
            
            # Interpolate to find r(q=m/n, theta)
            r_theta_ = lambda theta: np.polyval(np.polyfit(theta_r,r_norm,11),theta)
            r_theta = lambda theta: r_theta_(theta%(2*np.pi) ) # Radial coordinate vs theta
            
            # Magnetic center cordinates
            zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
            
        # We now have the radial coordinate r(theta) for the m/n surface
        # Assume theta is starting value, wind in phi
        for theta_ in theta[ind_m]:
            #theta_ = theta_ % (2*np.pi)
            theta_local = n*phi[ind_m]/m
            r = r_theta(theta_local+theta_)
            z = zmagx+r*np.sin(theta_local+theta_)# vertical coordinate
            r_x = r*np.cos(theta_local+theta_) # adjust major radial vector
            x = (rmagx+r_x)*np.cos(phi[ind_m])
            y = (rmagx+r_x)*np.sin(phi[ind_m])
            coords[-1].append([x,y,z])
    
        # Debug plots
        if debug:
            plt.close('filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
            fig,ax = plt.subplots(1,3,tight_layout=True,figsize=(7,4),
                    num='filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
            ax[0].plot(R_eq[contour[:,1]],Z_eq[contour[:,0]],'*')
            theta_fit=np.linspace(0,2*np.pi,50)#-np.pi
            ax[0].plot(r_theta(theta_fit)*np.cos(theta_fit)+rmagx,
                    r_theta(theta_fit)*np.sin(theta_fit)+zmagx,alpha=.6)
        
            ax[1].plot(theta_r/(2*np.pi),r_norm,'*',label=r'$\psi_{%d/%d}$'%(m,n))
            ax[1].plot(theta_fit/(2*np.pi),r_theta(theta_fit),alpha=.6,label='Spl. Fit')
            ax[1].legend(fontsize=8)
            
            ax[2].plot(phi[ind_m]/(2*np.pi),coords[ind_m][fil][0],label='X')
            ax[2].plot(phi[ind_m]/(2*np.pi),coords[ind_m][fil][1],label='Y')
            ax[2].plot(phi[ind_m]/(2*np.pi),coords[ind_m][fil][2],label='Z')
            ax[2].legend(fontsize=8)
            
            for i in range(len(ax)):ax[i].grid()
            
            ax[0].set_xlabel('R [m]')
            ax[0].set_ylabel('Z [m]')
            
            ax[1].set_xlabel(r'$\theta$ [2$\pi$ rad]')
            ax[1].set_ylabel(r'$||r_{%d/%d}||$ [m]'%(m,n))
            
            ax[2].set_xlabel(r'$\phi$ [2$\pi$ rad]')
            ax[2].set_ylabel(r'Filament \#%d Coordinatate [m]'%fil)
            plt.show()
    return coords
##########################
if __name__ == '__main__': 
    params={'m':[2,3],'n':[1,2],'r':.35,'R':1.9,'n_pts':100,'m_pts':20,\
    'f':1e3,'dt':1e-4,'periods':1,'n_threads':4,'I':10}
    theta ,phi = gen_filament_coords(params)
    #geqdsk_freegsu_run0_mod_00.geq
    # g1051202011.1000
    coords = calc_filament_coords_geqdsk('geqdsk_freegsu_run0_mod_00.geq',theta,phi,params,debug='_gEqdsk')
    coords = calc_filament_coords_geqdsk(None,theta,phi,params,debug='_Fixed_Radius')
