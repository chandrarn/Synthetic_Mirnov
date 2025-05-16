#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 21:26:44 2025

@author: rian
"""

from header_signal_generation import np, plt, geqdsk, cv2, make_smoothing_spline,\
    Fraction

    
########################
def gen_filament_coords(params):
    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate theta,phi coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point
    ratio = Fraction(m,n)
    m_local=ratio.numerator;n_local=ratio.denominator
    return np.linspace(0,2*np.pi/m_local*n_local,m_pts,endpoint=True),\
        np.linspace(0,m_local*2*np.pi,n_pts,endpoint=True)
 
########################
def calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params,debug=False,fil=0):
    # For a set of theta, phi mode coordinates, for an m/n mode, return the x,y,z
    # coordinates of the filaments, assigning the radial coordainte based on
    # either a GEQDSK file or a fixed minor radial parameter
    
    m=params['m'];n=params['n'];R=params['R'];a=params['r']
    
    if file_geqdsk is None:
        r_theta = lambda theta: a # running circular approximation
        zmagx=0;rmagx=R
    else: # Using geqdsk equilibrium to locate flux surfaces
        # Load eqdsk
        with open('input_data/'+file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
        
        # get q(psi(r,z))
        psi_eqdsk = eqdsk.psi
        R_eq=np.linspace(eqdsk.rleft,eqdsk.rleft+eqdsk.rdim,len(psi_eqdsk))
        Z_eq=np.linspace(eqdsk.zmid-eqdsk.zdim/2,eqdsk.zmid+eqdsk.zdim/2,len(psi_eqdsk))
        psi_lin = np.linspace(eqdsk.simagx,eqdsk.sibdry,eqdsk.nx)
        p = np.polyfit(psi_lin, eqdsk.qpsi,12)
        fn_q = lambda psi: np.polyval(p,psi)
        q_rz = fn_q(psi_eqdsk) # q(r,z)
        
        # Contour detectioon
        contour,hierarchy=cv2.findContours(np.array(q_rz<m/n,dtype=np.uint8),
                                           cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        # Algorithm will find non-closed contours (e.g. around the magnets)
        try: contour = np.squeeze(contour) # Check if only the main contour was found
        except:
            a_avg=[]#track average minor radial distance to contour
            for s in contour:
                s=np.squeeze(s) # remove extra dimensions
                a_avg.append(np.mean( (R_eq[s[1]]-eqdsk.rmagx)**2+(Z_eq[s[0]]-eqdsk.zmagx)**2))
            # Select contour closest on average to the magnetic center
            contour = np.squeeze( contour[np.argmin(a_avg)] )
    
        # Calculate r(theta) to stay on the surface as we wind around
        r_norm=np.sqrt((R_eq[contour[:,1]]-eqdsk.rmagx)**2+(Z_eq[contour[:,0]]-eqdsk.zmagx)**2)
        theta_r=np.arctan2(Z_eq[contour[:,0]]-eqdsk.zmagx,R_eq[contour[:,1]]-eqdsk.rmagx) % (2*np.pi)
        
        r_theta_=make_smoothing_spline(theta_r[np.argsort(theta_r)],r_norm[np.argsort(theta_r)],lam=.00001)
        r_theta = lambda theta: r_theta_(theta%(2*np.pi) ) # Radial coordinate vs theta
        
        zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
    # Assume theta is starting value, wind in phi
    coords=[]
    for theta_ in theta:
        #theta_ = theta_ % (2*np.pi)
        theta_local = n*phi/m
        r = r_theta(theta_local+theta_)
        z = zmagx+r*np.sin(theta_local+theta_)# vertical coordinate
        r_x = r*np.cos(theta_local+theta_) # adjust major radial vector
        x = (rmagx+r_x)*np.cos(phi)
        y = (rmagx+r_x)*np.sin(phi)
        coords.append([x,y,z])
    
    # Debug plots
    if debug:
        plt.close('filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
        fig,ax = plt.subplots(1,3,tight_layout=True,figsize=(7,4),
                  num='filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
        ax[0].plot(R_eq[contour[:,1]],Z_eq[contour[:,0]],'*')
        theta_fit=np.linspace(0,2*np.pi,50)#-np.pi
        ax[0].plot(r_theta(theta_fit)*np.cos(theta_fit)+eqdsk.rmagx,
                   r_theta(theta_fit)*np.sin(theta_fit)+eqdsk.zmagx,alpha=.6)
    
        ax[1].plot(theta_r/(2*np.pi),r_norm,'*',label=r'$\psi_{%d/%d}$'%(m,n))
        ax[1].plot(theta_fit/(2*np.pi),r_theta(theta_fit),alpha=.6,label='Spl. Fit')
        ax[1].legend(fontsize=8)
        
        ax[2].plot(phi/(2*np.pi),coords[fil][0],label='X')
        ax[2].plot(phi/(2*np.pi),coords[fil][1],label='Y')
        ax[2].plot(phi/(2*np.pi),coords[fil][2],label='Z')
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
    params={'m':18,'n':16,'r':.25,'R':1,'n_pts':100,'m_pts':20,\
    'f':1e3,'dt':1e-4,'periods':1,'n_threads':4,'I':10}
    theta ,phi = gen_filament_coords(params)
    coords = calc_filament_coords_geqdsk('geqdsk',theta,phi,params,debug=True)
    