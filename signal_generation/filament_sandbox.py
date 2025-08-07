# Sandbox file for testing new filament generator

import numpy as np
from scipy.interpolate import make_smoothing_spline
import matplotlib.pyplot as plt
from freeqdsk import geqdsk
import cv2
from fractions import Fraction
from os import getcwd
from sys import path; path.append('/home/rianc/Documents/TARS/')
from tars.filaments import EquilibriumFilament, TraceType
from tars.magnetic_field import EquilibriumField
import matplotlib;matplotlib.use('TkAgg')


plt.ion()
# generate contours
def gen_contours(file_geqdsk,doDebug=True):
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
    arg_sort_theta_r = np.argsort(theta_r)
    r_theta = make_smoothing_spline(theta_r[arg_sort_theta_r],r_norm[arg_sort_theta_r],lam=None)
    # r_theta_ = lambda theta: np.polyval(np.polyfit(theta_r[:-1],r_norm[:-1],11),theta)
    # r_theta = lambda theta: r_theta_(theta%(2*np.pi) ) # Radial coordinate vs theta
    
    # Magnetic center cordinates
    zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx

    # NEW: Need B_th, B_t for d_theta calculation
    eqilib_field = EquilibriumField(eqdsk)
    Br,Bt,Bz = eqilib_field.get_field_at_point(r_theta(theta_r)*np.cos(theta_r)+rmagx,r_theta(theta_r)*np.sin(theta_r)+zmagx)
    Bp = np.sqrt(Br**2 + Bz**2)


    # Need a function to extract Bp, Bt as functions of theta, to later find d_theta

    
    Bp_theta = make_smoothing_spline(theta_r[arg_sort_theta_r],Bp[arg_sort_theta_r],lam=None)
    Bt_theta = make_smoothing_spline(theta_r[arg_sort_theta_r],Bt[arg_sort_theta_r],lam=None)
    # Bp_theta = lambda theta: np.polyval(np.polyfit(theta_r[:-1],Bp[:-1],11),theta%(2*np.pi))
    # Bt_theta = lambda theta: np.polyval(np.polyfit(theta_r[:-1],Bt[:-1],11),theta%(2*np.pi))

    if doDebug:
        plt.close('Mag_Equilib_Debug')
        fig,ax = plt.subplots(1,4,num='Mag_Equilib_Debug',tight_layout=True)
        ax[0].plot(R_eq[contour[:,1]],Z_eq[contour[:,0]],'*')
        
        ax[1].plot(theta_r,r_norm,'*')
        ax[1].plot(theta_r,r_theta(theta_r))
        ax[2].plot(theta_r,Bp)
        ax[2].plot(theta_r,Bp_theta(theta_r))
        ax[3].plot(theta_r,Bt)
        ax[3].plot(theta_r,Bt_theta(theta_r))
        for i in range(4):ax[i].grid()
    return r_theta,Bt_theta, Bp_theta, rmagx, zmagx,eqilib_field
################################
def starting_phi(m,n,m_pts,n_pts):
    ratio = Fraction(m,n)
    m_local=ratio.numerator;n_local=ratio.denominator
    phi_start = np.linspace(0,2*np.pi/n_local,n_pts,endpoint=False)
    phi_advance = np.linspace(0,2*np.pi*m_local,m_pts)
    return phi_start, phi_advance
#############################3
def wind_in_phi(phi_start,phi_advance,r_theta,Bp_theta, Bt_theta, rmagx,zmagx, m_pts, n, m,eqilib_field):

    coords = []
    theta = np.array([0])
    d_phi = 2*np.pi / m_pts * m
    d_th = []
    for _ in range(m_pts-1):
        theta = np.append(theta, theta[-1] + _d_theta(r_theta(theta[-1]), rmagx, Bp_theta(theta[-1]), Bt_theta(theta[-1]), d_phi) )
        d_th.append( _d_theta(r_theta(theta[-1]), rmagx, Bp_theta(theta[-1]), Bt_theta(theta[-1]), d_phi))

    #TODO  May need to add a correction factor to make sure you end up in the right spot
    known_theta = np.linspace(0,2*np.pi*n,(2 * m) + 1)
    for i, known_theta_start in enumerate(known_theta[:-1]):
        known_theta_end = known_theta[i+1]

        theta_indices = np.where((theta >= known_theta_start) & (theta <= known_theta_end))

        actual_theta_start = theta[theta_indices[0]]
        actual_theta_end = theta[theta_indices[-1]]
        correction_factor = (known_theta_end - known_theta_start) / (actual_theta_end - actual_theta_start)
        theta[theta_indices] = known_theta_start + (theta[theta_indices] - actual_theta_start) * correction_factor

    for phi in phi_start:
        phi_ = phi + phi_advance # All phi points from a given starting point, of size m_pts
        r = rmagx + r_theta(theta)
        Z = zmagx + r_theta(theta)*np.sin(theta)
        X = r*np.cos(phi_) 
        Y = r*np.sin(phi_)
        coords.append([X,Y,Z])

    coords = np.array(coords)
    return coords, theta, d_th

def _d_theta(r, R, Bp, Bt, d_phi):
    return np.abs((R * Bp) / (Bt * r) * d_phi)


def debug_plots(coords, theta, phi,m,n,d_th):
    plt.close('Updated_Coordinate_Plot_m%d_n%d'%(m,n))
    fig = plt.figure(num='Updated_Coordinate_Plot_m%d_n%d'%(m,n),tight_layout=True)
    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2,projection='3d')
    ax3= fig.add_subplot(1,3,3)

    ax1.plot(phi/(2*np.pi),theta/(2*np.pi))
    ax1.plot(np.array([phi[0],phi[-1]])/(2*np.pi), np.array([theta[0],theta[-1]])/(2*np.pi),'--k' )
    ax1.set_xlabel(r'$\hat{\phi}$ [$2\pi$-rad]')
    ax1.set_ylabel(r'$\hat{\theta}$ [$2\pi$-rad]')
    ax1.grid()
    ax2.plot(coords[0,0,:],coords[0,1,:],coords[0,2,:],)

    ax3.plot(phi[:-1],d_th)
    ax3.grid()

    plt.show()

###############################################################
# Zander's way
def wind_in_theta(file_geqdsk,m,n):
    with open('input_data/'+file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
    eq_field = EquilibriumField(eqdsk)
    eq_filament = EquilibriumFilament(m,n,eq_field)
    filament_points,filament_etas = eq_filament._trace(method=TraceType.AVERAGE,num_filament_points=300)

    plt.close('Winding_Theta')
    plt.figure(num='Winding_Theta',tight_layout=True);
    plt.plot(filament_points[:,1]/(2*np.pi));
    plt.plot([0,len(filament_points)],[0,filament_points[-1,1]/(2*np.pi)],'--k');
    plt.grid();
    plt.ylabel(r'$\hat{\phi}$ [2$\pi$ Rad]')
    plt.xlabel('Filament Points')
    plt.show()

    # Nan Check
    return filament_points,filament_etas
#######################################

def conv_theta_wind_to_coords(filament_points,phi_start):
    coords = []

    for phi_0 in phi_start:
        phi_ = filament_points[:,1]+phi_0 # local phi for a given filament
        X = filament_points[:,0]*np.cos(phi_)
        Y = filament_points[:,0]*np.sin(phi_)
        Z = filament_points[:,2]
        coords.append([X,Y,Z])
    
    return np.array(coords).T

def debug_filament_currents(coords,phi_start,n):
    plt.close('Filament_New_Way_Test')
    fig,ax=plt.subplots(1,1,num='Filament_New_Way_Test',tight_layout=True,subplot_kw={'projection':'3d'})
    for ind,filament in enumerate(coords):
        color = plt.get_cmap('plasma')((np.cos(phi_start[ind]*n)+1)/2)
        ax.plot(filament[0,:],filament[1,:],filament[2,:],c=color,lw=2)
    
    plt.show()

def calc_filament_coords_field_lines(params,file_geqdsk,doDebug=False):

    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate theta,phi coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point

    starting_angle=[]; winding_angle=[]
    if type(m_pts) is int: m_pts = [m_pts]*len(m)
    if type(n_pts) is int: n_pts = [n_pts]*len(m_pts)
    coords = []
    for ind_m, m_ in enumerate(m if type(m) is list else [m]):
        phi_start, phi_advance =  starting_phi(m_,n[ind_m],m_pts[ind_m],n_pts[ind_m])

        filament_points,filament_etas = wind_in_theta(file_geqdsk,m_,n[ind_m])

        coords_ = conv_theta_wind_to_coords(filament_points,phi_start)

        if doDebug: debug_filament_currents(coords_,phi_start,n[ind_m])
        #coords_[:,0]= coords[:,1]
        coords_[0,:2,:]=coords_[1,:2,:]
        coords.append(coords_.T)

    return coords
if __name__ == '__main__':
    m = 1
    n = 1
    m_pts = 1000
    n_pts = 300
    file_geqdsk =  'g1051202011.1000'

    filament_points,filament_etas = wind_in_theta(file_geqdsk,m,n)

    # r_theta,Bt_theta, Bp_theta, rmagx, zmagx,eqilib_field = gen_contours(file_geqdsk)

    # coords,theta,d_th = wind_in_phi(phi_start,phi_advance,r_theta,Bp_theta, Bt_theta, rmagx,zmagx, m_pts, n, m,eqilib_field)

    # debug_plots(coords, theta, phi_advance,m,n,d_th)
    print('Complete')