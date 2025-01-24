#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:43:59 2024
 Basic wrapper code for arbitrary fillament generation and mesh integration
@author: rian
"""

# header
from header import struct,sys,os,h5py,np,plt,mlines,rc,cm,pyvista,ThinCurr,\
    Mirnov, save_sensors, build_XDMF, mu0, histfile, subprocess, geqdsk, cv2,\
        make_smoothing_spline
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated

from prep_sensors import conv_sensor
# main loop
def gen_synthetic_Mirnov(input_file='',mesh_file='thincurr_ex-torus.h5',
                         xml_filename='oft_in.xml',sensor_filename='floops.loc',\
                             params={'m':12,'n':10,'r':.25,'R':1,'n_pts':50,'m_pts':60,\
                            'f':1e3,'dt':1e-4,'periods':2,'n_threads':64,'I':10},
                                doSave='',save_ext='',file_geqdsk='geqdsk',
                                sensor_set='MIRNOV'):
    
    #os.system('rm -rf vector*') # kernal restart still required for vector numbering issue
    
    # Get coordinates for fillaments
    theta,phi=gen_filament_coords(params)
    filament_coords = calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params)
    
    # Generate sensors, filamanets
    gen_filaments(xml_filename,params,filament_coords)
    #sensors = gen_sensors() 
    #sensors = conv_sensor('sensorLoc.xyz')[0]
    sensors=gen_Sensors_Updated(select_sensor=sensor_set)
    # Get Mesh
    tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv = \
        get_mesh(mesh_file,xml_filename,sensor_filename,params)

    # Get mode amplitudes, assign to fillaments [eventually, from simulation]
    
    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(params)
    
    # Run time dependent simulation
    run_td(sensor_obj,tw_mesh,params, coil_currs)
    
    
    slices,slices_spl=makePlots(tw_mesh,params,coil_currs,sensors,doSave,
                                save_ext,Mc, L_inv,filament_coords)

    return sensors, coil_currs, tw_mesh, slices,slices_spl
#    return sensor_,currents
#####################################3
def get_mesh(mesh_file,filament_file,sensor_file,params,doPlot=True):
    tw_mesh = ThinCurr(nthreads=params['n_threads'],debug_level=2,)
    tw_mesh.setup_model(mesh_file=mesh_file,xml_filename=filament_file)
    print('checkpoint 0')
    tw_mesh.setup_io()
    
    print('checkpoint 1')
    # Sensor - mesh and sensor - filament inductances
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor(sensor_file)
    print('checkpoint 2')
    # Filament - mesh inductance
    Mc = tw_mesh.compute_Mcoil()
    print('checkpoint 3')
    # Build inductance matrix
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file='HOLDR_L.save')
    print('checkpoint 4')
    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction
    eig_vals, eig_vecs =(0,0)# tw_mesh.get_eigs(100,False)
    L_inv = 0#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))
    print('checkpoint 5')
    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

####################################
def gen_filaments(filament_file,params, ):
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    n_pts=params['n_pts'];m_pts=params['m_pts']
    #theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    eta='1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5'
    with open(filament_file,'w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n'%eta)
        
        for filament in filament_coords:
            f.write('\t<coil_set>\n\t\t<coil npts="%d" scale="1.0">\n'%n_pts)
            for xyz in np.array(filament).T:
                x=xyz[0];y=xyz[1];z=xyz[2]
                # for some starting theta, wind in phi
                f.write('\t\t\t %1.1f, %1.1f, %1.1f\n'%(x,y,z) )
            f.write('\t\t</coil>\n\t</coil_set>\n')
        f.write('\t</icoils>\n\t</thincurr>\n</oft>')
    # Assume we have the shaping jacobian for theta into minor radial corrdinanat
    # Eventually manually calc points
    
   
####################################
def gen_sensors():
    sensors = []
    for i, theta in enumerate(np.linspace(0.0-np.pi/4,2.0*np.pi/2.0-np.pi/4,5)):
        sensors.append(Mirnov(1.6*np.r_[np.cos(theta),np.sin(theta),0.0], np.r_[0.0,0.0,1.0], 'Bz_{0}'.format(i+1)))
    save_sensors(sensors)
    return sensors
####################################
def gen_coil_currs(param):
    # Assign currents to existing fillaments for time dependent calculation
    # Eventually, the balooning approximation goes here, as does straight field line apprxoiation
    m_pts=param['m_pts'];m=param['m'];dt=param['dt'];f=param['f'];periods=param['periods']
    I=param['I'];n=param['n'];n_pts=param['n_pts']
    theta_,phi_= gen_filament_coords(param)
    coil_currs = np.zeros((int(periods/f/dt),m_pts+1))
    for ind,t in enumerate(np.arange(0,periods/f,dt)):
        coil_currs[ind,1:]=[I*np.cos(m*theta+t*f*2*np.pi) for theta in theta_]
        
    # Time vector in first column
    coil_currs[:,0]=np.arange(0,periods/f,dt)
    
    return coil_currs
####################################
def run_td(sensor_obj,tw_mesh,param,coil_currs,doPlot=True):
    dt=param['dt'];f=param['f'];periods=param['periods']
    tw_mesh.run_td(dt,int(periods/f/dt),
                    coil_currs=coil_currs,sensor_obj=sensor_obj,status_freq=100,plot_freq=100)
    tw_mesh.plot_td(int(periods/f/dt),compute_B=False,sensor_obj=sensor_obj,plot_freq=100)
    
    # Save B-norm surface for later plotting # This may be unnecessar
    _, Bc = tw_mesh.compute_Bmat(cache_file='HODLR_B.save') 
       
    return coil_currs
########################
def makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_Ext,Mc, L_inv,
              filament_coords, t_pt=0,):
    
    # MEsh and Filaments
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
    f=params['f'];dt=params['dt'];I=params['I']

    # Calculate induced current in mesh
    
    #tw_plate.save_current(np.dot(Linv,M[0]),'M')
    # tw_mesh.save_current(np.dot(L_inv,np.dot(Mc.T,coil_currs[0,1:])), 'Ind_Curr')
    # Necessary to clear old vector file: otherwise vector # keeps counting up
    
    tw_mesh.build_XDMF()
    with h5py.File('mesh.0001.h5','r') as h5_file:
        r_ = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
    with h5py.File('vector_dump.0001.h5') as h5_file:
        Jfull = np.asarray(h5_file['J_v0001'])
        scale = 0.2/(np.linalg.norm(Jfull,axis=1)).max()
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r_)

    p = pyvista.Plotter()  
    
    # Plot Mesh
    #p.add_mesh(grid,color="white",opacity=.9,show_edges=True,scalars=Jfull)
    #slices=grid.slice_orthogonal()
    slice_coords=[np.linspace(0,1.6,10),[0]*10,np.linspace(-1.5,1.5,10)]
    slice_line = pyvista.Spline(np.c_[slice_coords].T,10)
    slices = grid.slice_along_line(slice_line)
    #slices.plot()
    p.add_mesh(slices,label='Mesh Frame')
    
    tmp=[]
    
    # Plot Filaments
    colors=['red','green','blue']
    #theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    for ind,filament in enumerate(filament_coords):
        pts = np.array(filament).T
        #print(np.array(calc_filament_coord(m,n,r,R,theta,phi)).shape)
        pts=np.array(pts)
        #print(pts.shape,theta)
        spl=pyvista.Spline(pts,100)
        #p.add_(spline,render_lines_as_tubes=True,line_width=5,show_scalar_bar=False)
        #p.add_mesh(spl,opacity=1,line_width=6,color=plt.get_cmap('viridis')(theta*m/(2*np.pi)))
        slices_spl=spl.slice_along_line(slice_line)#spl.slice_orthogonal()
        p.add_mesh(spl,color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2),
                   line_width=10,render_points_as_spheres=True,
                   label='Filament' if ind==0 else None)
        #p.add_points(pts,render_points_as_spheres=True,opaity=1,point_size=20,color=colors[ind])
        tmp.append(pts)
        
    # Plot Sensors
    for ind,s in enumerate(sensors):
        p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                     render_points_as_spheres=True,
                     label='Sensor' if ind==0 else None)
    p.add_legend()
    if doSave:p.save_graphic(doSave+'Mesh_and_Filaments%s.pdf'%save_Ext)
    p.show()
    
    plot_Currents(params, coil_currs, doSave, save_Ext)
          
    plt.show()
    return slices, slices_spl
########################
# ##############################
# def calc_filament_coord(m,n,r,R,theta,phi):
#     # Return x,y,z coordinate of a specific phi, theta point
#     # Locally: evolve theta as n*phi/m
#     # Treat theta as non-evolved offset parameter (to set initial angle)
#     return (R+r*np.cos(n*phi/m + theta))*np.cos(n*phi), \
#             (R+r*np.cos(n*phi/m+theta))*np.sin(n*phi), r*np.sin(n*phi/m+theta)
# ########################  
########################
def gen_filament_coords(params):
    m=params['m'];n=params['n'];n_pts=params['n_pts'];m_pts=params['m_pts']
    # generate phi, theta coordinates for fillaments
    # The points launch in a fractional sector of the poloidal plane, and
    # wrap toroidally enough times to return to their starting point
    return np.linspace(0,2*np.pi/m,m_pts,endpoint=True),\
        np.linspace(0,m*2*np.pi*n,n_pts,endpoint=True)
 
########################
def calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params,debug=False,fil=5):
    # For a set of theta, phi mode coordinates, for an m/n mode, return the x,y,z
    # coordinates of the filaments, assigning the radial coordainte based on
    # either a GEQDSK file or a fixed minor radial parameter
    
    m=params['m'];n=params['n'];R=params['R'];a=params['r']
    
    if file_geqdsk is None:
        r_theta = lambda theta: a # running circular approximation
        
    else: # Using geqdsk equilibrium to locate flux surfaces
        # Load eqdsk
        with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
        
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
            # Select contour closested on average to the magnetic center
            contour = np.squeeze( contour[np.argmin(a_avg)] )
    
        # Calculate r(theta) to stay on the surface as we wind around
        r_norm=np.sqrt((R_eq[contour[:,1]]-eqdsk.rmagx)**2+(Z_eq[contour[:,0]]-eqdsk.zmagx)**2)
        theta_r=np.arctan2(Z_eq[contour[:,0]]-eqdsk.zmagx,R_eq[contour[:,1]]-eqdsk.rmagx) % (2*np.pi)
        
        r_theta_=make_smoothing_spline(theta_r[np.argsort(theta_r)],r_norm[np.argsort(theta_r)],lam=.00001)
        r_theta = lambda theta: r_theta_(theta%(2*np.pi) ) # Radial coordinate vs theta
        
    # Assume theta is starting value, wind in phi
    coords=[]
    for theta_ in theta:
        r = r_theta(n*phi/m+theta_)
        z = eqdsk.zmagx+r*np.sin(n*phi/m+theta_)# vertical coordinate
        r_x = r*np.cos(n*phi/m+theta_) # adjust major radial vector
        x = (eqdsk.rmagx+r_x)*np.cos(phi)
        y = (eqdsk.rmagx+r_x)*np.sin(phi)
        coords.append([x,y,z])
    
    # Debug plots
    if debug:
        plt.close('filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
        fig,ax = plt.subplots(1,3,tight_layout=True,figsize=(7,4),
                  num='filament_debug_%d-%d%s'%(m,n,debug if type(debug) is str else ''))
        ax[0].plot(R_eq[contour[:,1]],Z_eq[contour[:,0]],'*')
        theta__=np.linspace(0,2*np.pi,50)#-np.pi
        ax[0].plot(r_theta(theta__)*np.cos(theta__)+eqdsk.rmagx,
                   r_theta(theta__)*np.sin(theta__)+eqdsk.zmagx,alpha=.6)
    
        ax[1].plot(theta_r,r_norm,'*',label=r'$\psi_{%d/%d}$'%(m,n))
        ax[1].plot(theta__,r_theta(theta__),alpha=.6,label='Spl. Fit')
        ax[1].legend(fontsize=8)
        
        ax[2].plot(phi/(2*np.pi),coords[fil][0],label='X')
        ax[2].plot(phi/(2*np.pi),coords[fil][1],label='Y')
        ax[2].plot(phi/(2*np.pi),coords[fil][2],label='Z')
        ax[2].legend(fontsize=8)
        
        for i in range(len(ax)):ax[i].grid()
        
        ax[0].set_xlabel('R [m]')
        ax[0].set_ylabel('Z [m]')
        
        ax[1].set_xlabel(r'$\theta$ [rad]')
        ax[1].set_ylabel(r'$r_{%d/%d}$ [m]'%(m,n))
        
        ax[2].set_xlabel(r'$\phi$ [2$\pi$ rad]')
        ax[2].set_ylabel(r'Filament \#%d Coordinatate [m]'%fil)
        plt.show()
    return coords
################# Currents 
def plot_Currents(params,coil_currs,doSave,save_Ext=''):
   m=params['m'];n=params['n'];r=params['r'];R=params['R'];
   n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
   f=params['f'];dt=params['dt'];I=params['I']
   
   hist_file = histfile('floops.hist')
   plt.close('Currents%s'%save_Ext)
   fig,ax=plt.subplots(2,1,tight_layout=True,figsize=(4,4),
               num='Currents%s'%save_Ext,sharex=True)
   times=np.arange(0,periods/f,dt)
   ax[0].plot(times*1e3,coil_currs[:,1],label=r'$\phi=0,\theta=0$')
   # import decimal 
   sensors=['MIRNOV_TOR_SET_160_H1','MIRNOV_TOR_SET_160_V1','MIRNOV_TOR_SET_340_H1']
   # decimal.Decimal(-.25).as_integer_ratio()
   for ind,s in enumerate(sensors):ax[1].plot(hist_file['time']*1e3,hist_file[s],
              label=s)
   ax[0].set_ylabel("I-Mode [A]")
   ax[1].set_ylabel(r'B$_z$ [T]')
   ax[1].set_xlabel("Time [ms]")
   
   
   for i in range(2):
       ax[i].grid()
       ax[i].legend(fontsize=9,loc='upper right')
   if doSave:fig.savefig(doSave+'Filament_and_Field%s.pdf'%save_Ext,
                         transparent=True)  
####################################
  
if __name__=='__main__':
    mesh_file='SPARC_Sept2023_noPR.h5'
    # mesh_file='thincurr_ex-torus.h5'
    sensor_filename='floops_Mirnov.loc'
    gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_filename=sensor_filename)
