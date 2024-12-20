#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:43:59 2024
 Basic wrapper code for arbitrary fillament generation and mesh integration
@author: rian
"""

# header
from header import *

# main loop
def gen_synthetic_Mirnov(input_file='',mesh_file='thincurr_ex-torus.h5',
                         xml_filename='oft_in.xml',sensor_filename='floops.loc',\
                            params={'m':2,'n':1,'r':.25,'R':1,'n_pts':200,'m_pts':60,\
                            'f':1e2,'dt':5e-4,'periods':4,'n_threads':8,'I':10},
                                doSave='',save_ext=''):
    
    # Generate sensors, filamanets
    gen_filaments(xml_filename,params)
    sensors = gen_sensors()
    # Get Mesh
    tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv = \
        get_mesh(mesh_file,xml_filename,sensor_filename,params)

    # Get mode amplitudes, assign to fillaments [eventually, from simulation]
    
    # Run time dependent simulation
    coil_currs = run_td(sensor_obj,tw_mesh,params)
    
    makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_ext,Mc, L_inv)

    return sensors, coil_currs
#    return sensor_,currents
#####################################3
def get_mesh(mesh_file,filament_file,sensor_file,params,doPlot=True):
    tw_mesh = ThinCurr(nthreads=params['n_threads'])
    tw_mesh.setup_model(mesh_file=mesh_file,xml_filename=filament_file)
    tw_mesh.setup_io()
    
    # Sensor - mesh and sensor - filament inductances
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor(sensor_file)
    # Filament - mesh inductance
    Mc = tw_mesh.compute_Mcoil()
    
    # Build inductance matrix
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file='HOLDR_L.save')
    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction
    eig_vals, eig_vecs = tw_mesh.get_eigs(100,False)
    L_inv = 0#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))

    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

####################################
def gen_filaments(filament_file,params):
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    n_pts=params['n_pts'];m_pts=params['m_pts']
    theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    with open(filament_file,'w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>1.2E-5</eta>\n\t<icoils>\n')
        
        for theta in theta_:
            f.write('\t<coil_set>\n\t\t<coil npts="%d" scale="1.0">\n'%n_pts)
            for phi in phi_:
                f.write('\t\t\t %1.1f, %1.1f, %1.1f\n'%(\
                              calc_filament_coord(m,n,r,R,theta,phi)))
            f.write('\t\t</coil>\n\t</coil_set>\n')
        f.write('\t</icoils>\n\t</thincurr>\n</oft>')
    # Assume we have the shaping jacobian for theta into minor radial corrdinanat
    # Eventually manually calc points
    
   
####################################
def gen_sensors():
    sensors = []
    for i, theta in enumerate(np.linspace(0.0-np.pi/4.2,2.0*np.pi/2.0-np.pi/4.2,5)):
        sensors.append(Mirnov(1.4*np.r_[np.cos(theta),np.sin(theta),0.0], np.r_[0.0,0.0,1.0], 'Bz_{0}'.format(i+1)))
    save_sensors(sensors)
    return sensors
####################################
def run_td(sensor_obj,tw_mesh,param,doPlot=True):
    m_pts=param['m_pts'];m=param['m'];dt=param['dt'];f=param['f'];periods=param['periods']
    I=param['I'];n=param['n'];n_pts=param['n_pts']
    theta_,phi_= gen_filament_coords(m, n, n_pts, m_pts)
    coil_currs = np.zeros((int(periods/f/dt),m_pts+1))
    for ind,t in enumerate(np.arange(0,periods/f,dt)):
        coil_currs[ind,1:]=[I*np.cos(m*theta+t*f*2*np.pi) for theta in theta_]

    coil_currs[:,0]=np.arange(0,periods/f,dt)
    #print(coil_currs)
    
    #coil_currs[:,-2]=0
    #coil_currs[:,-1]=0
    tw_mesh.run_td(dt,int(periods/f/dt),
                    coil_currs=coil_currs,sensor_obj=sensor_obj)
    tw_mesh.plot_td(int(periods/f/dt),compute_B=False,sensor_obj=sensor_obj)
    
    _, Bc = tw_mesh.compute_Bmat(cache_file='HODLR_B.save')
       
    return coil_currs
########################
def makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_Ext,Mc, L_inv, t_pt=5,):
    
    # MEsh and Filaments
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
    f=params['f'];dt=params['dt'];I=params['I']

    # Calculate induced current in mesh
    
    #tw_plate.save_current(np.dot(Linv,M[0]),'M')
    # tw_mesh.save_current(np.dot(L_inv,np.dot(Mc.T,coil_currs[0,1:])), 'Ind_Curr')
    tw_mesh.build_XDMF()
    with h5py.File('mesh.0001.h5','r') as h5_file:
        r_ = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
    with h5py.File('vector_dump.0001.h5') as h5_file:
        Jfull = np.asarray(h5_file['J_v0003'])
        scale = 0.2/(np.linalg.norm(Jfull,axis=1)).max()
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r_)

    p = pyvista.Plotter()  
    
    # Plot Mesh
    p.add_mesh(grid,color="white",opacity=.9,show_edges=True,scalars=Jfull)
    slices=grid.slice_orthogonal()
    #slices.plot()
    #p.add_mesh(slices,label='Mesh Frame')
    
    tmp=[]
    
    # Plot Filaments
    colors=['red','green','blue']
    theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    for ind,theta in enumerate(theta_):
        pts = []
        for phi in phi_:
            pts.append(calc_filament_coord(m,n,r,R,theta,phi))
        print(np.array(calc_filament_coord(m,n,r,R,theta,phi)).shape)
        pts=np.array(pts)
        print(pts.shape,theta)
        spl=pyvista.Spline(pts,100)
        #p.add_(spline,render_lines_as_tubes=True,line_width=5,show_scalar_bar=False)
        #p.add_mesh(spl,opacity=1,line_width=6,color=plt.get_cmap('viridis')(theta*m/(2*np.pi)))
        slices_spl=spl;#spl.slice_orthogonal()
        p.add_mesh(slices_spl,color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2),
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
    
    ################# Currents 
    hist_file = histfile('floops.hist')
    plt.close('Currents%s'%save_Ext)
    fig,ax=plt.subplots(2,1,tight_layout=True,figsize=(4,4),
                num='Currents%s'%save_Ext,sharex=True)
    times=np.arange(0,periods/f,dt)
    ax[0].plot(times*1e3,coil_currs[:,1],label=r'$\phi=0,\theta=0$')
    ax[1].plot(hist_file['time']*1e3,hist_file['Bz_1'],
               label=r'Sensor $\phi=0$')
    ax[1].plot(hist_file['time']*1e3,hist_file['Bz_2'],
               label=r'Sensor $\phi=\pi/30$')
    ax[1].plot(hist_file['time']*1e3,hist_file['Bz_3'],
               label=r'Sensor $\phi=\pi/15$')
    ax[0].set_ylabel("I-Mode [A]")
    ax[1].set_ylabel(r'B$_z$ [T]')
    ax[1].set_xlabel("Time [ms]")
    
    
    for i in range(2):
        ax[i].grid()
        ax[i].legend(fontsize=9,loc='upper right')
    if doSave:fig.savefig(doSave+'Filament_and_Field%s.pdf'%save_Ext,
                          transparent=True)
          
    plt.show()
########################
def gen_filament_coords(m,n,n_pts,m_pts):
    return np.linspace(0,2*np.pi/m,m_pts,endpoint=True),\
        np.linspace(0,m*2*np.pi/n,n_pts,endpoint=True)
##############################
def calc_filament_coord(m,n,r,R,theta,phi):
    # Locally: evolve theta as n*phi/m
    # Treat theta as non-evolved offset parameter (to set initial angle)
    return (R+r*np.cos(n*phi/m + theta))*np.cos(n*phi), \
            (R+r*np.cos(n*phi/m+theta))*np.sin(n*phi), r*np.sin(n*phi/m+theta)
  ########################      
if __name__=='__main__':gen_synthetic_Mirnov()