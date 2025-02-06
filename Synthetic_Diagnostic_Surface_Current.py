#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:23:27 2025
    Surface current mode driver
    - Produce surface mesh and sin/cos structure
        - Either gen b-norm, or get from simulation
        - plot tous b-norm
    - Load VV mesh, compute linked inductances
    - Generate voltage, current time traces (?)
    - Run time dependnt simulation
    
@author: rian
"""

from header import np, plt, ThinCurr, histfile, geqdsk, cv2,make_smoothing_spline,\
    h5py, build_torus_bnorm_grid, build_periodic_mesh,write_periodic_mesh, pyvista
from M3DC1_to_Bnorm import convert_to_Bnorm
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
########################################################
def Synthetic_Mirnov_Surface(mesh_file='SPARC_Sept2023_noPR.h5',doSave='',save_ext='',file_geqdsk='geqdsk',
sensor_set='MRNV',xml_filename='oft_in.xml',params={'m':2,'n':1,'r':.25,'R':1,'n_pts':70,'m_pts':60,\
'f':500e3,'dt':1e-7,'periods':3,'n_threads':64,'I':10},C1_file='',doPlot=True):
    
    # Generate 2D b-norm sin/cos
    if C1_file: convert_to_Bnorm(C1_file,params['n'],params['n_pts'])
    else: __gen_b_norm_manual(file_geqdsk,params)
    
    # Build mode mesh
    __gen_b_norm_mesh(C1_file,params['m_pts'],params['n_pts'],params['n_threads'],doPlot)
    
    return
    # Build linked inductances and mode/current drivers
    mode_driver, sensor_mode, sensor_obj, tw_torus = \
        __gen_linked_inductances(mesh_file, params['n_threads'], sensor_set)
    
    # Run time dependent calculation
    __run_td(mode_driver,sensor_mode,tw_torus,sensor_obj, sensor_set,save_ext)
########################################################
def __run_td(mode_driver,sensor_mode,tw_torus,sensor_obj,params,\
             sensor_set,save_Ext):
    mode_freq = params['f']
    mode_growth = 2.E3
    dt = params['dt']
    nsteps = params['n_pts']
    
    timebase_current = np.arange(0.0,dt*nsteps+1,dt/4.0); 
    timebase_voltage = (timebase_current[1:]+timebase_current[:-1])/2.0
    
    cos_current = timebase_current/mode_growth*np.cos(mode_freq*2.0*np.pi*timebase_current);
    cos_voltage = np.diff(cos_current)/np.diff(timebase_current)
    sin_current = timebase_current/mode_growth*np.sin(mode_freq*2.0*np.pi*timebase_current);
    sin_voltage = np.diff(sin_current)/np.diff(timebase_current)
    
    volt_full = np.zeros((nsteps+2,tw_torus.nelems+1))
    sensor_signals = np.zeros((nsteps+2,sensor_mode.shape[1]+1))
    
    # Unclear 
    for i in range(nsteps+2):
        volt_full[i,0] = dt*i
        sensor_signals[i,0] = dt*i
        if i > 0:
            volt_full[i,1:] = mode_driver[0,:]*np.interp(volt_full[i,0],timebase_voltage,cos_voltage) \
              + mode_driver[1,:]*np.interp(volt_full[i,0],timebase_voltage,sin_voltage)
              
            sensor_signals[i,1:] = sensor_mode[0,:]*np.interp(volt_full[i,0],timebase_current,cos_current) \
              + sensor_mode[1,:]*np.interp(volt_full[i,0],timebase_current,sin_current)
              
    tw_torus.run_td(dt,nsteps,status_freq=10,full_volts=volt_full,\
                    sensor_obj=sensor_obj,direct=True,sensor_values=sensor_signals)
    
    hist_file = histfile('data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                         (sensor_set,params['m'],params['n'],\
                          params['f']*1e-3,save_Ext))    
    
#########################################################
def __gen_linked_inductances(mesh_file,n_threads,sensor_set):
    
    # Prep mesh resistances 
    eta = '1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5'
    with open('oft_in.xml','w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n'%eta)
        
    tw_torus = ThinCurr(nthreads=n_threads)
    tw_torus.setup_model(mesh_file=mesh_file,xml_filename='oft_in.xml')
    tw_torus.setup_io()
    
    # Gen sensors
    sensors=gen_Sensors_Updated(select_sensor=sensor_set)
    Msensor, Msc, sensor_obj = tw_torus.compute_Msensor('floops_%s.loc'%sensor_set)

    Mc = tw_torus.compute_Mcoil()
    tw_torus.compute_Lmat()
    tw_torus.compute_Rmat()

    tw_mode = ThinCurr(nthreads=n_threads)
    tw_mode.setup_model(mesh_file='thincurr_mode.h5')
    with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
        mode_drive = np.asarray(h5_file['thincurr/driver'])

    Msensor_plasma, _, _ = tw_mode.compute_Msensor('floops_%s.loc'%sensor_set)
    # Significance? 
    mode_driver = tw_mode.cross_eval(tw_torus,mode_drive)
    sensor_mode = np.dot(mode_drive,Msensor_plasma)
    
    return mode_driver, sensor_mode, sensor_obj, tw_torus
#########################################################
def __gen_b_norm_mesh(C1_file,n_theta,n_phi,n_threads,doPlot=False):
    # Convert B-normal from C1 file LCFS to equivalent suface current via self inductance
    
    # Build mesh on which the surface current lives?
    fname = (C1_file+'tCurr_mode.dat') if C1_file else 'tCurr_mode.dat'
    # Extend 2D mode slice to toroidal grid of points
    r_grid, bnorm, nfp = build_torus_bnorm_grid(fname,n_theta,n_phi,
                            resample_type='theta',use_spline=False)
    # Build a mesh supporting the above grid
    lc_mesh, r_mesh, tnodeset, pnodesets, r_map = build_periodic_mesh(r_grid,nfp)
    
    write_periodic_mesh('thincurr_mode.h5', r_mesh, lc_mesh+1, np.ones((lc_mesh.shape[0],)),
                        tnodeset, pnodesets, pmap=r_map, nfp=nfp, include_closures=True)

    #return r_grid, bnorm, nfp,lc_mesh,r_mesh,tnodeset,pnodesets,r_map
    # Plasma vs Driver? 
    # Convert from B-norm to J(theta) using self inductance
    tw_mode = ThinCurr(nthreads= n_threads)
    tw_mode.setup_model(mesh_file='thincurr_mode.h5')
    tw_mode.setup_io(basepath='plasma/')

    tw_mode.compute_Lmat()
    # Condense self inductance model to single mode period [necessary for mesh periodicity reasons]
    if nfp > 1:
        nelems_new = tw_mode.Lmat.shape[0]-nfp+1
        Lmat_new = np.zeros((nelems_new,nelems_new))
        Lmat_new[:-1,:-1] = tw_mode.Lmat[:-nfp,:-nfp]
        Lmat_new[:-1,-1] = tw_mode.Lmat[:-nfp,-nfp:].sum(axis=1)
        Lmat_new[-1,:-1] = tw_mode.Lmat[-nfp:,:-nfp].sum(axis=0)
        Lmat_new[-1,-1] = tw_mode.Lmat[-nfp:,-nfp:].sum(axis=None)
    else:
        Lmat_new = tw_mode.Lmat
    # Get inverse
    Linv = np.linalg.inv(Lmat_new)

    bnorm_flat = bnorm.reshape((2,bnorm.shape[1]*bnorm.shape[2]))
    # Get surface flux from normal field [scale B-field by mesh vertex area va]
    flux_flat = bnorm_flat.copy()
    #print(bnorm_flat.shape,bnorm_flat[0,r_map].shape,flux_flat[0,r_map].shape)
    flux_flat[0,r_map] = tw_mode.scale_va(bnorm_flat[0,r_map].squeeze())
    flux_flat[1,r_map] = tw_mode.scale_va(bnorm_flat[1,r_map].squeeze())
    # Field periodicitry issue again
    if nfp > 1:
        tw_mode.save_scalar(bnorm_flat[0,r_map],'Bn_c')
        tw_mode.save_scalar(bnorm_flat[1,r_map],'Bn_s')
        output = np.zeros((2,nelems_new+nfp-1))
        for j in range(2):
            # Dot product of L^-1 with surface flux gives surface current
            output[j,:nelems_new] = np.dot(Linv,np.r_[flux_flat[j,1:-bnorm.shape[2]],0.0,0.0]) # 0?
            output[j,-nfp+1:] = output[j,-nfp]
    else:
        tw_mode.save_scalar(bnorm_flat[0,:],'Bn_c')
        tw_mode.save_scalar(bnorm_flat[1,:],'Bn_s')
        output = np.zeros((2,tw_mode.Lmat.shape[0]))
        for j in range(2):
            output[j,:] = np.dot(Linv,np.r_[flux_flat[j,1:],0.0,0.0]) 
    
    # Save surface current
    with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
        h5_file.create_dataset('thincurr/driver', data=output, dtype='f8')
    
    # Plot J(theta,phi)
    #return output,nfp,n_phi,n_theta,bnorm
    if doPlot:__do_plot_B_J(output,nfp,n_phi,n_theta,bnorm)
        
#########################################################
def __do_plot_B_J(output,nfp,nphi,ntheta,bnorm):
    
    # 2D Plot of B-norm and J
    theta=np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi=np.linspace(0,2*np.pi,(nphi-1) if nfp>1 else nphi,endpoint=False)
    phi_bnorm=np.linspace(0,2*np.pi,nphi,endpoint=False)
    plt.close('B-norm_J')
    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,num='B-norm_J')
    if nfp > 1:
        ax[0,0].contour(phi,theta,np.r_[0.0,output[0,:-nfp-1]].reshape((nphi-1,ntheta)).transpose(),30)
        ax[0,1].contour(phi,theta,np.r_[0.0,output[1,:-nfp-1]].reshape((nphi-1,ntheta)).transpose(),30)
    else:
        ax[0,0].contour(phi,theta,np.r_[0.0,output[0,:-nfp-1]].reshape((nphi,ntheta)).transpose(),30)
        ax[0,1].contour(phi,theta,np.r_[0.0,output[1,:-nfp-1]].reshape((nphi,ntheta)).transpose(),30)
    ax[1,0].contour(phi_bnorm,theta,bnorm[0,:,:].transpose(),10)
    _ = ax[1,1].contour(phi_bnorm,theta,bnorm[1,:,:].transpose(),10)
    
    ax[0,0].set_ylabel(r'$\sigma(\theta)$')
    ax[1,0].set_ylabel(r'$B_\perp(\theta)$')
    for i in range(2):ax[1,i].set_xlabel(r'$\phi$')
    plt.show()
    
    
    ###############################
    # mesh plotting
    with h5py.File('mesh.0001.h5','r') as h5_file:
        r = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
    
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r)
    
    p = pyvista.Plotter()
    p.add_mesh(grid, color="white", opacity=1.0, show_edges=True)
    p.show()

#########################################################
def __gen_b_norm_manual(file_geqdsk,params,orig_example_bnorm=False):
    
    if orig_example_bnorm:
        def create_circular_bnorm(filename,R0,Z0,a,n,m,npts=params['n_pts']):
            theta_vals = np.linspace(0.0,2*np.pi,npts,endpoint=False)
            with open(filename,'w+') as fid:
                fid.write('{0} {1}\n'.format(npts,n))
                for theta in theta_vals:
                    fid.write('{0} {1} {2} {3}\n'.format(
                        R0+a*np.cos(theta),
                        Z0+a*np.sin(theta),
                        np.cos(m*theta),
                        np.sin(m*theta)
                    ))
        # Create n=2, m=3 mode
        create_circular_bnorm('tCurr_mode.dat',1.0,0.0,0.4,2,3)

    else:
        a=params['r'];R=params['R'];m=params['m'];n=params['n']
        n_pts_theta=params['m_pts'];n_pts_n=params['n_pts']
        
        # Get radial vector
        r_theta,Z0,R0 = __gen_r_theta(file_geqdsk, a, R, m, n)
        theta = np.linspace(0,2*np.pi,n_pts_n,endpoint=False)
        plt.figure(tight_layout=True);
        plt.plot(r_theta(theta)*np.cos(theta)+R0,r_theta(theta)*np.sin(theta)+Z0)
        plt.xlabel('R [m]');plt.ylabel('Z [m]')
        plt.legend([r'$\psi_{%d/%d}(\vec{r})$'%(m,n)],fontsize=8)
        plt.grid()
        plt.show()
        # Write output
        with open('tCurr_mode.dat','w+') as fid:
            fid.write('{0} {1}\n'.format(n_pts_n,n))
            for theta in theta:
                fid.write('{0} {1} {2} {3}\n'.format(
                    R0+r_theta(theta)*np.cos(theta),
                    Z0+r_theta(theta)*np.sin(theta),
                    np.cos(m*theta),
                    np.sin(m*theta)
                ))

#########################################################        
def __gen_r_theta(file_geqdsk,a,R,m,n,):
    if file_geqdsk is None:
        r_theta = lambda theta: a #-0.1*np.cos(2*theta) # running circular approximation
        zmagx=0;rmagx=R
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
        
        zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
    
    return r_theta, zmagx,rmagx
    
#########################################################
if __name__=='__main__':Synthetic_Mirnov_Surface()