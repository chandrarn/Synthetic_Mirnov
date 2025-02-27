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
    h5py, build_torus_bnorm_grid, build_periodic_mesh,write_periodic_mesh, pyvista, subprocess,\
        gethostname, server, I_KM, F_KM
from M3DC1_to_Bnorm import convert_to_Bnorm
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
########################################################
def Synthetic_Mirnov_Surface(mesh_file='SPARC_Sept2023_noPR.h5',doSave='',save_ext='',file_geqdsk=None,
sensor_set='BP',xml_filename='oft_in.xml',params={'m':2,'n':1,'r':.25,'R':1,'n_pts':40,'m_pts':60,\
'f':F_KM,'dt':1e-6,'periods':3,'n_threads':8,'I':I_KM,'T':2e-3},doPlot=True,\
    C1_file='/nobackup1/wenhaw42/Linear/01_n1_test_cases/1000_bate1.0_constbz_0_cp0501/C1.h5'):
    
    # Generate 2D b-norm sin/cos
    if C1_file: convert_to_Bnorm(C1_file,params['n'],params['n_pts'])
    else: __gen_b_norm_manual(file_geqdsk,params)
    
    # Build mode mesh
    __gen_b_norm_mesh('C1' if server else '',params['m_pts'],params['n_pts'],params['n_threads'],
                      sensor_set,doSave,save_ext,params,doPlot)
    
    #return 
    # Build linked inductances and mode/current drivers
    mode_driver, sensor_mode, sensor_obj, tw_torus = \
        __gen_linked_inductances(mesh_file, params['n_threads'], sensor_set)
    
    # Run time dependent calculation
    __run_td(mode_driver,sensor_mode,tw_torus,sensor_obj, params,sensor_set,save_ext)
########################################################
def __run_td(mode_driver,sensor_mode,tw_torus,sensor_obj,params,\
             sensor_set,save_Ext):
    
    mode_growth = 2.E3
    dt = params['dt']
    periods = params['periods']
    nsteps = int(params['T']/dt)
    m = params['m']
    n = params['n']
    # Extending support for time dependent frequencies
    if type(params['f']) == float: mode_freq = params['f']*np.ones((nsteps+1,)) # always just return f
    # assume frequency is a function taken [0,1] as the argument
    else:mode_freq = params['f'](np.linspace(0,1,nsteps+1)) 
    
    if type(params['I']) == int: mode_amp = params['I']*np.ones((nsteps+1,))
    else: mode_amp = params['I'](np.linspace(0,1,nsteps+1)) 
    
    timebase_current = np.arange(0.0,dt*(nsteps+1),dt); 
    timebase_voltage = (timebase_current[1:]+timebase_current[:-1])/2.0
    #fn_ramp = 1# timebase_current/mode_growth
    cos_current = mode_amp*np.cos(mode_freq*2.0*np.pi*timebase_current);
    cos_current[:5] *= np.linspace(0,1,5)
    cos_voltage = np.diff(cos_current)/np.diff(timebase_current)
    sin_current = mode_amp*np.sin(mode_freq*2.0*np.pi*timebase_current);
    sin_current[:5] *= np.linspace(0,1,5)
    sin_voltage = np.diff(sin_current)/np.diff(timebase_current)
    
    volt_full = np.zeros((nsteps+2,tw_torus.nelems+1))
    sensor_signals = np.zeros((nsteps+2,sensor_mode.shape[1]+1))
    
    # Unclear 
    for i in range(nsteps+2):
        volt_full[i,0] = dt*i
        sensor_signals[i,0] = dt*i
        if i > 0: # Leave 0-th step voltage, current, as zero
            volt_full[i,1:] = mode_driver[0,:]*np.interp(volt_full[i,0],timebase_voltage,cos_voltage) \
              + mode_driver[1,:]*np.interp(volt_full[i,0],timebase_voltage,sin_voltage)
              
            sensor_signals[i,1:] = sensor_mode[0,:]*np.interp(volt_full[i,0],timebase_current,cos_current) \
              + sensor_mode[1,:]*np.interp(volt_full[i,0],timebase_current,sin_current)
              
    tw_torus.run_td(dt,nsteps,status_freq=10,full_volts=volt_full,\
                    sensor_obj=sensor_obj,direct=True,sensor_values=sensor_signals)
    
    # Rename output 
    subprocess.run(['cp','floops.hist','data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                    (sensor_set,m,n,mode_freq[0]*1e-3,save_Ext)])
    '''
    hist_file = histfile('data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                         (sensor_set,params['m'],params['n'],\
                          params['f']*1e-3,save_Ext))    
    '''
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
def __gen_b_norm_mesh(C1_file,n_theta,n_phi,n_threads,sensor_set,
                      doSave,save_Ext,params,doPlot=False):
    # Convert B-normal from C1 file LCFS to equivalent suface current via self inductance
    
    # Build mesh on which the surface current lives?
    fname = (C1_file+'_tCurr_mode.dat') if C1_file else 'tCurr_mode.dat'
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
    # Get surface flux from normal field
    flux_flat = bnorm_flat.copy()
    
    if nfp > 1:
        flux_flat[0,r_map] = tw_mode.scale_va(bnorm_flat[0,r_map])
        flux_flat[1,r_map] = tw_mode.scale_va(bnorm_flat[1,r_map])
        tw_mode.save_scalar(bnorm_flat[0,r_map],'Bn_c')
        tw_mode.save_scalar(bnorm_flat[1,r_map],'Bn_s')
        output = np.zeros((2,nelems_new+nfp-1))
        for j in range(2):
            output[j,:nelems_new] = np.dot(Linv,np.r_[flux_flat[j,1:-bnorm.shape[2]],0.0,0.0])
            output[j,-nfp+1:] = output[j,-nfp]
    else:
        flux_flat[0,:] = tw_mode.scale_va(bnorm_flat[0,:])
        flux_flat[1,:] = tw_mode.scale_va(bnorm_flat[1,:])
        tw_mode.save_scalar(bnorm_flat[0,:],'Bn_c')
        tw_mode.save_scalar(bnorm_flat[1,:],'Bn_s')
        output = np.zeros((2,tw_mode.Lmat.shape[0]))
        for j in range(2):
            output[j,:] = np.dot(Linv,np.r_[flux_flat[j,1:],0.0,0.0])
            
    tw_mode.save_current(output[0,:],'Jc')
    tw_mode.save_current(output[1,:],'Js')
    tw_mode.build_XDMF()
    
    with h5py.File('plasma/mesh.0001.h5','r') as h5_file:
        r = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
        
    if doPlot:
        plt.close('Mesh_Orig')
        fig = plt.figure(num='Mesh_Orig')
        ax = fig.add_subplot(1,2,1, projection='3d')
        ax.plot(r_mesh[tnodeset,0], r_mesh[tnodeset,1], r_mesh[tnodeset,2], c='red')
        for pnodeset in pnodesets:
            ax.plot(r_mesh[pnodeset,0], r_mesh[pnodeset,1], r_mesh[pnodeset,2], c='blue')
        
        ax = fig.add_subplot(1,2,2, projection='3d')
        _ = ax.plot_trisurf(r_mesh[:,0], r_mesh[:,1], r_mesh[:,2], triangles=lc_mesh, cmap='viridis')
    #return output
    tw_mode.save_current(output[0,:],'J_c')
    tw_mode.save_current(output[1,:],'J_s')
    # Save surface current
    with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
        h5_file.create_dataset('thincurr/driver', data=output, dtype='f8')
    
    # Plot J(theta,phi)
    #return output,nfp,n_phi,n_theta,bnorm
    if doPlot:__do_plot_B_J(output,nfp,n_phi,n_theta,bnorm,tw_mode,sensor_set,
                            doSave,save_Ext,params)
        
#########################################################
def __do_plot_B_J(output,nfp,nphi,ntheta,bnorm,tw_mode,sensor_set,doSave,
                  save_Ext,params):
    
    # 2D Plot of B-norm and J
    theta=np.linspace(-np.pi,np.pi,ntheta,endpoint=False)/np.pi
    phi=np.linspace(0,2*np.pi,(nphi-1) if nfp>1 else nphi,endpoint=False)/np.pi
    phi_bnorm=np.linspace(0,2*np.pi,nphi,endpoint=False)/np.pi
    plt.close('B-norm_J')
    fig, ax = plt.subplots(2,2,sharex=True,sharey=True,num='B-norm_J')
    redo_inds = np.concatenate(( np.arange(int(len(theta)/2),len(theta)),
                                np.arange(0,int(len(theta)/2)) ))
    #theta=theta[redo_inds]
    if nfp > 1:
        out_1=np.r_[0.0,output[0,:-nfp-1]].reshape((nphi-1,ntheta)).transpose()
        out_2=np.r_[0.0,output[1,:-nfp-1]].reshape((nphi-1,ntheta)).transpose()
    else:
        out_1=np.r_[0.0,output[0,:-nfp-1]].reshape((nphi,ntheta)).transpose()
        out_2=np.r_[0.0,output[1,:-nfp-1]].reshape((nphi,ntheta)).transpose()
    out_1=out_1[redo_inds,:]
    out_2=out_2[redo_inds,:]
    ax[0,0].contour(phi,theta,out_1,50)
    ax[0,1].contour(phi,theta,out_2,50)
    print(phi_bnorm.shape, theta.shape,bnorm.shape,bnorm[0,:,redo_inds].transpose().shape)
    out_1=bnorm[0,:,redo_inds].transpose()
    out_2=bnorm[1,:,redo_inds].transpose()
    print(phi_bnorm.shape,theta.shape, out_1.shape)
    ax[1,0].contour(phi_bnorm,theta,out_1.transpose(),50)
    _ = ax[1,1].contour(phi_bnorm,theta,out_2.transpose(),50)
    
    ax[0,0].set_ylabel(r'$\sigma(\theta)$ [$\pi$-rad]')
    ax[1,0].set_ylabel(r'$B_{\hat{n}}(\theta)$ [$\pi$-rad]')
    for i in range(2):ax[1,i].set_xlabel(r'$\phi$ [$\pi$-rad]')
    ax[0,0].set_title(r'Cosine')
    ax[0,1].set_title('Sine')
    plt.show()
    m=params['m'];n=params['n'];mode_freq=params['f']
    if doSave:fig.savefig(doSave+'Surface_Current_2D_%s_m-n_%d-%d_f_%d%s.pdf'%\
                    (sensor_set,m,n,mode_freq*1e-3,save_Ext))
    
    ###############################
    # mesh plotting
    tw_mode.build_XDMF()
    with h5py.File('plasma/mesh.0001.h5','r') as h5_file:
        r = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
    with h5py.File('plasma/scalar_dump.0001.h5','r') as h5_file:
        bn_c = np.asarray(h5_file['Bn_c0000'])
        bn_s = np.asarray(h5_file['Bn_s0000'])
        bn_c = np.asarray(h5_file['Bn_c0000'])
        J_s = np.asarray(h5_file['J_s_p0000'])
        J_c = np.asarray(h5_file['J_c_p0000'])

    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r)
    
    p = pyvista.Plotter()
    p.add_mesh(grid, color="white", opacity=1.0, show_edges=True,
               scalars=J_c,scalar_bar_args={'title':'J_c'})
    #p.add_scalar_bar(title='J_c')
    # Plot Sensors
    sensors=gen_Sensors_Updated(select_sensor=sensor_set)
    for ind,s in enumerate(sensors):
        p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                     render_points_as_spheres=True,
                     label='Sensor' if ind==0 else None)
    if doSave:p.save_graphic(doSave+'Surface_J_s_%s_m-n_%d-%d_f_%d%s.pdf'%\
                    (sensor_set,m,n,mode_freq*1e-3,save_Ext))
    p.show()

#########################################################
def __gen_b_norm_manual(file_geqdsk,params,orig_example_bnorm=True,doPlot=True):
    
    if orig_example_bnorm:
        def create_circular_bnorm(filename,R0,Z0,a,n,m,npts=200,streach=0.05):
            theta_vals = np.linspace(0.0,2*np.pi,npts,endpoint=False)
            a_local = lambda theta: a +streach*a*np.abs(np.sin(theta))
            with open(filename,'w+') as fid:
                fid.write('{0} {1}\n'.format(npts,n))
                for theta in theta_vals:
                    fid.write('{0} {1} {2} {3}\n'.format(
                        R0+a*np.cos(theta),
                        Z0+a_local(theta)*np.sin(theta),
                        np.cos(m*theta),
                        np.sin(m*theta)
                    ))
        # Create n=2, m=3 mode
        create_circular_bnorm('tCurr_mode.dat',1.8,0.0,0.4,1,2)
        

    else:
        a=params['r'];R=params['R'];m=params['m'];n=params['n']
        n_pts_theta=params['m_pts'];n_pts_n=params['n_pts']
        
        # Get radial vector
        n_pts_n=200
        a=0.4;R=1
        r_theta,Z0,R0 = __gen_r_theta(file_geqdsk, a, R, m, n)
        theta = np.linspace(0,2*np.pi,n_pts_n,endpoint=False)
        if doPlot:
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
        #r_theta = lambda theta: a +0.01*np.abs(np.sin(theta)) # running circular approximation
        r_theta = lambda theta: np.sqrt( (a*np.cos(theta))**2 + ( a*1.05*np.sin(theta))**2)
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