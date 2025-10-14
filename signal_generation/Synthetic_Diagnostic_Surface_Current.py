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

from header_signal_generation import np, plt, ThinCurr, histfile, geqdsk, cv2,make_smoothing_spline,\
    h5py, build_torus_bnorm_grid, OFT_env,ThinCurr_periodic_toroid, pyvista, subprocess,\
        gethostname, server, I_KM, F_KM, F_KM_plot, build_periodic_mesh,write_periodic_mesh, cm, Normalize
from M3DC1_to_Bnorm import convert_to_Bnorm
from FAR3D_to_Bnorm import convert_FAR3D_to_Bnorm, __gen_b_norm_manual
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
########################################################
def Synthetic_Mirnov_Surface(mesh_file='SPARC_Sept2023_noPR.h5',doSave='',save_ext='',file_geqdsk=None,
    sensor_set='BP',xml_filename='oft_in.xml',params={'m':2,'n':1,'r':.25,'R':1,'n_pts':40,'m_pts':60,\
    'f':F_KM,'dt':1e-6,'periods':3,'n_threads':8,'I':I_KM,'T':1e-3},doPlot=True,\
    C1_file='/nobackup1/wenhaw42/SPARC_dir/Linear/01_n1_test_cases/1000_bate1.0_constbz_0_cp0501/C1.h5',\
    FAR3D_br_file='',FAR3D_bth_file='',plotOnly=False,eta='1.8E-5, 3.6E-5, 2.4E-5',doSave_Bode=False,\
        scan_in_freq=False,psi_contour_select=0.8):
    
   
    # Initialize OFT environment
    oft_env = OFT_env(nthreads=params['n_threads'])

    # Generate oft .xml file with resistivity values for mesh (no filament necessary here)
    gen_OFT_filemant_and_eta_file(xml_filename, eta)

    # Gen sensors location file
    sensors=gen_Sensors_Updated(select_sensor=sensor_set)

    
    


    # Generate 2D b-norm sin/cos
    if C1_file: convert_to_Bnorm(C1_file,params['n'],params['n_pts'])
    elif FAR3D_br_file: convert_FAR3D_to_Bnorm(FAR3D_br_file,FAR3D_bth_file,m=params['m'],n=params['n'],\
                                               eqdsk_file=file_geqdsk,debug=True,psi_contour_select=psi_contour_select)
    else: __gen_b_norm_manual(file_geqdsk,params)
    # __gen_b_norm_manual(file_geqdsk,params)

    # Build mode mesh [creates inputdata/thincurr_mode.h5]
    prefix = ('C1' if C1_file else 'FAR3D') if server else ''
    __gen_b_norm_mesh(prefix,params['m_pts'],params['n_pts'],oft_env,
                      sensor_set,doSave,save_ext,params,doPlot)
    
    print('Halt')
    # Build linked inductances and mode/current drivers 
    mode_driver_flux,mode_drive_currents, sensor_mode, sensor_obj, tw_mesh, tw_mode,\
          Msensor, Msensor_plasma = \
        get_mesh_and_driver(mesh_file, oft_env, sensor_set)
    
    # Run time dependent calculation
    if not plotOnly:
            # pt, norm = (0,0,0), (0,0,1)
            # sensor_all.append(Mirnov(pt, norm, 'C_MOD_CENTER_CONTROL', dx))
        if scan_in_freq:
            run_frequency_scan(tw_mesh, tw_mode, mode_driver_flux, mode_drive_currents, Msensor, \
                               Msensor_plasma, sensor_obj, doSave_Bode, save_ext)
        else:
            __run_td(mode_driver_flux, sensor_mode, tw_mesh, sensor_obj, \
                      params, sensor_set, save_ext)

    print('Done')
########################################################
def __run_td(mode_driver_flux,sensor_mode,tw_mesh,sensor_obj,params,\
             sensor_set,save_Ext):
    
    mode_growth = 2.E3
    dt = params['dt']
    periods = params['periods']
    nsteps = int(params['T']/dt)
    m = params['m']
    n = params['n']
    
    timebase_current = np.arange(0.0,dt*(nsteps+1),dt); 
    timebase_voltage = (timebase_current[1:]+timebase_current[:-1])/2.0
    
    # Extending support for time dependent frequencies
    if type(params['f']) == float: mode_freq = params['f']*2*np.pi*timebase_current # always just return f
    # assume frequency is a function taken [0,1] as the argument
    else:mode_freq = params['f'](timebase_current) 
    
    if type(params['I']) == int: mode_amp = params['I']*np.ones((nsteps+1,))
    else: mode_amp = params['I'](timebase_current) 
    

    #fn_ramp = 1# timebase_current/mode_growth
    cos_current = mode_amp*np.cos(mode_freq);
    cos_current[:5] *= np.linspace(0,1,5)
    cos_voltage = np.diff(cos_current)/np.diff(timebase_current)
    sin_current = mode_amp*np.sin(mode_freq);
    sin_current[:5] *= np.linspace(0,1,5)
    sin_voltage = np.diff(sin_current)/np.diff(timebase_current)
    
    volt_full = np.zeros((nsteps+2,tw_mesh.nelems+1))
    sensor_signals = np.zeros((nsteps+2,sensor_mode.shape[1]+1))
    
    # Unclear 
    for i in range(nsteps+2):
        volt_full[i,0] = dt*i
        sensor_signals[i,0] = dt*i
        if i > 0: # Leave 0-th step voltage, current, as zero
            volt_full[i,1:] = mode_driver_flux[0,:]*np.interp(volt_full[i,0],timebase_voltage,cos_voltage) \
              + mode_driver_flux[1,:]*np.interp(volt_full[i,0],timebase_voltage,sin_voltage)
              
            sensor_signals[i,1:] = sensor_mode[0,:]*np.interp(volt_full[i,0],timebase_current,cos_current) \
              + sensor_mode[1,:]*np.interp(volt_full[i,0],timebase_current,sin_current)
              
    tw_mesh.run_td(dt,nsteps,status_freq=10,full_volts=volt_full,\
                    sensor_obj=sensor_obj,direct=True,sensor_values=sensor_signals)
    
    # Rename output 
    f_save = '../data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                    (sensor_set,m if type(m) is int else m[0],n,F_KM_plot(0)*1e-3,save_Ext)
    subprocess.run(['cp','floops.hist',f_save])
    print('Saved: %s'%f_save)
    '''
    hist_file = histfile('data_output/floops_surface_%s_m-n_%d-%d_f_%d%s.hist'%\
                         (sensor_set,params['m'],params['n'],\
                          params['f']*1e-3,save_Ext))    
    '''
################################################################################################
################################################################################################
def run_frequency_scan(tw_mesh,tw_mode, mode_driver_flux, mode_drive_currents,Msensor,\
                        Msensor_plasma, sensor_obj,doSave_Bode,save_ext):

    # Run frequency dependent scan using conducting mesh, mode surface mesh/currents

    freqs = params['f'] # Frequency range for the scan

    # Data vector storage
    sensors_bode = []
    # Loop through frequencies to test
    for freq in freqs if np.ndim(freqs) > 0 else [freqs]:
        
        result = tw_mesh.compute_freq_response(fdriver=mode_driver_flux,freq=freq)
        result[np.isnan(result)] = 0 # Remove NaNs
        # contribution from the mesh current to the sensor, with the mesh current at a given frequency
        probe_signals = np.dot(result,Msensor) + np.dot(mode_drive_currents,Msensor_plasma)
        # Contribuio from the coil current directly to the sensor

        # probe_signals[0,:] += np.dot(np.r_[coil_current_magnitude],Msc)

        for i in range(probe_signals.shape[1]):
            print('Real: {0:13.5E}, Imaginary: {1:13.5E}'.format(*probe_signals[:,i]))
        probe_signals = probe_signals[0,:] + 1j*probe_signals[1,:] # Combine real and imaginary parts   
        sensors_bode.append(probe_signals)
    
    # Save the probe signals to a file
    if doSave_Bode: 
        freqs = [freqs] if np.ndim(freqs) == 0 else freqs
        fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz%s_Surface_Current.npz'%(mesh_file,sensor_set, freqs[0], freqs[-1],save_ext)
        np.savez('../data_output/'+fName, probe_signals=sensors_bode, freqs=freqs,\
                 sensor_names=sensor_obj['names']) 
    
        # Plot the probe signals
        print('Saved frequency scan data to %s'%fName)

    # Only compute the mesh induced currents once
    tw_mesh.save_current(result[0,:],'Jr_mode')
    tw_mesh.save_current(result[1,:],'Ji_mode')
    _ = tw_mesh.build_XDMF()


    return np.array(sensors_bode)

################################################################################################
#########################################################
def gen_OFT_filemant_and_eta_file(xml_filename, eta):
    # Prep mesh resistances in OFT file format
    # No filament is necessary for the case of just surface currents

    with open('input_data/'+xml_filename,'w+') as f:
        print('File open for writing: %s'%xml_filename)
        f.write(f'<oft>\n\t<thincurr>\n\t\t<eta>{eta}</eta>\n\t\t<icoils>\n\t\t</icoils>\n\t</thincurr>\n</oft>')
################################################################################################
#########################################################
def get_mesh_and_driver(mesh_file,oft_env,sensor_set):
    
    # # Pull in the conducting structure mesh
    tw_mesh = ThinCurr(oft_env)
    
    tw_mesh.setup_model(mesh_file='input_data/'+mesh_file,
                         xml_filename='input_data/oft_in.xml')
    tw_mesh.setup_io()
    
    # Link the conducting structures to the sensors
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor('input_data/floops_%s.loc'%sensor_set)


    # Build self inductance and resistances
    Mc = tw_mesh.compute_Mcoil() # Inductance to the coils not necessary for surface current case
    tw_mesh.compute_Lmat()
    tw_mesh.compute_Rmat()

    # Load in the surface current mode driver
    tw_mode = ThinCurr(oft_env)
    tw_mode.setup_model(mesh_file='input_data/thincurr_mode.h5')
    with h5py.File('input_data/thincurr_mode.h5', 'r+') as h5_file:
        mode_drive_currents = np.asarray(h5_file['thincurr/driver'])

    # Link the inductance the mode and the sensors
    Msensor_plasma, _, _ = tw_mode.compute_Msensor('input_data/floops_%s.loc'%sensor_set)

    # link the mode to the conducting structure mesh
    mode_driver_flux = tw_mode.cross_eval(tw_mesh,mode_drive_currents)
    sensor_mode = np.dot(mode_drive_currents,Msensor_plasma)
    
    return mode_driver_flux,mode_drive_currents, sensor_mode, sensor_obj, tw_mesh, tw_mode, Msensor, Msensor_plasma
#########################################################
def __gen_b_norm_mesh(C1_file,n_theta,n_phi,oft_env,sensor_set,
                      doSave,save_Ext,params,doPlot):
    # Convert B-normal from C1 or FAR3D file LCFS to equivalent suface current via self inductance
    

    # Build mesh on which the surface current lives
    fname = (C1_file+'_tCurr_mode.dat') if C1_file else 'input_data/tCurr_mode.dat'
    # Extend 2D mode slice to toroidal grid of points
    # Generate Mesh
    r_grid, bnorm, nfp = build_torus_bnorm_grid(fname,n_theta,n_phi,
                            resample_type='theta',use_spline=False)
    
    __b_norm_surf_plot(r_grid,bnorm,nfp,n_theta,n_phi,params=params,doSave=doSave,
                       save_Ext=save_Ext)
    
    # Upgraded method
    plasma_mode = ThinCurr_periodic_toroid(r_grid,nfp,n_theta,n_phi)
    plasma_mode.write_to_file('input_data/thincurr_mode.h5')

    # Plot Mesh
    fig = plt.figure(figsize=(12,5))
    plasma_mode.plot_mesh(fig)
    plt.show()
  
    # Convert from toroidal grid of periodic B-norm to J(theta) using self inductance
    tw_mode = ThinCurr(oft_env)
    tw_mode.setup_model(mesh_file='input_data/thincurr_mode.h5')
    tw_mode.setup_io(basepath='plasma/')

    tw_mode.compute_Lmat()
    # Collapse periodic copies into final inductance matrix for DoF of periodic system (?)
    Lmat_new = plasma_mode.condense_matrix(tw_mode.Lmat)

    # Get inverse
    Linv = np.linalg.inv(Lmat_new)

    # Compute currents and fluxes and save
    bnorm_flat = bnorm.reshape((2,bnorm.shape[1]*bnorm.shape[2]))
    # Get surface flux from normal field
    flux_flat = bnorm_flat.copy()
    
    flux_flat[0,plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[0,plasma_mode.r_map])
    flux_flat[1,plasma_mode.r_map] = tw_mode.scale_va(bnorm_flat[1,plasma_mode.r_map])
    tw_mode.save_scalar(bnorm_flat[0,plasma_mode.r_map],'Bn_c')
    tw_mode.save_scalar(bnorm_flat[1,plasma_mode.r_map],'Bn_s')
    output_full = np.zeros((2,tw_mode.nelems))
    output_unique = np.zeros((2,Linv.shape[0]))
    for j in range(2):
        output_unique[j,:] = np.dot(Linv,plasma_mode.nodes_to_unique(flux_flat[j,:]))
        output_full[j,:] = plasma_mode.expand_vector(output_unique[j,:])
    
    tw_mode.save_current(output_full[0,:],'Jc'),
    tw_mode.save_current(output_full[1,:],'Js')
    _ = tw_mode.build_XDMF()

    

    # with h5py.File('plasma/mesh.0001.h5','r') as h5_file:
    #     r = np.asarray(h5_file['R_surf'])
    #     lc = np.asarray(h5_file['LC_surf'])
        
    if doPlot:
        plt.close('Mesh_Orig')
        fig = plt.figure(num='Mesh_Orig')
        plasma_mode.plot_mesh(fig)
        # ax = fig.add_subplot(1,2,1, projection='3d')
        # ax.plot(r_mesh[tnodeset,0], r_mesh[tnodeset,1], r_mesh[tnodeset,2], c='red')
        # for pnodeset in pnodesets:
        #     ax.plot(r_mesh[pnodeset,0], r_mesh[pnodeset,1], r_mesh[pnodeset,2], c='blue')
        
        # ax = fig.add_subplot(1,2,2, projection='3d')
        # _ = ax.plot_trisurf(r_mesh[:,0], r_mesh[:,1], r_mesh[:,2], triangles=lc_mesh, cmap='viridis')
    # #return output
    # tw_mode.save_current(output[0,:],'J_c')
    # tw_mode.save_current(output[1,:],'J_s')
    # # Save surface current
    with h5py.File('input_data/thincurr_mode.h5', 'r+') as h5_file:
        h5_file.create_dataset('thincurr/driver', data=output_full, dtype='f8')
    
    # Plot J(theta,phi)
    #return output,nfp,n_phi,n_theta,bnorm
    if doPlot:
        # Pull in the conducting structure mesh
        tw_mesh = ThinCurr(oft_env)
        tw_mesh.setup_model(mesh_file='input_data/'+mesh_file,
                            xml_filename='input_data/oft_in.xml')
        tw_mesh.setup_io()
        __do_plot_B_J(output_full,nfp,n_phi,n_theta,bnorm,tw_mode,tw_mesh,sensor_set,
                            doSave,save_Ext,params)
        
###########################################################
def __b_norm_surf_plot(r_grid,bnorm,nfp,ntheta,nphi,doSave='',save_Ext='',params={}): 
        
    # 2D Plot of B-norm and J
    theta=np.linspace(-np.pi,np.pi,ntheta,endpoint=False)/np.pi
    phi=np.linspace(0,2*np.pi,(nphi-1) if nfp>1 else nphi,endpoint=False)/np.pi
    phi_bnorm=np.linspace(0,2*np.pi,nphi,endpoint=False)/np.pi
    plt.close('B-norm_J')
    fig, ax = plt.subplots(1,2,sharex=True,sharey=True,num='B-norm_J',\
                           figsize=(6,3),squeeze=False,tight_layout=True)
    redo_inds = np.concatenate(( np.arange(int(len(theta)/2),len(theta)),
                                np.arange(0,int(len(theta)/2)) ))
    #theta=theta[redo_inds]
    # if nfp > 1:
    #     out_1=np.r_[0.0,output[0,:-nfp-1]].reshape((nphi-1,ntheta)).transpose()
    #     out_2=np.r_[0.0,output[1,:-nfp-1]].reshape((nphi-1,ntheta)).transpose()
    # else:
    #     out_1=np.r_[0.0,output[0,:-nfp-1]].reshape((nphi,ntheta)).transpose()
    #     out_2=np.r_[0.0,output[1,:-nfp-1]].reshape((nphi,ntheta)).transpose()
    # out_1=out_1[redo_inds,:]
    # out_2=out_2[redo_inds,:]
    # ax[0,0].contour(phi,theta,out_1,50)
    # ax[0,1].contour(phi,theta,out_2,50)
    # print(phi_bnorm.shape, theta.shape,bnorm.shape,bnorm[0,:,redo_inds].transpose().shape)
    out_1=bnorm[0,:,:].transpose()
    out_2=bnorm[1,:,:].transpose()
    print(phi_bnorm.shape,theta.shape, out_1.shape)
    ax[0,0].contour(phi_bnorm,theta,out_1,50)
    _ = ax[0,1].contour(phi_bnorm,theta,out_2,50)
    
    #ax[0,0].set_ylabel(r'$\sigma(\theta)$ [$\pi$-rad]')
    ax[0,0].set_ylabel(r'$B_{\hat{n}}(\theta)$ [$\pi$-rad]')
    for i in range(2):ax[0,i].set_xlabel(r'$\phi$ [$\pi$-rad]')
    ax[0,0].set_title(r'Cosine')
    ax[0,1].set_title('Sine')

    norm = Normalize(vmin=np.min(bnorm),vmax=np.max(bnorm))
    fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap('plasma')),
        label=r'$\hat{r}\cdot\hat{n}$' ,ax=ax[0,1])
    plt.show()
    m=params['m'];n=params['n'];
    if doSave:fig.savefig(doSave+'Surface_Current_2D_%s_m-n_%d-%d_%s.pdf'%\
                    (sensor_set,m if type(m) is int else m[0],n,save_Ext))
#########################################################fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz%s_Surface_Current.npz'%(mesh_file,sensor_set, freqs[0], freqs[-1],save_ext)
def __do_plot_B_J(output,nfp,nphi,ntheta,bnorm,tw_mode,tw_mesh,sensor_set,doSave,
                  save_Ext,params):

    
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
    # actor = p.add_mesh(grid, color="white", opacity=1.0, show_edges=True,
    #            scalars=J_c,scalar_bar_args={'title':r'$\sigma_c(\theta,\phi)\, [A/m]$'})
    actor = p.add_mesh(grid, color="white", opacity=1.0, show_edges=True,
            scalars=bn_c,scalar_bar_args={'title':r'$b_{\hat{n}}(\theta,\phi)\, [T]$'})
    actor.prop.edge_opacity = 0.5
    # p.add_scalar_bar(title='J_c',)

    # Plot Conducting Structures
    # tw_mesh.build_XDMF()
    # with h5py.File('plasma/mesh.0001.h5','r') as h5_file:
    #     r = np.asarray(h5_file['R_surf'])
    #     lc = np.asarray(h5_file['LC_surf'])
    # celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    # cells = np.insert(lc, [0,], 3, axis=1)
    # grid = pyvista.UnstructuredGrid(cells, celltypes, r)
    # actor = p.add_mesh(grid, color="white", opacity=1.0, show_edges=True)
    
    # New ThinCurr way of getting mesh
    plot_data = tw_mesh.build_XDMF()
    grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()
    p.add_mesh(grid,color="white",opacity=.3,show_edges=True)


    # Plot Sensors
    sensors=gen_Sensors_Updated(select_sensor=sensor_set)
    for ind,s in enumerate(sensors):
        p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                     render_points_as_spheres=True,
                     label='Sensor' if ind==0 else None)
    if doSave:p.save_graphic(doSave+'Surface_J_s_%s_m-n_%d-%d_f_%d%s.pdf'%\
                    (sensor_set,params['m'] if type (params['m']) is int else params['m'][0],params['n'],params['f']*1e-3,save_Ext))
    
    if doSave: p.save_graphic(doSave+ 'Surface_J_s_%s_m-n_%d-%d_f_%d%s.svg'%\
                    (sensor_set,params['m'] if type (params['m']) is int else params['m'][0],params['n'],params['f']*1e-3,save_Ext))

    p.show()

#########################################################
if __name__=='__main__':
    # SPARC Side
    mesh_file='SPARC_Sept2023_noPR.h5';doSave='../output_plots/';save_ext='_FAR3D_NonLinear_Scale_30_3D_Tiles';file_geqdsk='g1051202011.1000',
    sensor_set='BP';xml_filename='oft_in.xml';
    # params={'m':2,'n':1,'r':.25,'R':1,'n_pts':40,'m_pts':60,\
    # 'f':F_KM,'dt':1e-6,'periods':3,'n_threads':8,'I':I_KM,'T':1e-3}

    # C-Mod Side
    mesh_file='C_Mod_ThinCurr_Combined-homology.h5'
    # mesh_file = 'vacuum_mesh.h5'
    # mesh_file = 'C_Mod_ThinCurr_Limiters_Combined_3D_Tiles-homology.h5'
    file_geqdsk='g1051202011.1000'
    # eta = '1.8E-5, 1.8E-5, 3.6E-5, 1.8E-5, 3.6E-5, 2.4E-5'#, 6.54545436E-5, 2.4E-5' )
    # sensor_set='Synth-C_MOD_BP_T';cmod_shot=1051202011
    sensor_set='C_MOD_ALL';cmod_shot=1051202011
    # sensor_set = 'C_MOD_ALL'


    # eta = '1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5
    #     # Bulk resistivities
    Mo = 53.4e-9 # Ohm * m at 20c
    SS = 690e-9 # Ohm * m at 20c
    w_tile_lim = 1.5e-2 *1e-20  # Tile limiter thickness
    w_tile_arm = 1.5e-2 *1 # Tile extention thickness
    w_vv = 3e-2 # Vacuum vessel thickness
    w_ss = 1e-2  # Limiter Support structure thickness
    w_shield = 0.43e-3 
    eta = f'{SS/w_ss}, {Mo/w_tile_lim}, {SS/w_ss}, {Mo/w_tile_lim}, {SS/w_vv}, {SS/w_shield}, {Mo/w_tile_arm}, {SS/w_tile_lim}' 
    
    [9,10,11,12]
    params={'m':[14,13,12,11,10,9,8],'n':11,'r':.25,'R':1,'n_pts':30,'m_pts':57,\
    'f':650e3,'dt':1e-7,'periods':2,'n_threads':28,'I':I_KM,'T':1e-5}
    doPlot=True; plotOnly=False
    C1_file=None#'/nobackup1/wenhaw42/SPARC_dir/Linear/01_n1_test_cases/1000_bate1.0_constbz_0_cp0501/C1.h5'
    FAR3D_br_file = 'br_1103';FAR3D_bth_file='bth_1103'

    doSave_Bode=True
    scan_in_freq=True
    psi_contour_select=0.4

    Synthetic_Mirnov_Surface(mesh_file=mesh_file,doSave=doSave,save_ext=save_ext,file_geqdsk=file_geqdsk,\
                             sensor_set=sensor_set,xml_filename=xml_filename,params=params,doPlot=doPlot,\
                            C1_file=C1_file,FAR3D_br_file=FAR3D_br_file,FAR3D_bth_file=FAR3D_bth_file,\
                            plotOnly=plotOnly, eta=eta, doSave_Bode=doSave_Bode, scan_in_freq=scan_in_freq,
                            psi_contour_select=psi_contour_select)
    
    print('All done')