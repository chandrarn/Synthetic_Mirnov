#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from header_synthetic_mirnov_generation import np, plt, pyvista, ThinCurr,\
    Mirnov, save_sensors, OFT_env, mu0, histfile, geqdsk

################################################################################################
################################################################################################
def get_mesh(mesh_file,filament_file,params,sensor_set,debug=True):
    # Load mesh, compute inductance matrices   
    if debug: print('Loading mesh from %s'%mesh_file)
    oft_env = OFT_env(nthreads=params['n_threads'],abort_callback=True)
    tw_mesh = ThinCurr(oft_env)
    tw_mesh.setup_model(mesh_file='input_data/'+mesh_file,xml_filename='input_data/'+filament_file)
    tw_mesh.setup_io()


    # Sensor - mesh and sensor - filament inductances
    if debug: print(('Computing mutual inductances between mesh and %s sensors'%sensor_set))
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor('input_data/floops_%s.loc'%sensor_set)

    # Filament - mesh inductance
    if debug: print('Computing mutual inductances between mesh and filaments')
    Mc = tw_mesh.compute_Mcoil()

    # Build inductance matrix
    if debug: print('Computing mesh self-inductance')
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file='input_data/HOLDR_L_%s_%s.save'%(mesh_file,sensor_set))

    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction [Not used presently]
    eig_vals, eig_vecs = None,None#tw_mesh.get_eigs(10,False)
    L_inv = None#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))

    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

################################################################################################
################################################################################################
def run_frequency_scan(tw_mesh,params,sensor_set,mesh_file,sensor_obj,doSave_Bode,save_ext):

    coil_current_magnitude = params['I'] # Current magnitude for the coil
    freqs = params['f'] # Frequency range for the scan

    Mcoil = tw_mesh.compute_Mcoil()
    driver = np.zeros((2,tw_mesh.nelems))
    driver[0,:] = Mcoil[0,:]*coil_current_magnitude

    # Mutual between the mesh and sensors, and coil and sensors
    Msensor, Msc, _ = tw_mesh.compute_Msensor('input_data/floops_%s.loc'%sensor_set)

    # Data vector storage
    sensors_bode = []
    # Loop through frequencies to test
    for freq in freqs if np.ndim(freqs) > 0 else [0, freqs]:
        result = tw_mesh.compute_freq_response(fdriver=driver,freq=freq)

        # contribution from the mesh current to the sensor, with the mesh current at a given frequency
        probe_signals = np.dot(result,Msensor)
        # Contribuio from the coil current directly to the sensor
        probe_signals[0,:] += np.dot(np.r_[coil_current_magnitude],Msc)

        # for i in range(probe_signals.shape[1]):
        #     print('Real: {0:13.5E}, Imaginary: {1:13.5E}'.format(*probe_signals[:,i]))
        probe_signals = probe_signals[0,:] + 1j*probe_signals[1,:] # Combine real and imaginary parts   
        sensors_bode.append(probe_signals)
    
    # Only compute the mesh induced currents once
    tw_mesh.save_current(result[0,:],'Jr_coil')
    tw_mesh.save_current(result[1,:],'Ji_coil')
    _ = tw_mesh.build_XDMF()

    # Save the probe signals to a file
    if doSave_Bode: 
        fName = 'Frequency_Scan_on_%s_using_%s_from_%2.2e-%2.2eHz%s.npz'%(mesh_file,sensor_set, freqs[0], freqs[-1],save_ext)
        np.savez('../data_output/'+fName, probe_signals=sensors_bode, freqs=freqs,\
                 sensor_names=sensor_obj['names']) 
    
        # Plot the probe signals
        print('Saved frequency scan data to %s'%fName)
    return np.array(sensors_bode)


################################################################################################
################################################################################################
def makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_Ext,Mc, L_inv,
              filament_coords,file_geqdsk, sensor_set,t_pt=0,plot_B_surf=True,debug=True,
              clim_J=None,scan_in_freq=False):
    # Generate plots of mesh, filaments, sensors, and currents
    # Will plot induced current on the mesh if plot_B_surf is True

    if debug:print('Generating output plots')
    # Mesh and Filaments
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    if type(m) is int: m = [m]# Convert type to correct format
    if type(n) is int: n = [n]# Convert type to correct format
    n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
    f=params['f'];dt=params['dt'];I=params['I']
    

    # New ThinCurr way of getting mesh
    plot_data = tw_mesh.build_XDMF()
    grid = plot_data['ThinCurr']['smesh'].get_pyvista_grid()
    if plot_B_surf: 
        if scan_in_freq:
            Jfull = plot_data['ThinCurr']['smesh'].get_field('Jr_coil')
        else:
            Jfull = plot_data['ThinCurr']['smesh'].get_field('J_v',timestep=0)
    if debug:print('Built Pyvista grid from ThinCurr mesh')


    pyvista.global_theme.allow_empty_mesh = True
    p = pyvista.Plotter()  
    if debug:print('Launched Plotter')

    # Plot Mesh
    if plot_B_surf: 
        p.add_mesh(grid,color="white",opacity=.6,show_edges=True, \
                   scalars=Jfull,clim=clim_J,smooth_shading=True,\
                       scalar_bar_args={'title':'Eddy Current [A/m]'})
    else: p.add_mesh(grid,color="white",opacity=.9,show_edges=True)
    
    tmp=[]
    if debug:print('Plotted Mesh')


    ###############################################
    # Plot Filaments
    colors=['red','green','blue']
    # Modify to accept filament_coords as a list
    cumulative_filament_coords = np.cumsum([len(filament) for filament in filament_coords])
    cumulative_filament_coords = np.insert(cumulative_filament_coords, 0, 0) # prepend 0 to cumulative coords
    for ind_mn, filament_list in enumerate(filament_coords):
       
        for ind,filament in enumerate(filament_list):

            pts = np.array(filament).T
            p.add_points(pts[0],render_points_as_spheres=True,opacity=1,point_size=20,\
                         color='k',label='Launch Point' if ind == 0 else None)
            
            spl=pyvista.Spline(pts,len(pts))

            p.add_mesh(spl,color=plt.get_cmap('plasma')((coil_currs[t_pt,1+ind+cumulative_filament_coords[ind_mn]]/np.max(coil_currs[t_pt,:]) + 1) /2),
                    line_width=10,render_points_as_spheres=True,
                    label='Filament %d/%d'%(m[ind_mn],n[ind_mn]) if ind==0 else None, opacity=1-.5*ind_mn/len(filament_coords))
           
            tmp.append(pts)
    if debug:print('Plotted Fillaments')
    
    ###################################################
    # Plot Sensors
    for ind,s in enumerate(sensors):
        p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                     render_points_as_spheres=True, 
                     label='Mirnov' if ind==0 else None)
    p.add_legend()
    if debug:print('Plotted Sensors')
    if doSave:p.save_graphic(doSave+'Mesh_and_Filaments%s.pdf'%save_Ext)
    if debug:print('Saved figure')
    p.show()
    if debug:print('Plotted Figure')
    # plot_Currents(params, coil_currs, doSave, save_Ext,file_geqdsk=file_geqdsk,
    #               sensor_set=sensor_set)
          
    plt.show()
    return []
########################