#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:43:59 2024
 Wrapper code to generate synthetic Mirnov data from an arbitrary mesh, arbitrary filament path,
 with arbitrary current evolution, and aribtrary sensors
@author: rian
"""

# header
from header_signal_generation import struct,sys,os,h5py,np,plt,mlines,rc,cm,pyvista,ThinCurr,\
    Mirnov, save_sensors, OFT_env, mu0, histfile, subprocess, geqdsk, cv2,\
        make_smoothing_spline, factorial, json, subprocess, F_AE, I_AE, F_AE_plot, sys,\
              gen_coupled_freq, debug_mode_frequency_plot, getcwd, j_id, working_directory
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
from geqdsk_filament_generator import gen_filament_coords, calc_filament_coords_geqdsk, calc_filament_coords_field_lines
from prep_sensors import conv_sensor
sys.path.append('../signal_analysis/')
from plot_sensor_output import plot_Currents, plot_single_sensor

jobId = os.environ.get('SLURM_JOB_ID','')

# main loop
def gen_synthetic_Mirnov(input_file='',mesh_file='C_Mod_ThinCurr_VV-homology.h5',
                         xml_filename='oft_in.xml',\
                             params={'m':8,'n':7,'r':.25,'R':1,'n_pts':30,'m_pts':10,\
                            'f':F_AE,'dt':1e-4,'T':1e-3,'periods':1,'n_threads':64,'I':I_AE},
                                doSave='',save_ext='',file_geqdsk='g1051202011.1000',
                                sensor_set='Synth-C_MOD_BP_T',cmod_shot=1051202011,
                                plotOnly=False ,archiveExt='',doPlot=False,
                                eta = '1.8E-5, 3.6E-5, 2.4E-5',wind_in='phi', debug=False,
                                scan_in_freq=False,clim_J=None,doSave_Bode=False,oft_env=None):
    
    # os.chdir('../signal_generation/')
    # Get mode amplitudes, indexed to filament positions
    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(params,wind_in=wind_in,scan_in_freq=scan_in_freq)

   
    # Get starting coorindates for fillaments
    theta,phi=gen_filament_coords(params,wind_in=wind_in)

    if wind_in == 'phi':
        filament_coords = calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params)
    else: 
        filament_coords = calc_filament_coords_field_lines(params,file_geqdsk,doDebug=debug)

    # Put filamanets in OFT file format
    gen_OFT_filement_and_eta_file(xml_filename,params,filament_coords, eta)
    
    # Generate sensors in OFT format
    sensors=gen_Sensors_Updated(select_sensor=sensor_set,cmod_shot=cmod_shot, skipBP= True, debug=debug)

    
    # Get finite element Mesh
    tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv = \
        get_mesh(mesh_file,xml_filename,params,sensor_set,\
                  oft_env=oft_env,debug=debug)



    
    # Run time dependent simulation
    if not plotOnly:
        if scan_in_freq:
            sensors_bode = run_frequency_scan(tw_mesh,params,sensor_set,mesh_file,sensor_obj,\
                                              doSave_Bode=doSave_Bode,save_ext=save_ext)
            #make_plots_bode(sensors_bode, sensors,params,doSave,save_ext)
        else:
            coil_currs =  run_td(sensor_obj,tw_mesh,params, coil_currs,sensor_set,
                            save_ext,mesh_file,archiveExt,doPlot=doPlot,doPlot_Currents=doPlot,)
        
    if not doPlot: return coil_currs,filament_coords,sensors
    
    # Plot mesh, filaments, sensors, and currents
    scale= makePlots(tw_mesh,params,coil_currs,sensors,doSave,
                                save_ext,Mc, L_inv,filament_coords,file_geqdsk,sensor_set,
                                plot_B_surf= not plotOnly,debug=debug,scan_in_freq=scan_in_freq,
                                clim_J=clim_J,)
    
    
    return tw_mesh,params,coil_currs,sensors,doSave,\
            save_ext,Mc, L_inv,filament_coords,file_geqdsk,sensor_set, scale, oft_env

################################################################################################
################################################################################################
def get_mesh(mesh_file,filament_file,params,sensor_set,oft_env=None,debug=True):
    # Load mesh, compute inductance matrices   
    if debug: print('Loading mesh from %s'%mesh_file)
    if oft_env is None: oft_env = OFT_env(nthreads=params['n_threads'],abort_callback=True)
    tw_mesh = ThinCurr(oft_env)
    tw_mesh.setup_model(mesh_file=f'{j_id}input_data/'+mesh_file,xml_filename=f'{j_id}input_data/'+filament_file)
    tw_mesh.setup_io(basepath=f'{j_id}input_data/')


    # Sensor - mesh and sensor - filament inductances
    if debug: print(('Computing mutual inductances between mesh and %s sensors'%sensor_set))
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor(f'{j_id}input_data/floops_%s.loc'%sensor_set)

    # Filament - mesh inductance
    if debug: print('Computing mutual inductances between mesh and filaments')
    Mc = tw_mesh.compute_Mcoil()

    # Build inductance matrix
    if debug: print('Computing mesh self-inductance')
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file=f'{j_id}input_data/HOLDR_L_{mesh_file}_{sensor_set}.save')

    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction [Not used presently]
    eig_vals, eig_vecs = None,None#tw_mesh.get_eigs(10,False)
    L_inv = None#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))

    if debug: print('Mesh setup complete')
    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

################################################################################################
def gen_OFT_filement_and_eta_file(filament_file,params,filament_coords,\
                  eta = '1.8E-5, 3.6E-5, 2.4E-5'):
    # Write filament (x,y,z) coordinates to xml file in OFT format 

    # Coords comes in as a list of m/n pairs, each with a list of filaments
    # Each filament is a list of points, each point is a list of [x,y,z] points
    # This holds even if there's only one m/n pair

    m=params['m'];n=params['n'];r=params['r'];R=params['R']
    n_pts=params['n_pts'];m_pts=params['m_pts']

    with open(f'{j_id}input_data/'+filament_file,'w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n'%eta)
        
        print('File open for writing: %s'%(f'{j_id}input_data/'+filament_file))

        for filament_ in filament_coords:
            
            for filament in filament_:
                f.write('\t<coil_set>\n')
                f.write('\n\t\t<coil npts="%d" scale="1.0">\n'%np.shape(filament)[1])

                for xyz in np.array(filament).T:
                    x=xyz[0];y=xyz[1];z=xyz[2]
                    f.write('\t\t\t %1.3f, %1.3f, %1.3f\n'%(x,y,z) )

                f.write('\t\t</coil>\n')
                f.write('\t</coil_set>\n')

        f.write('\t</icoils>\n\t</thincurr>\n</oft>')

   
################################################################################################
################################################################################################
def gen_coil_currs(param,debug=True,doSave=False,wind_in='theta',scan_in_freq=False):
    # Assign currents to existing fillaments for time dependent calculation
    # The ordering of the filaments and currents must match for the field topology to be correct

    m_pts=param['m_pts'];m=param['m'];dt=param['dt'];f=param['f'];periods=param['periods']
    noise_envelope = param['noise_envelope'] # Random noise envelope
    if type(m) is int: m = [m]# Convert type to correct format

    I=param['I'];n=param['n'];n_pts=param['n_pts']
    nsteps = int(param['T']/dt)
    
    # Need to get the list of starting angles, to correctly assign filament colors
    starting_angle,_= gen_filament_coords(param,wind_in=wind_in) # returns as lists
    time=np.linspace(0,param['T'],nsteps)
    
    # assume that m_pts can be a list
    pts = m_pts if wind_in == 'phi' else n_pts
    if type(pts) == int: num_filaments = pts*len(starting_angle)
    else: num_filaments = sum([len(theta) for theta in starting_angle])
    if debug:print('Number of filaments: %d'%num_filaments)
    coil_currs = np.zeros((time.size,num_filaments+1))

    # Time vector in first column
    coil_currs[:,0]=time
    if debug:print('Initialized coil currents array with shape (%d,%d)'%coil_currs.shape)


    if not scan_in_freq:
        # Loop through each set of starting angle values for each m/n pair
        # and assign the current to each filament

        # Necessary step for dealing with uneven number of filaments for different m/n pairs
        cumulative_filament_coords = np.cumsum([len(starting_angle_) for starting_angle_ in starting_angle]) 
        cumulative_filament_coords = np.insert(cumulative_filament_coords, 0, 0) # prepend 0 to cumulative coords
        for ind_filament, starting_angle_list in enumerate(starting_angle):
            if debug: print('Assigning currents to filaments for m/n = %d/%d'%(m[ind_filament],n[ind_filament]))

            # Get (potentially time dependent) mode amplitude 
            if type(param['I']) == int: mode_amp = param['I']*np.ones((nsteps+1,))
            elif type(param['I']) == list: mode_amp = param['I'][ind_filament]
            else: mode_amp = param['I']

            # Potentially time dependent frequency
            if type(param['f']) == float: mode_freq = param['f']*2*np.pi*time # always just return f
            # assume frequency is a function taken [0,1] as the argument
            elif type(param['f']) == list: mode_freq = param['f'][ind_filament]
            else:mode_freq = param['f'] #  
            
            # set initial mode phase based on starting angle position for a given filament
            initial_mode_phase = m[ind_filament] if wind_in == 'phi' else n[ind_filament]

            for ind,t in enumerate(time):
                np.random.seed(ind) # Set seed for reproducibility
                coil_currs[ind,cumulative_filament_coords[ind_filament]+1:cumulative_filament_coords[ind_filament+1]+1] = \
                    np.random.normal(1, noise_envelope, len(starting_angle_list)) * \
                        [mode_amp[ind]*np.cos(initial_mode_phase*initial_angle+mode_freq[ind]) for initial_angle in starting_angle_list] 
                    
                

    # Save coil currents to .npy for manual inspection later
    if doSave: np.savez('../data_output/coil_currs.npz', coil_currs=coil_currs,m=m,n=n,n_pts=n_pts,m_pts=m_pts)

    return coil_currs

################################################################################################
################################################################################################
def run_td(sensor_obj,tw_mesh,param,coil_currs,sensor_set,save_Ext,mesh_file,
           archiveExt='',doPlot=True,doPlot_Currents=True,debug=True):
    # Run time dependent simulation

    dt=param['dt'];f=param['f'];periods=param['periods'];m=param['m'];
    n=param['n']
    nsteps = int(param['T']/dt)


    # run time depenent simulation, save floops.hist file for output sensor measurements
    # Note: plot_freq creates output files for the nth timestep: needs to be less than nsteps for 
    # plotting at minimum the J_surf plot :

    tw_mesh.run_td(dt,nsteps,
                    coil_currs=coil_currs,sensor_obj=sensor_obj,status_freq=100,plot_freq=10000)
    os.system(f"mv floops.hist {j_id}input_data/floops.hist")

    if doPlot:tw_mesh.plot_td(nsteps,compute_B=True,sensor_obj=sensor_obj,plot_freq=100)
    
    # Save B-norm surface for later plotting # This may be unnecessar
    if doPlot: _, Bc = tw_mesh.compute_Bmat(cache_file=f'input_data/HODLR_B_{mesh_file}_{sensor_set}.save') 
     
    # Saves floops.hist 
    hist_file = histfile(f'{j_id}input_data/floops.hist') # floops.hist created after run_td(), but it's loaded back in here


    if debug:
        print('Ran time dependent simulation for the following sensors:')
        for h in hist_file:print(h)

    f_save = __do_save_output(f,m,n,archiveExt,sensor_set,mesh_file,save_Ext, debug, hist_file)
   
    # Test plot for a few single sensors
    if doPlot_Currents:plot_single_sensor(f_save+'.hist',['BP_EF_TOP', 'BP_EF_BOT'],coil_currs=coil_currs,\
                       coil_inds=[1],params=param)
    print('Saved: %s'%f_save)
                    
    return coil_currs
################################################################################################
def __do_save_output(f,m,n,archiveExt,sensor_set,mesh_file,save_Ext, debug, hist_file):
        # Rename output 
    if type(f) is float: f_out = '%d'%(f*1e-3) 
    else: f_out = 'custom'
    if type(m) is not list: mn_out = '%d-%d'%(m,n)
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])
    f_save = working_directory+'../data_output/%sfloops_filament_%s_m-n_%s_f_%s_%s%s'%\
                    (archiveExt,sensor_set,mn_out,f_out,mesh_file,save_Ext)

    subprocess.run(['cp',f'input_data/floops.hist',f_save+'.hist'])
    if debug: print('Saved output to %s.hist'%f_save)


    # Also save as output as .json file for easy reading
    data_out={}
    for key in hist_file.keys():data_out[key]=hist_file[key].tolist()
    with open(f_save+'.json','w', encoding='utf-8') as f:
        json.dump(data_out,f,ensure_ascii=False, indent=4)

    return f_save
################################################################################################
################################################################################################
def run_frequency_scan(tw_mesh,params,sensor_set,mesh_file,sensor_obj,doSave_Bode,save_ext):

    coil_current_magnitude = params['I'] # Current magnitude for the coil
    freqs = params['f'] # Frequency range for the scan

    # Compute mutual inductance between coil filaments and mesh
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
        # Contribuion from the coil current directly to the sensor
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
              filament_coords,file_geqdsk, sensor_set,t_pt=1,plot_B_surf=True,debug=True,
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
            Jfull = plot_data['ThinCurr']['smesh'].get_field('J_v',timestep=1)
    if debug:print('Built Pyvista grid from ThinCurr mesh')


    pyvista.global_theme.allow_empty_mesh = True
    p = pyvista.Plotter()  
    if debug:print('Launched Plotter')

    # Plot Mesh
    if plot_B_surf: 
        p.add_mesh(grid,color="white",opacity=.1,show_edges=True, \
                   scalars=Jfull,clim=clim_J,smooth_shading=True,\
                       scalar_bar_args={'title':'Eddy Current [A/m]'})
    else: p.add_mesh(grid,color="white",opacity=.1,show_edges=True)
    
    tmp=[]
    if debug:print('Plotted Mesh')


    ###############################################
    # Plot Filaments
    colors=['red','green','blue']
    # Modify to accept filament_coords as a list
    cumulative_filament_coords = np.cumsum([len(filament) for filament in filament_coords])
    cumulative_filament_coords = np.insert(cumulative_filament_coords, 0, 0) # prepend 0 to cumulative coords
    while not np.any(coil_currs[t_pt,:]):t_pt+=1 # Check for first non-zero timepoint

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

################################################################################################
################################################################################################
################################################################################################
  
if __name__=='__main__':
   
    # params={'m':3,'n':1,'r':.3,'R':2,'n_pts':100,'m_pts':70,\
    #     'f':1e4,'dt':1e-5,'T':3e-4,'periods':3,'n_threads':64,'I':30}
    #params={'m':18,'n':16,'r':.25,'R':1,'n_pts':70,'m_pts':60,'f':500e3,'dt':1e-7,'periods':3,'n_threads':64,'I':10}
    # params={'m':[4],'n':[2],'r':.3,'R':2,'n_pts':[100],'m_pts':[60],\
    #     'f':1e4,'dt':1e-6,'T':1e-3,'periods':2,'n_threads':12,'I':30,'noise_envelope':0.00}
    
    # C-Mod Side
    # mesh_file='C_Mod_ThinCurr_Combined-homology.h5'
    # mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    # file_geqdsk='g1051202011.1000'

     #'1.8E-5, 1.8E-5, 3.6E-5'#'1.8E-5, 3.6E-5, 2.4E-5'#, 6.54545436E-5, 2.4E-5' ) #3.5
    # Bulk resistivities
    Mo = 53.4e-9 # Ohm * m at 20c
    SS = 690e-9 # Ohm * m at 20c
    w_tile_lim = 1.5e-2  # Tile limiter thickness
    w_tile_arm = 1.5e-2 *1 # Tile extention thickness
    w_vv = 3e-2 # Vacuum vessel thickness
    w_ss = 1e-2  # Support structure thickness
    w_shield = 0.43e-3 
    # Surface resistivity: eta/thickness
    # Assume that limiter support structures are 0.6-1.5cm SS, tiles are 1.5cm thick Mo, VV is 3cm thick SS 
    # For more accuracy, could break up filaments into different eta values based on position
    #
    eta = f'{SS/w_ss}, {Mo/w_tile_lim}, {SS/w_ss}, {Mo/w_tile_lim}, {SS/w_vv}, {SS/w_ss}, {Mo/w_tile_arm}, {SS/w_shield}' 
    # eta = f'{SS/2e-2}, {SS/15e-2}'
    # eta = '1E-6'
    # sensor_set='Synth-C_MOD_BP_T';cmod_shot=1051202011
    # sensor_set='C_MOD_LIM';cmod_shot=1051202011
    # sensor_set = 'C_MOD_ALL'
    # wind_in = 'theta'
    
    # C-Mod Frequency Scan
    # mesh_file = 'C_Mod_ThinCurr_Limiters_Combined-homology.h5'
    mesh_file='C_Mod_ThinCurr_Combined-homology.h5'
    # mesh_file='C_Mod_ThinCurr_VV-homology.h5'#
    # mesh_file='C_Mod_ThinCurr_VV_Improved-homology.h5'
    
    # mesh_file = 'vacuum_mesh.h5'
    # params={'m':[12],'n':[10],'r':0,'R':0.8,'n_pts':[20],'m_pts':[20],\
    #     'f':np.linspace(1e3,1e3,1),'dt':1.0e-6,'T':2e-2,'periods':1,'n_threads':12,'I':1,'noise_envelope':0.00}
    params={'m':[12],'n':[10],'r':0,'R':0.8,'n_pts':[60],'m_pts':[61],\
        'f':700e3,'dt':5.0e-5,'T':1e-3,'periods':2,'n_threads':12,'I':1,'noise_envelope':0.00}
    sensor_set = 'C_MOD_ALL'
    file_geqdsk='g1160930034.1200'#'g1051202011.1000' # Not used for frequency scan
    cmod_shot = 1160714026#1160930034#1151208900 	
    wind_in = 'theta' # Note: advanced `theta' winding does not work for single filament m/n=1 case
    scan_in_freq = False # Set to True to run frequency scan, False to run time dependent simulation
    clim_J = [0,.5] # Color limits for eddy current plot
    doSave_Bode = True

    # SPARC Side
    #file_geqdsk = 'geqdsk_freegsu_run0_mod_00.geq'
    #mesh_file='SPARC_Sept2023_noPR.h5'
    #mesh_file = 'SPARC_mirnov_plugwest_v2-homology.h5'
    #sensor_set='MRNV'
    #eta = '1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5' 
    # {'dt':1e-6,'T':10e-3,'periods':3}
    # Misc
    # mesh_file='thincurr_ex-torus.h5'True
    #mesh_file='vacuum_mesh.h5'

    save_ext='_f-sweep_All-Release_Comparison_FAR3d_Filaments'
    doSave='../output_plots/'*True
    plotOnly = True

    # # Frequency, amplitude modulation
    # # Note: If the amplitude and frequency are not set correctly for LF signals, 
    # # the modulation frequency will dominate the spectrogram
    # # Separately, if the noise envelope is too high, it induces some odd integration noise
    
    time = np.linspace(0,params['T'],int(params['T']/params['dt']))

    # Frequency sweep
    # periods = 1
    # dead_fraction = 0.0
    # f_mod = lambda t: 100e3 + 2e3*t
    # I_mod = lambda t: params['I']*np.ones_like(t)
    # f_out_1, I_out_1, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,I_chirp_smooth_percent=0.001)
    # # params['f'] = f_out_1
    # # params['I'] = I_out_1
    # print('Generated frequency sweep from %1.1f kHz to %1.1f kHz'%(f_out_1[0]*1e-3,f_out_1[-1]*1e-3))

    # periods = 5
    # dead_fraction = 0.4
    # f_mod = lambda t: 7e3 + 3e3*t
    # I_mod = lambda t: .1*4*(5 + 7*t**4)
    
    # f_out_1, I_out_1, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
    #                                                   random_seed=42)

    
    # periods = 2
    # dead_fraction = 0.2
    # f_mod = lambda t: 35e3 + 5e3*t
    # I_mod = lambda t: .05*(6 + 3*np.sin(periods*2*np.pi*t))
    
    # f_out_2, I_out_2, f_out_plot_2 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
    #                                                     random_seed=42)

    # periods = 3
    # dead_fraction = 0.2
    # f_mod = lambda t: 200e3 - 5e3*t**2
    # I_mod = lambda t: .02*(2 + 4*t**2)
    
    # f_out_3, I_out_3, f_out_plot_3 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
    #                                                     random_seed=42)
    

    # debug_mode_frequency_plot(time,f_out_1,I_out_1,f_out_plot_1,f_out_2,I_out_2,f_out_plot_2,f_out_3,I_out_3,f_out_plot_3,save_ext)

    # params['f'] = [f_out_1, f_out_2, f_out_3]  # List of frequencies for each m/n pair
    # params['I'] = [I_out_1, I_out_2, I_out_3]  # List of amplitudes for each m/n pair



    #coil_currs=gen_coil_currs(params,True)
    
    

    gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_set=sensor_set,params=params,wind_in=wind_in,
         save_ext=save_ext,doSave=doSave, eta = eta, doPlot = True, file_geqdsk = file_geqdsk,
           plotOnly=plotOnly, scan_in_freq= scan_in_freq, clim_J=clim_J, doSave_Bode=doSave_Bode)
    
    print('Run complete')
