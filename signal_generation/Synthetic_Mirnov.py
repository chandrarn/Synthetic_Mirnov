#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:43:59 2024
 Basic wrapper code for arbitrary fillament generation and mesh integration
@author: rian
"""

# header
from header_signal_generation import struct,sys,os,h5py,np,plt,mlines,rc,cm,pyvista,ThinCurr,\
    Mirnov, save_sensors, build_XDMF, mu0, histfile, subprocess, geqdsk, cv2,\
        make_smoothing_spline, factorial, json, subprocess, F_AE, I_AE, F_AE_plot, sys, gen_coupled_freq
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
from geqdsk_filament_generator import gen_filament_coords, calc_filament_coords_geqdsk
from prep_sensors import conv_sensor
sys.path.append('../signal_analysis/')
from plot_sensor_output import plot_Currents, plot_single_sensor
import matplotlib;matplotlib.use('TkAgg') # Use TkAgg backend for plotting

# main loop
def gen_synthetic_Mirnov(input_file='',mesh_file='C_Mod_ThinCurr_VV-homology.h5',
                         xml_filename='oft_in.xml',\
                             params={'m':8,'n':7,'r':.25,'R':1,'n_pts':30,'m_pts':10,\
                            'f':F_AE,'dt':1e-4,'T':1e-3,'periods':1,'n_threads':64,'I':I_AE},
                                doSave='',save_ext='',file_geqdsk='g1051202011.1000',
                                sensor_set='Synth-C_MOD_BP_T',cmod_shot=1051202011,
                                plotOnly=False ,archiveExt='',doPlot=False,
                                eta = '1.8E-5, 3.6E-5, 2.4E-5'):
    
    #os.system('rm -rf vector*') # kernal restart still required for vector numbering issue
    
    # Get mode amplitudes, assign to fillaments [eventually, from simulation]

    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(params)

    
    # Get coordinates for fillaments
    theta,phi=gen_filament_coords(params)
    
    #return theta,phi
    filament_coords = calc_filament_coords_geqdsk(file_geqdsk,theta,phi,params)
    

    # Generate sensors, filamanets
    gen_filaments(xml_filename,params,filament_coords, eta)
    sensors=gen_Sensors_Updated(select_sensor=sensor_set,cmod_shot=cmod_shot, skipBP= True, debug=True)

    
    # Get Mesh
    tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv = \
        get_mesh(mesh_file,xml_filename,params,sensor_set)



    
    # Run time dependent simulation
    if not plotOnly:coil_currs, plot_data =  run_td(sensor_obj,tw_mesh,params, coil_currs,sensor_set,
                            save_ext,mesh_file,archiveExt)
    if not doPlot: return
    # return tw_mesh,params,coil_currs,sensors,doSave,\
    #                             save_ext,Mc, L_inv,filament_coords,file_geqdsk, plot_data
    scale= makePlots(tw_mesh,params,coil_currs,sensors,doSave,
                                save_ext,Mc, L_inv,filament_coords,file_geqdsk,sensor_set)
    
    
    return tw_mesh,params,coil_currs,sensors,doSave,\
                                save_ext,Mc, L_inv,filament_coords,file_geqdsk,sensor_set, scale


    # slices,slices_spl=makePlots(tw_mesh,params,coil_currs,sensors,doSave,
    #                             save_ext,Mc, L_inv,filament_coords,file_geqdsk,sensor_set)

    # return sensors, coil_currs, tw_mesh, slices,slices_spl
#    return sensor_,currents
#####################################3
def get_mesh(mesh_file,filament_file,params,sensor_set,debug=False):
    tw_mesh = ThinCurr(nthreads=params['n_threads'],debug_level=2,)
    tw_mesh.setup_model(mesh_file='input_data/'+mesh_file,xml_filename='input_data/'+filament_file)
    print('checkpoint 0')
    tw_mesh.setup_io()
    
    print('checkpoint 1')
    # Sensor - mesh and sensor - filament inductances
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor('input_data/floops_%s.loc'%sensor_set)
    #print('floops_%s.loc'%sensor_set)
    print('checkpoint 2')
    # Filament - mesh inductance
    Mc = tw_mesh.compute_Mcoil()
    print('checkpoint 3')
    # Build inductance matrix
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file='input_data/HOLDR_L_%s_%s.save'%(mesh_file,sensor_set))
    print('checkpoint 4')
    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction
    eig_vals, eig_vecs =(0,0)# tw_mesh.get_eigs(100,False)
    L_inv = 0#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))
    print('checkpoint 5')
    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

####################################
def gen_filaments(filament_file,params,filament_coords,\
                  eta = '1.8E-5, 3.6E-5, 2.4E-5'):#, 6.54545436E-5, 2.4E-5' ):
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    n_pts=params['n_pts'];m_pts=params['m_pts']
    #theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    #eta='1E6, 1E6, 1E6, 1E6, 1E6'# Effective insulator wall
    # Coords comes in as a list of m/n pairs, each with a list of filaments
    # Each filament is a list of points, each point is a list of [x,y,z] points
    # This holds even if there's only one m/n pair
    with open('input_data/'+filament_file,'w+') as f:
        f.write('<oft>\n\t<thincurr>\n\t<eta>%s</eta>\n\t<icoils>\n'%eta)
        
       
        for filament_ in filament_coords:
            
            
            for filament in filament_:
                f.write('\t<coil_set>\n')
                f.write('\n\t\t<coil npts="%d" scale="1.0">\n'%n_pts)
                for xyz in np.array(filament).T:
                    x=xyz[0];y=xyz[1];z=xyz[2]
                    # for some starting theta, wind in phi
                    f.write('\t\t\t %1.1f, %1.1f, %1.1f\n'%(x,y,z) )
                f.write('\t\t</coil>\n')
                f.write('\t</coil_set>\n')
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
def gen_coil_currs(param,debug=True,doSave=True):
    # Assign currents to existing fillaments for time dependent calculation
    # Eventually, the balooning approximation goes here, as does straight field line apprxoiation

    # TODO: Set up to run with multiple sets of theta, phi for different m/n pairs
    # TODO: Set up to run with multiple frequency functions or single valued frequencies

    m_pts=param['m_pts'];m=param['m'];dt=param['dt'];f=param['f'];periods=param['periods']
    noise_envelope = param['noise_envelope'] # Random noise envelope
    if type(m) is int: m = [m]# Convert type to correct format

    I=param['I'];n=param['n'];n_pts=param['n_pts']
    nsteps = int(param['T']/dt)
    
    theta_,phi_= gen_filament_coords(param) # returns as lists
    time=np.linspace(0,param['T'],nsteps)
    
    # assume that m_pts can be a list
    if type(m_pts) == int: num_filaments = m_pts*len(theta_)
    else: num_filaments = sum([len(theta) for theta in theta_])
    if debug:print('Number of filaments: %d'%num_filaments)
    coil_currs = np.zeros((time.size,num_filaments+1))

    # Loop through each set of theta values for each m/n pair
    # and assign the current to each filament

    # Necessary step for dealing with uneven number of filaments
    cumulative_filament_coords = np.cumsum([len(theta__) for theta__ in theta_]) 
    cumulative_filament_coords = np.insert(cumulative_filament_coords, 0, 0) # prepend 0 to cumulative coords
    for ind_filament, theta_list in enumerate(theta_):
        # Get (potentially time dependent) mode amplitude and frequency
        if type(param['I']) == int: mode_amp = param['I']*np.ones((nsteps+1,))
        elif type(param['I']) == list: mode_amp = param['I'][ind_filament]
        else: mode_amp = param['I']

       
        if type(param['f']) == float: mode_freq = param['f']*2*np.pi*time # always just return f
        # assume frequency is a function taken [0,1] as the argument
        elif type(param['f']) == list: mode_freq = param['f'][ind_filament]
        else:mode_freq = param['f'] #  

        for ind,t in enumerate(time):
            np.random.seed(ind) # Set seed for reproducibility
            coil_currs[ind,cumulative_filament_coords[ind_filament]:cumulative_filament_coords[ind_filament+1]] = \
                np.random.normal(1, noise_envelope, len(theta_list)) * \
                    [mode_amp[ind]*np.cos(m[ind_filament]*theta+mode_freq[ind]) for theta in theta_list] 
                 
            
    # Time vector in first column
    coil_currs[:,0]=time
    
    if doSave: np.savez('../data_output/coil_currs.npz', coil_currs=coil_currs,m=m,n=n,n_pts=n_pts,m_pts=m_pts)
    return coil_currs

####################################
def run_td(sensor_obj,tw_mesh,param,coil_currs,sensor_set,save_Ext,mesh_file,
           archiveExt='',doPlot=False,doPlot_Currents=True):
    dt=param['dt'];f=param['f'];periods=param['periods'];m=param['m'];
    n=param['n']
    nsteps = int(param['T']/dt)

    

    # run time depenent simulation, save floops.hist file
    tw_mesh.run_td(dt,nsteps,
                    coil_currs=coil_currs,sensor_obj=sensor_obj,status_freq=1000,plot_freq=1000)
    if doPlot:tw_mesh.plot_td(nsteps,compute_B=True,sensor_obj=sensor_obj,plot_freq=1000)
    
    # Save B-norm surface for later plotting # This may be unnecessar
    if doPlot: _, Bc = tw_mesh.compute_Bmat(cache_file='input_data/HODLR_B_%s_%s.save'%(mesh_file,sensor_set)) 
     
    # Saves floops.hist (no, run_Td does this?)
    plot_data = tw_mesh.build_XDMF()
    hist_file = histfile('floops.hist');
    for h in hist_file:print(h)
    # Rename output 
    if type(f) is float: f_out = '%d'%f*1e-3 
    else: f_out = 'custom'
    if type(m) is not list: mn_out = '%d-%d'%(m,n)
    else: mn_out = '-'.join([str(m_) for m_ in m])+'---'+\
        '-'.join([str(n_) for n_ in n])
    f_save = '../data_output/%sfloops_filament_%s_m-n_%s_f_%s_%s%s'%\
                    (archiveExt,sensor_set,mn_out,f_out,mesh_file,save_Ext)
    subprocess.run(['cp','floops.hist',f_save+'.hist'])

    # Also save as .json file for easy reading
    data_out={}
    for key in hist_file.keys():data_out[key]=hist_file[key].tolist()
    with open(f_save+'.json','w', encoding='utf-8') as f:
        json.dump(data_out,f,ensure_ascii=False, indent=4)
    # Save coil currents
    #with open(f_save+'.json','w', encoding='utf-8') as f:
    # plot 
    plot_single_sensor(f_save+'.hist',['BP1T_ABK','BP01_ABK'],coil_currs=coil_currs,\
                       coil_inds=[1,30,55],params=params)
    print('Saved: %s'%f_save)
                    
    return coil_currs, plot_data
########################
def makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_Ext,Mc, L_inv,
              filament_coords,file_geqdsk, sensor_set,t_pt=100,plot_B_surf=False,debug=True):
    
    # MEsh and Filaments
    m=params['m'];n=params['n'];r=params['r'];R=params['R'];
    if type(m) is int: m = [m]# Convert type to correct format
    if type(n) is int: n = [n]# Convert type to correct format
    n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
    f=params['f'];dt=params['dt'];I=params['I']

    # Calculate induced current in mesh
    
    #tw_plate.save_current(np.dot(Linv,M[0]),'M')
    # tw_mesh.save_current(np.dot(L_inv,np.dot(Mc.T,coil_currs[0,1:])), 'Ind_Curr')
    # Necessary to clear old vector file: otherwise vector # keeps counting up
    
    
    with h5py.File('mesh.0001.h5','r') as h5_file:
        r_ = np.asarray(h5_file['R_surf'])
        lc = np.asarray(h5_file['LC_surf'])
    if plot_B_surf:
        with h5py.File('vector_dump.0001.h5') as h5_file:
            print('test')
            for h in h5_file.keys():print(h)
            Jfull = np.asarray(h5_file['J0001'])
            #return Jfull
            scale = (np.linalg.norm(Jfull,axis=1))
            print(scale.max())
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r_)

    pyvista.global_theme.allow_empty_mesh = True
    p = pyvista.Plotter()  
    if debug:print('Launched Plotter')
    # Plot Mesh
    if plot_B_surf: 
        p.add_mesh(grid,color="white",opacity=1,show_edges=True, \
                   scalars=Jfull,clim=[0,150],smooth_shading=True,\
                       scalar_bar_args={'title':'Eddy Current [A/m]'})
    else: p.add_mesh(grid,color="white",opacity=.9,show_edges=True)
    #slices=grid.slice_orthogonal()
    #slice_coords=[np.linspace(0,1.6,10),[0]*10,np.linspace(-1.5,1.5,10)]
    #slice_line = pyvista.Spline(np.c_[slice_coords].T,10)
    #slices = grid.slice_along_line(slice_line)
    #slices.plot()
    #p.add_mesh(slices,label='Mesh Frame')
    
    tmp=[]
    if debug:print('Plotted Mesh')
    # Plot Filaments
    colors=['red','green','blue']
    #theta_,phi_=gen_filament_coords(m,n,n_pts,m_pts)
    # Modify to accept filament_coords as a list
    cumulative_filament_coords = np.cumsum([len(filament) for filament in filament_coords])
    cumulative_filament_coords = np.insert(cumulative_filament_coords, 0, 0) # prepend 0 to cumulative coords
    for ind_mn, filament_list in enumerate(filament_coords):
        for ind,filament in enumerate(filament_list):
            pts = np.array(filament).T
            #print(np.array(calc_filament_coord(m,n,r,R,theta,phi)).shape)
            pts=np.array(pts)
            #print(pts.shape,theta)
            spl=pyvista.Spline(pts,len(pts))
            #p.add_(spline,render_lines_as_tubes=True,line_width=5,show_scalar_bar=False)
            #p.add_mesh(spl,opacity=1,line_width=6,color=plt.get_cmap('viridis')(theta*m/(2*np.pi)))
            #slices_spl=spl.slice_along_line(slice_line)#spl.slice_orthogonal()
            if coil_currs[t_pt,1+ind+cumulative_filament_coords[ind_mn]] == 0: continue
            p.add_mesh(spl,color=plt.get_cmap('plasma')((coil_currs[t_pt,1+ind+cumulative_filament_coords[ind_mn]]/np.max(coil_currs[t_pt,:]) + 1) /2),
                    line_width=10,render_points_as_spheres=True,
                    label='Filament %d/%d'%(m[ind_mn],n[ind_mn]) if ind==0 else None, opacity=1-.5*ind_mn/len(filament_coords))
            #p.add_points(pts,render_points_as_spheres=True,opaity=1,point_size=20,color=colors[ind])
            tmp.append(pts)
    if debug:print('Plotted Fillaments')
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
    #return slices, #slices_spl
    return []#scale
########################

        
    
#####################################
  
if __name__=='__main__':
   
    # params={'m':3,'n':1,'r':.3,'R':2,'n_pts':100,'m_pts':70,\
    #     'f':1e4,'dt':1e-5,'T':3e-4,'periods':3,'n_threads':64,'I':30}
    #params={'m':18,'n':16,'r':.25,'R':1,'n_pts':70,'m_pts':60,'f':500e3,'dt':1e-7,'periods':3,'n_threads':64,'I':10}
    params={'m':[1,4,8],'n':[1,3,4],'r':.3,'R':2,'n_pts':90,'m_pts':[40,60,80],\
        'f':1e4,'dt':1e-6,'T':10e-3,'periods':3,'n_threads':12,'I':30,'noise_envelope':0.05}
    
    # C-Mod Side
    mesh_file='C_Mod_ThinCurr_Combined-homology.h5'
    #mesh_file = 'C_Mod_ThinCurr_Limiters-homology.h5'
    file_geqdsk='g1051202011.1000'
    eta = '1.8E-5, 1.8E-5, 3.6E-5'#'1.8E-5, 3.6E-5, 2.4E-5'#, 6.54545436E-5, 2.4E-5' )
    # sensor_set='Synth-C_MOD_BP_T';cmod_shot=1051202011
    sensor_set='C_MOD_LIM';cmod_shot=1051202011

    # SPARC Side
    #file_geqdsk = 'geqdsk_freegsu_run0_mod_00.geq'
    #mesh_file='SPARC_Sept2023_noPR.h5'
    #mesh_file = 'SPARC_mirnov_plugwest_v2-homology.h5'
    #sensor_set='MRNV'
    #eta = '1.8E-5, 3.6E-5, 2.4E-5, 6.54545436E-5, 2.4E-5' 
    
    # Misc
    # mesh_file='thincurr_ex-torus.h5'
    #mesh_file='vacuum_mesh.h5'

    # Frequency, amplitude modulation
    # Note: If the amplitude and frequency are not set correctly for LF signals, 
    # the modulation frequency will dominate the spectrogram
    # Separately, if the noise envelope is too high, it induces some odd integration noise
    time = np.linspace(0,params['T'],int(params['T']/params['dt']))
    periods = 5
    dead_fraction = 0.4
    f_mod = lambda t: 7e3 + 3e3*t
    I_mod = lambda t: .1*4*(5 + 7*t**4)
    
    f_out_1, I_out_1, f_out_plot_1 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                      random_seed=42)

    
    periods = 2
    dead_fraction = 0.2
    f_mod = lambda t: 35e3 + 5e3*t
    I_mod = lambda t: .05*(6 + 3*np.sin(periods*2*np.pi*t))
    
    f_out_2, I_out_2, f_out_plot_2 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                        random_seed=42)

    periods = 3
    dead_fraction = 0.2
    f_mod = lambda t: 200e3 - 5e3*t**2
    I_mod = lambda t: .02*(2 + 4*t**2)
    
    f_out_3, I_out_3, f_out_plot_3 = gen_coupled_freq(time, periods, dead_fraction, f_mod, I_mod,\
                                                        random_seed=42)
    
    
    params['f'] = [f_out_1, f_out_2, f_out_3]  # List of frequencies for each m/n pair
    params['I'] = [I_out_1, I_out_2, I_out_3]  # List of amplitudes for each m/n pair

    save_ext=''
    doSave='../output_plots/'

    #coil_currs=gen_coil_currs(params,True)

    gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_set=sensor_set,params=params,
         save_ext=save_ext,doSave=doSave, eta = eta, doPlot = True, file_geqdsk = file_geqdsk, plotOnly=False)
    
    print('Run complete')
