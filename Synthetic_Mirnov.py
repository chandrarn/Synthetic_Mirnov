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
        make_smoothing_spline, factorial, json
from gen_MAGX_Coords import gen_Sensors,gen_Sensors_Updated
from geqdsk_filament_generator import gen_filament_coords, calc_filament_coords_geqdsk
from prep_sensors import conv_sensor
# main loop
def gen_synthetic_Mirnov(input_file='',mesh_file='thincurr_ex-torus.h5',
                         xml_filename='oft_in.xml',\
                             params={'m':12,'n':10,'r':.25,'R':1,'n_pts':50,'m_pts':60,\
                            'f':500e3,'dt':2e-7,'periods':3,'n_threads':64,'I':10},
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
        get_mesh(mesh_file,xml_filename,sensor_set,params)

    # Get mode amplitudes, assign to fillaments [eventually, from simulation]
    
    # Generate coil currents (for artificial mode)
    coil_currs = gen_coil_currs(params)
    
    # Run time dependent simulation
    run_td(sensor_obj,tw_mesh,params, coil_currs)
    
    
    slices,slices_spl=makePlots(tw_mesh,params,coil_currs,sensors,doSave,
                                save_ext,Mc, L_inv,filament_coords,file_geqdsk)

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
    Msensor, Msc, sensor_obj = tw_mesh.compute_Msensor('floops_%s.loc'%sensor_file)
    print('checkpoint 2')
    # Filament - mesh inductance
    Mc = tw_mesh.compute_Mcoil()
    print('checkpoint 3')
    # Build inductance matrix
    tw_mesh.compute_Lmat(use_hodlr=True,cache_file='HOLDR_L_%s.save'%sensor_set)
    print('checkpoint 4')
    # Buld resistivity matrix
    tw_mesh.compute_Rmat()
    
    # Get eigenvectors of inductance for low rank reconstruction
    eig_vals, eig_vecs =(0,0)# tw_mesh.get_eigs(100,False)
    L_inv = 0#np.linalg.pinv(np.dot(np.dot(eig_vecs.T,np.diag(eig_vals)),np.linalg.pinv(eig_vecs.T)))
    print('checkpoint 5')
    return tw_mesh, sensor_obj, Mc, eig_vals, eig_vecs, L_inv

####################################
def gen_filaments(filament_file,params,filament_coords ):
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
    time=np.arange(0,periods/f,dt)
    coil_currs = np.zeros((time.size,m_pts+1))
    for ind,t in enumerate(time):
        coil_currs[ind,1:]=[I*np.cos(m*theta+t*f*2*np.pi) for theta in theta_]
        
    # Time vector in first column
    coil_currs[:,0]=np.arange(0,periods/f,dt)
    
    return coil_currs
####################################
def run_td(sensor_obj,tw_mesh,param,coil_currs,doPlot=True):
    dt=param['dt'];f=param['f'];periods=param['periods']
    tw_mesh.run_td(dt,int(periods/f/dt),
                    coil_currs=coil_currs,sensor_obj=sensor_obj,status_freq=100,plot_freq=100)
    #tw_mesh.plot_td(int(periods/f/dt),compute_B=False,sensor_obj=sensor_obj,plot_freq=100)
    
    # Save B-norm surface for later plotting # This may be unnecessar
    #_, Bc = tw_mesh.compute_Bmat(cache_file='HODLR_B.save') 
       
    return coil_currs
########################
def makePlots(tw_mesh,params,coil_currs,sensors,doSave,save_Ext,Mc, L_inv,
              filament_coords,file_geqdsk, t_pt=0,plot_B_surf=False):
    
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
    if plot_B_surf:
        with h5py.File('vector_dump.0001.h5') as h5_file:
            Jfull = np.asarray(h5_file['J_v0001'])
            scale = 0.2/(np.linalg.norm(Jfull,axis=1)).max()
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
    cells = np.insert(lc, [0,], 3, axis=1)
    grid = pyvista.UnstructuredGrid(cells, celltypes, r_)

    p = pyvista.Plotter()  
    
    # Plot Mesh
    if plot_B_surf: p.add_mesh(grid,color="white",opacity=.9,show_edges=True,scalars=Jfull)
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
    
    plot_Currents(params, coil_currs, doSave, save_Ext,file_geqdsk=file_geqdsk)
          
    plt.show()
    return slices, slices_spl
########################

################# Currents 
def plot_Currents(params,coil_currs,doSave,save_Ext='',
                  sensor_file='MAGX_Coordinates.json',doVoltage=True,
                  manualCurrents=True,current_phi=350,file_geqdsk='geqdsk'):
   m=params['m'];n=params['n'];r=params['r'];R=params['R'];
   n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
   f=params['f'];dt=params['dt'];I=params['I']
   
   # Load sensor parameters for voltage conversion
   sensor_params= json.load(open(sensor_file,'r'))
   
   hist_file = histfile('floops.hist')
   plt.close('Currents%s'%save_Ext)
   fig,ax=plt.subplots(2,1,tight_layout=True,figsize=(4,4),
               num='Currents%s'%save_Ext,sharex=True)
   times=np.arange(0,periods/f,dt)
   currents = I*np.cos(m*0+n*current_phi+f*2*np.pi*times) if manualCurrents else coil_currs[:,1]
   current_label = r'$\phi=%d,\theta=0$'%current_phi  if manualCurrents else r'$\phi=0,\theta=0$'
   ax[0].plot(times*1e3,currents,label=current_label)
   # import decimal 
   sensors=['MIRNOV_TOR_SET_340_V5','MIRNOV_TOR_SET_340_V9','MIRNOV_TOR_SET_340_H6']
   # decimal.Decimal(-.25).as_integer_ratio()

   for ind,s in enumerate(sensors):
       label=gen_label('MIRNOV',s,sensor_params,file_geqdsk,params)
       ax[1].plot(hist_file['time'][:-1]*1e3,field_to_current(hist_file[s],\
              dt,f,sensor_params,'MIRNOV',s) if doVoltage else hist_file[s][:-1]*1e4, label=label)
   ax[0].set_ylabel("I-Mode [A]")
   ax[1].set_ylabel(r'V$_\mathrm{out}$ [V]' if doVoltage else r'B$_z$ [G]')
   ax[1].set_xlabel("Time [ms]")
   
   
   for i in range(2):
       ax[i].grid()
       ax[i].legend(fontsize=8,loc='lower right',handlelength=1.5)
   if doSave:fig.savefig(doSave+'Filament_and_Field%s.pdf'%save_Ext,
                         transparent=True)  
####################################
def field_to_current(B,dt,w_mode,sensor_params,sensor_set,sensor_name):
    # Assume: wc = 2MHz
    wc = 2e6*2*np.pi
    
    # Get sensor turns*area
    if sensor_set != 'MIRNOV':NA = sensor_params[sensor_set][sensor_name]['NA']
    else: NA = sensor_params[sensor_set][sensor_name[7:18]][sensor_name[19:]]['NA']['NA'][0]
    
    # Signal damping factor, in SI units
    factor = lambda NA, wc, w_mode: -1 * NA / (1 + (w_mode*2*np.pi)**2/wc**2)
    
    
    V = np.zeros((len(B)-1,1))
    for ind in range(len(B)-1):
        # Autodetermine derivative order based on length of availible data
        s=np.arange(-3 if ind >=3 else -ind,4 if len(B)-ind>=4 else len(B)-ind)
        V[ind]=np.dot(B[s+ind],__finDiff(s, 1))/dt # Calculate dB/dt
    
    return V * factor(NA,wc,w_mode)

def __finDiff(s,d): # Finite difference stencil
    # Build with automatic s generator: input: order, derivative
    if not len(s)>d:raise SyntaxError("Insufficient Points for Derivative")
    S_mat=np.zeros((len(s),len(s)))
    for i in range(len(s)):
        S_mat[i]=s**i
    d_vec=np.zeros((len(s),1))
    d_vec[d]=factorial(d)

    return np.matmul(np.linalg.inv(S_mat),d_vec)
#####################################
def gen_label(sensor_set,sensor_name,sensor_params,file_geqdsk,params):
    if file_geqdsk is None:zmagx=0;rmagx=params['R']
    else:
        with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
        zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
    if sensor_set != 'MIRNOV':
        Z = sensor_params[sensor_set][sensor_name]['Z']
        R = sensor_params[sensor_set][sensor_name]['R']
        PHI = sensor_params[sensor_set][sensor_name]['PHI']
    else: 
        Z = sensor_params[sensor_set][sensor_name[7:18]][sensor_name[19:]]['Z']
        R = sensor_params[sensor_set][sensor_name[7:18]][sensor_name[19:]]['R']
        PHI = sensor_params[sensor_set][sensor_name[7:18]][sensor_name[19:]]['PHI']
        
    theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
    
    # Generate name label
    if sensor_set == 'MIRNOV': 
        return r'%s %s: $\theta=%1.1f^\circ,\,\phi=%1.1f^\circ$'%\
            (sensor_set,sensor_name[19:],theta,PHI)
        
    
#####################################
  
if __name__=='__main__':
    mesh_file='SPARC_Sept2023_noPR.h5'
    # mesh_file='thincurr_ex-torus.h5'
    sensor_set='MIRNOV'
    gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_set=sensor_set)
