#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:43:15 2025
 Suite of codes for plotting currents from ThinCurr sensor output
@author: rian
"""
from header import json,plt,np,histfile,geqdsk,factorial, Normalize,cm, cv2

# Surface plot for arb sensors
def plot_Current_Surface(params,coil_currs=None,sensor_file='MAGX_Coordinates_CFS.json',
                         sensor_set='MRNV',doVoltage=True,phi_sensor=[340],
                         doSave='',save_Ext='',timeScale=1e3,file_geqdsk='geqdsk',
                         filament=True):
    
    # Load sensor parameters for voltage conversion
    sensor_params= json.load(open(sensor_file,'r'))
    # Load ThinCurr sensor output
    hist_file = histfile('data_output/floops_%s_%s_m-n_%d-%d_f_%d%s.hist'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,save_Ext))
    '''
    print('data_output/floops_%s_m-n_%d-%d_f_%d%s.hist'%\
                 (sensor_set,params['m'],params['n'],params['f']*1e-3,save_Ext))
    return hist_file
    '''    
    # Select usable sensors from set
    sensor_dict = __select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params)
    #return sensor_dict, sensor_params
    # build datasets
    X,Y,Z = __gen_surface_data(sensor_dict,hist_file,doVoltage,params,
                               sensor_set, sensor_params)
    # return sensor_dict,X,Y,Z
    # build plot
    doPlot(sensor_set,save_Ext,sensor_dict,X,Y,Z,timeScale,doSave,params,
           doVoltage,filament)
    return sensor_dict,X,Y,Z
##########################
def doPlot(sensor_set,save_Ext,sensor_dict,X,Y,Z,timeScale,doSave,params,
           doVoltage,filament,cLims=[]):
    plt.close('%s_Current_Surface_%s%s'%(sensor_set,
                         'filament' if filament else 'surface',save_Ext))
    fig,ax=plt.subplots(len(sensor_dict),1,tight_layout=True,sharex=True,
                        num='%s_Current_Surface_%s%s'%(sensor_set,
                           'filament' if filament else 'surface',save_Ext))
    for ind, s in enumerate(sensor_dict):
        if not filament:
            trimTime=10
            X[ind] = X[ind][trimTime:]
            Z[ind] = Z[ind][:,trimTime:]
        norm = Normalize(np.min(Z[ind]),np.max(Z[ind]))
        ax[ind].contourf(X[ind]*timeScale,Y[ind],Z[ind],levels=50,
                         norm=norm,cmap='plasma',zorder=-5,
             vmin=cLims[ind][0] if cLims else None,vmax=cLims[ind][1] if cLims else None)
        ax[ind].set_rasterization_zorder(-1)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax[ind],
                     label= r'V$_\mathrm{out}$ [V]' if doVoltage else \
                         r'B$_\mathrm{z}$ [G]')
        ax[ind].set_ylabel(s[0]['y_label'])
        ax[ind].tick_params(top=True)
    ax[-1].set_xlabel(r'Time [%s]'%('ms' if timeScale==1e3 else '$\mu$s'))
    plt.show()
    if doSave: 
        fName=doSave+'Sensor_surface_%s_%s_%d-%d_%dkHz_%s%s.pdf'%\
            ('filament' if filament else 'surface',sensor_set,params['m'],
             params['n'],params['f']*1e-3,'V' ,save_Ext)
        fig.savefig(fName)
        print('Saved: %s'%fName)
def __gen_surface_data(sensor_dict,hist_file,doVoltage,params,sensor_set,
                       sensor_params):
    # Build 2D data
    dt=params['dt'];f=params['f']
    X = [hist_file['time'][:-1]]*len(sensor_dict)
    Y=[]
    Z=[]
    for s in sensor_dict:
        Y.append( [s[i]['y_vals'] for i in range(len(s))] )
        inds=np.argsort(Y[-1])
        Y[-1] = np.array(Y[-1])[inds]
        Z.append([])
        Z[-1].append(np.array([\
       (field_to_current(hist_file[s_['Sensor']],\
              dt,f,sensor_params,sensor_set,s_['Sensor']) if doVoltage else \
                hist_file[s_['Sensor']][:-1]*1e4)  for s_ in s]).squeeze())
        Z[-1]=np.array(Z[-1]).squeeze()[inds]
    #return Z
    
    return X,Y,Z
def __select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params):
    # Return list of dictionaries for subplots
    # returns: full sensor name, yaxis, y label, 
    sensor_dict=[]
    if sensor_set == 'MIRNOV':
        
        subset = sensor_params[sensor_set]['TOR_SET_%03d'%phi_sensor[0]]
        sensor_dict.append([]); sensor_dict.append([])
        #sensor
        for s in subset:
            if file_geqdsk is None:zmagx=0;rmagx=params['R']
            else:
                with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
                zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
            R = subset[s]['R']
            Z = subset[s]['Z']
            PHI = subset[s]['PHI']
            theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
            
            if s[0] == 'V':
                sensor_dict[0].append({'Sensor':'%s_%s_%s'%\
                       (sensor_set,'TOR_SET_%03d'%phi_sensor[0],s), 
                       'y_vals':theta,'y_label':r'Mirnov-V $\theta$ [deg]'})
            if s[0] == 'H':
                sensor_dict[1].append({'Sensor':'%s_%s_%s'%\
                       (sensor_set,'TOR_SET_%03d'%phi_sensor[0],s), 
                       'y_vals':PHI,'y_label':r'Mirnov-H $\phi$ [deg]'})
                    
    elif sensor_set == 'MRNV':
        subset = sensor_params[sensor_set]
        sensor_dict.append([]);sensor_dict.append([])
        for s in subset:
            # Gen coords
            if file_geqdsk is None:zmagx=0;rmagx=params['R']
            else:
                with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
                zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
            #print(s,subset[s])
            R = subset[s]['R']
            Z = subset[s]['Z']
            PHI = subset[s]['PHI']
            theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
            
            #print(s,PHI,not (phi_sensor[0] - 20 <= PHI <= phi_sensor[0] + 20))
            # separate sets
            if not (phi_sensor[0] - 20 <= PHI <= phi_sensor[0] + 20): continue
            
            if s[-2] == 'V':
                sensor_dict[0].append({'Sensor':'%s'%s, 
                       'y_vals':theta,'y_label':r'Mirnov-V $\theta$ [deg]'})
            if s[-2] == 'H':
                sensor_dict[1].append({'Sensor':'%s'%s, 
                       'y_vals':PHI,'y_label':r'Mirnov-H $\phi$ [deg]'})
    elif sensor_set == 'BP' or sensor_set == 'BN' or sensor_set == 'MRNV':
        subset = sensor_params[sensor_set]
        sensor_dict.append([]);sensor_dict.append([])
        for s in subset:
            # Gen coords
            if file_geqdsk is None:zmagx=0;rmagx=params['R']
            else:
                with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
                zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
            #print(s,subset[s])
            R = subset[s]['R']
            Z = subset[s]['Z']
            PHI = subset[s]['PHI']
            theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
            
            # poloidal side
            if phi_sensor[0] - 20 <= PHI <= phi_sensor[0] + 20:
                sensor_dict[0].append({'Sensor':'%s'%(s),'y_vals':theta,\
                   'y_label':r'%s$_{\phi=%d}\,\hat{\theta}$ [deg]'%(sensor_set,PHI),
                   })
            if 0-10 <= theta <= 10: # Toroidal ''set''
                sensor_dict[1].append({'Sensor':'%s'%(s),'y_vals':PHI,\
                   'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,theta)})
    return sensor_dict
#######################################################
def __sensor_position_verify(sensor_set,sensor_file='MAGX_Coordinates_CFS.json',
                             phi_sensor=[0],file_geqdsk='geqdsk',params=None,
                             highlight=[],doSave=''):
    sensor_params= json.load(open(sensor_file,'r'))
    sensor_dict=__select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params)
    
    R_lcfs, Z_lcfs = __LCFS_Compute(file_geqdsk)
    if sensor_set == 'BP' or sensor_set == 'BN':
        plt.close('Sensor_ID')
        fig,ax=plt.subplots(1,1,num='Sensor_ID',tight_layout=True,figsize=(4,6))
        with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
        zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
        ax.plot(rmagx,zmagx,'x',ms=10)
        for s in sensor_dict[0]:
            X=sensor_params[sensor_set][s['Sensor']]['X']
            Z=sensor_params[sensor_set][s['Sensor']]['Z']
            
            ax.plot(X,Z,'*k')
            if s['Sensor'] in highlight:
                ax.plot([rmagx,X],[zmagx,Z],'--',
                        label=r'$\theta=%1.1f^\circ$'%s['y_vals'])
        ax.plot(R_lcfs,Z_lcfs,'k-',alpha=.6)
        plt.grid()
        if np.any(highlight):ax.legend(fontsize=8,handlelength=1)
        ax.set_xlabel('R [m]');ax.set_ylabel('Z [m]')
        plt.show()
        if doSave: fig.savefig(doSave+'Sensor_positions.pdf',transparent=True)
    return sensor_dict

########################################
def __LCFS_Compute(file_geqdsk='geqdsk',trim=[1,1.5]):
    with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
    # get q(psi(r,z))
    psi_eqdsk = eqdsk.psi
    R_eq=np.linspace(eqdsk.rleft,eqdsk.rleft+eqdsk.rdim,len(psi_eqdsk))
    Z_eq=np.linspace(eqdsk.zmid-eqdsk.zdim/2,eqdsk.zmid+eqdsk.zdim/2,len(psi_eqdsk))
    psi_lin = np.linspace(eqdsk.simagx,eqdsk.sibdry,eqdsk.nx)
    #p = np.polyfit(psi_lin, eqdsk.qpsi,12)
    #fn_q = lambda psi: np.polyval(p,psi)
    #q_rz = fn_q(psi_eqdsk) # q(r,z)
    
    # Contour detection
    contour,hierarchy=cv2.findContours(np.array(psi_eqdsk>=eqdsk.sibdry,dtype=np.uint8),
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
    
    r_lcfs, z_lcfs = R_eq[contour[:,1]],Z_eq[contour[:,0]]
    if np.any(trim):
        ind = np.argwhere(r_lcfs>=trim[0]).squeeze()
        r_lcfs=r_lcfs[ind];z_lcfs=z_lcfs[ind]
        ind = np.argwhere((-trim[1]<=z_lcfs) & (z_lcfs<=trim[1])).squeeze()
        r_lcfs=r_lcfs[ind];z_lcfs=z_lcfs[ind]
        break_ind =  np.argwhere(np.diff(r_lcfs)<-.3).squeeze()+1
        r_lcfs=np.insert(r_lcfs,break_ind,np.nan)
        z_lcfs=np.insert(z_lcfs,break_ind,np.nan)
    plt.figure();plt.plot(r_lcfs,z_lcfs);plt.show()
    return r_lcfs,z_lcfs 
######################################################
################# Currents 1D Plot
def plot_Currents(params,coil_currs,doSave=False,save_Ext='',
                  sensor_file='MAGX_Coordinates_CFS.json',doVoltage=True,
                  manualCurrents=True,current_phi=350,file_geqdsk='geqdsk',
                  timeScale=1e6,sensor_set='MIRNOV'):
   m=params['m'];n=params['n'];r=params['r'];R=params['R'];
   n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
   f=params['f'];dt=params['dt'];I=params['I']
   
   # Load sensor parameters for voltage conversion
   sensor_params= json.load(open(sensor_file,'r'))
   
   hist_file = histfile('data_output/floops_%s_m-n_%d-%d_f_%d%s.hist'%\
                (sensor_set,params['m'],params['n'],params['f']*1e-3,save_Ext) )
   plt.close('Currents%s'%save_Ext)
   fig,ax=plt.subplots(2,1,tight_layout=True,figsize=(4,4),
               num='Currents%s'%save_Ext,sharex=True)
   times=np.arange(0,periods/f,dt)
   currents = I*np.cos(m*0+n*current_phi+f*2*np.pi*times) if manualCurrents else coil_currs[:,1]
   current_label = r'$\phi=%d^\circ,\,\theta=0^\circ$'%current_phi  if manualCurrents else r'$\phi=0,\theta=0$'
   ax[0].plot(times*timeScale,currents,label=current_label)
   # import decimal 
   sensors=['MIRNOV_TOR_SET_340_V5','MIRNOV_TOR_SET_340_V9','MIRNOV_TOR_SET_340_H6']
   sensors=['BP-LOMN-008M']
   # decimal.Decimal(-.25).as_integer_ratio()

   for ind,s in enumerate(sensors):
       label=gen_label(sensor_set,s,sensor_params,file_geqdsk,params)
       ax[1].plot(hist_file['time'][:-1]*timeScale,field_to_current(hist_file[s],\
              dt,f,sensor_params,sensor_set,s) if doVoltage else hist_file[s][:-1]*1e4, label=label)
   ax[0].set_ylabel("I-Mode [A]")
   ax[1].set_ylabel(r'V$_\mathrm{out}$ [V]' if doVoltage else r'B$_z$ [G]')
   ax[1].set_xlabel(r"Time [%s]"%('ms' if timeScale==1e3 else r'$\mu s$'))
   
   
   for i in range(2):
       ax[i].grid()
       ax[i].legend(fontsize=8,loc='lower right',handlelength=1.5)
   if doSave:
       fName = doSave+'Filament_and_Field_%s_%d-%d_%dkHz_%s%s.pdf'%\
           (sensor_set,m,n,f*1e-3,'V' if doVoltage else 'Bz' ,save_Ext)
       fig.savefig(fName,transparent=True) 
       print('Saved: %s'%fName)
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
        deriv_operator,stencil=__finDiff(B,ind,3, 1)
        V[ind]=np.dot(B[stencil+ind],deriv_operator)/dt # Calculate dB/dt
    
    return V * factor(NA,wc,w_mode)

def __finDiff(signal,ind,order,deriv): # Finite difference stencil
    # Autodetermine derivative order based on length of availible data
    s=np.arange(-order if ind >=order else -ind,
                (order+1) if len(signal)-ind>=(order+1) else len(signal)-ind)

    # Build with automatic s generator: input: order, derivative
    if not len(s)>deriv:raise SyntaxError("Insufficient Points for Derivative")
    S_mat=np.zeros((len(s),len(s)))
    for i in range(len(s)):
        S_mat[i]=s**i
    d_vec=np.zeros((len(s),1))
    d_vec[deriv]=factorial(deriv)

    return np.matmul(np.linalg.inv(S_mat),d_vec),s
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
