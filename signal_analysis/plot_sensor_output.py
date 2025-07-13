#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:43:15 2025
 Suite of codes for plotting currents from ThinCurr sensor output
@author: rian
"""
from header_signal_analysis import json,plt,np,histfile,geqdsk,factorial, Normalize,cm, cv2,sys
sys.path.append('../signal_generation/')
from header_signal_generation import F_AE_plot

from get_signal_data import get_signal_data, __select_sensors

# Surface plot for arb sensors
def plot_Current_Surface(params,coil_currs=None,sensor_file='../signal_generation/input_data/MAGX_Coordinates_CFS.json',
                         sensor_set='MRNV',doVoltage=True,phi_sensor=[160],
                         doSave='',save_Ext='',timeScale=1e6,file_geqdsk='geqdsk',
                         filament=True,scale_unit={},plotExt='',mesh_file='',saveDataFile=False):
    
    # # Load sensor parameters for voltage conversion
    # sensor_params= json.load(open(sensor_file,'r'))
    # # Load ThinCurr sensor output
    # hist_file = histfile('data_output/floops_%s_%s_m-n_%d-%d_f_%d%s.hist'%\
    #          ('filament' if filament else 'surface', sensor_set,params['m'],
    #           params['n'],params['f']*1e-3,save_Ext))
    # '''
    # print('data_output/floops_%s_m-n_%d-%d_f_%d%s.hist'%\
    #              (sensor_set,params['m'],params['n'],params['f']*1e-3,save_Ext))
    # return hist_file
    # '''    
    # # Select usable sensors from set
    # sensor_dict = __select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params)
    # #return sensor_dict, sensor_params
    # # build datasets
    # X,Y,Z = __gen_surface_data(sensor_dict,hist_file,doVoltage,params,
    #                            sensor_set, sensor_params)
    
    X,Y,Z, sensor_dict  = get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                        sensor_set,file_geqdsk,doVoltage,mesh_file)
    
    #return sensor_dict,X,Y,Z
    # build plot
    doPlot(sensor_set,save_Ext,sensor_dict,X,Y,Z,timeScale,doSave,params,
           doVoltage,filament,scale_unit,plotExt=plotExt)
    
    if saveDataFile:save_output(X,Y,Z,sensor_dict,saveDataFile,filament,sensor_set,\
                    params,mesh_file,plotExt)
    return sensor_dict,X,Y,Z
##########################
def doPlot(sensor_set,save_Ext,sensor_dict,X,Y,Z,timeScale,doSave,params,
           doVoltage,filament,scale_unit,cLims=[],shotno=None,plotExt=''):
    plt.close('%s_Current_Surface_%s%s'%(sensor_set,
                         'filament' if filament else 'surface',plotExt))
    fig,ax=plt.subplots(len(sensor_dict),1,tight_layout=True,sharex=True,
                        num='%s_Current_Surface_%s%s'%(sensor_set,
                           'filament' if filament else 'surface',plotExt),squeeze=False)
    if ax.ndim==2:ax=ax[:,0]
    if cLims and np.size(cLims[0])==1: cLims = [cLims]*len(sensor_dict)
    scale = scale_unit['scale'] if scale_unit else 1
    unit = scale_unit['unit'] if scale_unit else 'arb'
    
    for ind, s in enumerate(sensor_dict):
        Z[ind] *= scale
        if not filament:
            trimTime=20
            X[ind] = X[ind][trimTime:]
            Z[ind] = Z[ind][:,trimTime:]
        norm = Normalize(np.min(Z[ind]),np.max(Z[ind])) if not cLims else \
            Normalize(*cLims[ind])
        ax[ind].contourf(X[ind]*timeScale,Y[ind],Z[ind],levels=50,
                         cmap='plasma',zorder=-5,
             vmin=cLims[ind][0] if cLims else None,vmax=cLims[ind][1] if cLims else None)
        ax[ind].set_rasterization_zorder(-1)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='plasma'),ax=ax[ind],
                     label= r'V$_\mathrm{out}$ [V]' if doVoltage else \
                         r'B$_\theta$ [%s]'%unit)
        ax[ind].set_ylabel(s[0]['y_label'])
        ax[ind].tick_params(top=True)
    ax[-1].set_xlabel(r'Time [%s]'%('ms' if timeScale==1e3 else '$\mu$s'))
    plt.show()
    if doSave: 
        if not shotno:
            fName=doSave+'Sensor_surface_%s_%s_%d-%d_%dkHz_%s%s.pdf'%\
            ('filament' if filament else 'surface',sensor_set,params['m'],
             params['n'],params['f']*1e-3,'V' ,plotExt)
        else: fName = doSave+'Sensor_Surface_%d_%s_t%2.2f_%2.2f%s%s.pdf'%\
            (shotno,sensor_set,X[0][0],X[0][-1],'_V'*doVoltage,plotExt)
        fig.savefig(fName,transparent=True)
        print('Saved: %s'%fName)

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
                  sensor_file='input_data/MAGX_Coordinates_CFS.json',doVoltage=True,
                  manualCurrents=True,current_phi=350,file_geqdsk='geqdsk',
                  timeScale=1e6,sensor_set='MIRNOV',archiveExt=''):
   m=params['m'];n=params['n'];r=params['r'];R=params['R'];
   n_pts=params['n_pts'];m_pts=params['m_pts'];periods=params['periods']
   f=params['f'];dt=params['dt'];I=params['I']
   
   # Load sensor parameters for voltage conversion


   sensor_params= json.load(open(sensor_file,'r'))
   
   f_out = f*1e-3 if type(f) is float else F_AE_plot(0)[0]*1e-3
   hist_file = histfile('../data_output/%sfloops_filament_%s_m-n_%d-%d_f_%d%s.hist'%\
                   (archiveExt,sensor_set,m,n,f_out,save_Ext) )
       
   plt.close('Currents%s'%save_Ext)
   fig,ax=plt.subplots(2,1,tight_layout=True,figsize=(4,4),
               num='Currents%s'%save_Ext,sharex=True)
   times=np.arange(0,periods/f,dt)
   currents = I*np.cos(m*0+n*current_phi+f*2*np.pi*times) if manualCurrents else coil_currs[:,1]
   current_label = r'$\phi=%d^\circ,\,\theta=0^\circ$'%current_phi  if manualCurrents else r'$\phi=0,\theta=0$'
   ax[0].plot(times*timeScale,currents,label=current_label)
   # import decimal 
   #sensors=['MIRNOV_TOR_SET_340_V5','MIRNOV_TOR_SET_340_V9','MIRNOV_TOR_SET_340_H6']
   sensors=['BP1T_ABK']
   #sensors=['BP-LOMN-008M']
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

# Simplified single sensor plotting
def plot_single_sensor(hist_file_name,sensor_name,coil_currs=None,coil_inds=None,params=None,doSave=True):
    hist_file = histfile(hist_file_name)
    # Sanitize input
    if type(sensor_name) is str: sensor_name = [sensor_name]
    if coil_inds is not None:
        if type(coil_inds) is int: coil_inds = [coil_inds]
        m = params['m'] if type(params['m']) is list else [params['m']]
        n = params['n'] if type(params['n']) is list else [params['n']]

    plt.close('Single_Sensor_%s'%sensor_name)
    fig,ax=plt.subplots(1+(1 if np.any(coil_currs) else 0),1,sharex=True,
            tight_layout=True,num='Single_Sensor_%s'%sensor_name,squeeze=False)
    for name in sensor_name:
        ax[0,0].plot(hist_file['time'][:]*1e3,hist_file[name][:]*1e4,label=name,\
                     alpha=1-.5*sensor_name.index(name)/len(sensor_name))
    
    ax[0,0].set_ylabel(r'B [G]')
    ax[0,0].legend(fontsize=8,loc='upper right',handlelength=1.5)
    ax[0,0].grid()
    if np.any(coil_currs):
        for i in range(0,len(coil_inds)):
            ax[1,0].plot(coil_currs[:,0]*1e3,coil_currs[:,coil_inds[i]],label='Filament %d/%d'%(m[i],n[i]),\
                         alpha=1-.5*i/len(coil_inds))
        ax[1,0].set_ylabel('Filament Current [A]')
        #ax[1,0].set_xlabel('Time [s]')
        ax[1,0].grid()
        ax[1,0].legend(fontsize=8)
    ax[-1,0].set_xlabel('Time [ms]')
    if doSave:
        save_name = hist_file_name.split('/')[-1].split('.')[0]
        fig.savefig('../output_plots/'+save_name+'_Single_Sensor.pdf',transparent=True)
    plt.show()

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
###########################################
def save_output(X,Y,Z,sensor_dict,output_file,filament,sensor_set,\
                params,mesh_file,plotExt):
    
    fName='_Data_%s_%s_%d-%d_%dkHz_%s_%s%s.pdf'%\
    ('filament' if filament else 'surface',sensor_set,params['m'],
     params['n'],params['f']*1e-3,'V' ,mesh_file,plotExt)
    
    # save sensor dict
    with open(output_file+'Info_'+fName+'.json','w') as f: json.dump(sensor_dict,f)
    np.savez(output_file+'Data_'+fName+'.npz', X=np.array(X,dtype=object),\
             Y=np.array(Y,dtype=object),Z=np.array(Z,dtype=object))
    