#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:40:49 2025
     Module for loading in sensor output from ThinCurr run 
     Standard input: parameters to build .hist file name, sensor angle specification
     Standard output: arrays of time (1D), space (1D), signal (2D), for toroidal
         poloidal sensor sets within a given sensor type, from .hist output file
         Also return sensor names (includes labeling information)
         
         Individual sensors within set are selected based on naming convention
         layed out in 
@author: rianc
"""

from header_signal_analysis import histfile, np, json, geqdsk, factorial, sys,\
    CModEFITTree

sys.path.append('../C-Mod/')
from get_Cmod_Data import BP, __loadData

def get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                    sensor_set,file_geqdsk,doVoltage,mesh_file):
    # Load sensor parameters for voltage conversion
    sensor_params= json.load(open(sensor_file,'r'))
    
    # Load ThinCurr sensor output
    hist_file = histfile('../data_output/floops_%s_%s_m-n_%d-%d_f_%d_%s%s.hist'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,mesh_file,save_Ext))
    
    
    print('Loaded: ../data_output/floops_%s_%s_m-n_%d-%d_f_%d_%s%s.hist'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,mesh_file,save_Ext))
    '''
    return hist_file
    '''    
    # Select usable sensors from set
    sensor_dict = __select_sensors(sensor_set,sensor_params,phi_sensor,
                                   file_geqdsk,params)
    #return sensor_dict, sensor_params, hist_file
    # build datasets
    X,Y,Z = __gen_surface_data(sensor_dict,hist_file,doVoltage,params,
                               sensor_set, sensor_params)
    return X,Y,Z, sensor_dict

def __gen_surface_data(sensor_dict,hist_file,doVoltage,params,sensor_set,
                       sensor_params):
    # Build 2D data
    dt=params['dt'];f=params['f']
    X = [hist_file['time'][:]]*len(sensor_dict)
    Y=[]
    Z=[]
    for s in sensor_dict:
        Y.append( [s[i]['y_vals'] for i in range(len(s))] )
        inds=np.argsort(Y[-1])
        Y[-1] = np.array(Y[-1])[inds]
        #Z.append([])
        Z_tmp= np.array([\
       (field_to_current(hist_file[s_['Sensor']],\
              dt,f,sensor_params,sensor_set,s_['Sensor']) if doVoltage else \
                hist_file[s_['Sensor']][:])  for s_ in s]).squeeze()
        # length check 
        if Z_tmp.shape[1]>X[0].shape[0]: Z_tmp = Z_tmp[:,:-1]
        elif Z_tmp.shape[1]<X[0].shape[0]: 
            for i in range(len(X)): X[i] = X[i][:-1]
        #print('\n\n\n ', Z_tmp[inds].squeeze().shape,'\n\n')
        Z.append(Z_tmp[inds].squeeze()) # Unclear if this should be in a list or not
    #return Z
    
    return X,Y,Z
def __select_sensors(sensor_set,sensor_params,phi_sensor,file_geqdsk,params,
                     shotno=None,tLim=None):
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
    
    ###########################################################################
    ###########################################################################
    
    elif sensor_set == 'Synth-C_MOD_BP':
        tLim
        if file_geqdsk is None:zmagx=0;rmagx=params['R']
        else:
            with open(file_geqdsk,'r') as f: eqdsk=geqdsk.read(f)
            zmagx=eqdsk.zmagx;rmagx=eqdsk.rmagx
            
        bp = BP(0)
        R = bp.BC['R']
        Z = bp.BC['Z']
        PHI = bp.BC['PHI']
        theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
        names = bp.BC['NAMES']
        
        
        # Poloidal side
        sensor_dict.append([])
        for ind,s in enumerate(names):
            sensor_dict[-1].append({'Sensor':'%s'%(s),'y_vals':theta[ind],\
               'y_label':r'%s$_{\phi=%d}\,\hat{\theta}$ [deg]'%(sensor_set,PHI[ind]),
               })
        
        # Toroidal
        sensor_dict.append([])
        tor_ind = 14
        sensor_dict[-1].append({'Sensor':'%s'%(bp.BC['NAMES'][tor_ind]),'y_vals':bp.BC['PHI'][0],\
           'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,0)})
        sensor_dict[-1].append({'Sensor':'%s'%(bp.DE['NAMES'][tor_ind]),'y_vals':bp.DE['PHI'][0],\
           'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,0)})
        sensor_dict[-1].append({'Sensor':'%s'%(bp.GH['NAMES'][tor_ind]),'y_vals':bp.GH['PHI'][0],\
           'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,0)})
        sensor_dict[-1].append({'Sensor':'%s'%(bp.JK['NAMES'][tor_ind]),'y_vals':bp.JK['PHI'][0],\
           'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,0)})
    
    ###########################################################################
    ###########################################################################
    
    elif sensor_set == 'C_MOD_MIRNOV_T':#'Synth-C_MOD_BP_T':
        sensor_dict.append([]);sensor_dict.append([])
        for i in np.arange(1,4):
            for ind,set_ in enumerate(['AB','GH']):
                name = 'BP%dT_%sK'%(i,set_)
                sensor_dict[ind].append({'Sensor':name ,'y_vals': sensor_params[name],\
                       'y_label':r'Synth C-Mod Mirnov $\hat{\phi}$ [deg]'})
        
    ###########################################################################
    ###########################################################################
    elif sensor_set == 'C_Mod_BP':
        eq = CModEFITTree(shotno)
        time = eq.getTimeBase()
        tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] ) 
        if np.size(tInd)==0:tInd = np.argmin((time-tLim[0])**2) 
        zmagx = np.mean(eq.getMagZ()[tInd])
        rmagx = np.mean(eq.getMagR()[tInd])

        
        sensor = BP(shotno) #Redo time for real data
        time=sensor.time
        tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] ) 
        
        # Poloidal side
        sensor_sets =[sensor.BC,sensor.DE,sensor.GH,sensor.JK]
        phi = np.array([s['PHI'][0] for s in sensor_sets])
        ind = np.argmin((phi-phi_sensor)**2)
        sensor=sensor_sets[ind]
        
        R = sensor['R']
        Z = sensor['Z']
        PHI = sensor['PHI']
        theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
        names = sensor['NAMES']
        
        # Poloidal side
        sensor_dict.append([])
        for ind,s in enumerate(names):
            sensor_dict[-1].append({'Sensor':'%s'%(s),'y_vals':theta[ind],\
               'y_label':r'%s$_{\phi=%d}\,\hat{\theta}$ [deg]'%(sensor_set,PHI[ind]),
               'Signal':sensor['SIGNAL'][ind,tInd]*(1 if 0<=theta[ind]<=180 else -1),\
                   'Time':time[tInd]})
        
        # Toroidal
        sensor_dict.append([])
        tor_ind = 14
        for ind_,s in enumerate(sensor_sets):
            if ind_==ind:continue # No need to double count sensor from poloidal array
            
            R = s['R']
            Z = s['Z']
            theta = np.arctan2(Z- zmagx, R-rmagx)*180/np.pi
            tor_ind = np.argmin(np.abs(theta))
            
            sensor_dict[-1].append({'Sensor':'%s'%(s['NAMES'][tor_ind]),\
                'y_vals':s['PHI'][0],\
               'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,0),\
                   'Signal':s['SIGNAL'][tor_ind,tInd],'Time':time[tInd]})
    ###########################################################################
    elif sensor_set =='C_Mod_BP_T':
        # Assume using the upper row always
        try:eq = CModEFITTree(shotno)
        except:eq = CModEFITTree(1160930034)
        time = eq.getTimeBase()
        tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] ) 
        # if tLim points are closer than dt-EFIT, it won't work
        if np.size(tInd)==0:tInd = np.argmin((time-tLim[0])**2) 
        zmagx = np.mean(eq.getMagZ()[tInd])
        rmagx = np.mean(eq.getMagR()[tInd])
        
        
        sensor = __loadData(shotno,pullData=['bp_t'])['bp_t']
        theta = np.arctan2(sensor.ab_z-zmagx,sensor.ab_r-rmagx)[0]*180/np.pi
        
        time=sensor.time
        tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] )
        dt =np.mean(np.diff(time))
        # Only one side for now
        sensor_dict.append([])
        sensor_dict.append([])
        
        for i in np.arange(0,3):
            sensor_dict[-2].append({'Sensor':'%s'%(sensor.ab_names[i]),\
                'y_vals':sensor.ab_phi[i]+360,\
               'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,theta),\
                   'Signal':np.cumsum(sensor.ab_data[i-1])[tInd]*dt,'Time':time[tInd]})
                
            sensor_dict[-1].append({'Sensor':'%s'%(sensor.gh_names[i]),\
                    'y_vals':sensor.gh_phi[i]+360,\
                   'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,theta),\
                       'Signal':np.cumsum(sensor.gh_data[i-1])[tInd]*dt,'Time':time[tInd]})
                
        '''
        sensor_dict[-1].append({'Sensor':'%s'%('Blank'),\
                'y_vals':np.mean([sensor.ab_phi[0],sensor.gh_phi[0]])+360,\
               'y_label':r'%s$_{\theta=%d}\,\hat{\phi}$ [deg]'%(sensor_set,theta),\
                   'Signal':sensor.gh_data[i-1,tInd]*np.nan,'Time':time[tInd]})
        '''
    else: raise SyntaxWarning('Diagnostic Not Implimented')
    return sensor_dict
###############################################################################
####################################
def field_to_current(B,dt,w_mode,sensor_params,sensor_set,sensor_name):
    
    
    # Get sensor turns*area
    if sensor_set != 'MIRNOV':
        NA = sensor_params[sensor_set][sensor_name]['NA']
        wc = 23.1875 * 1e3 * 2*np.pi # fitted critical frequency from emperical sensor tests
    else: 
        NA = sensor_params[sensor_set][sensor_name[7:18]][sensor_name[19:]]['NA']['NA'][0]
        wc = 2e6*2*np.pi
        
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