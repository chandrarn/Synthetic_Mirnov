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

from header_signal_analysis import histfile, np, json, geqdsk, factorial

def get_signal_data(params,filament,save_Ext,phi_sensor,sensor_file,
                    sensor_set,file_geqdsk,doVoltage):
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
    return X,Y,Z, sensor_dict

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