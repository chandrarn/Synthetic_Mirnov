#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:47:17 2025
    Pull sensor coordinates from Granetz tree
@author: rian
"""

from header_signal_generation import Mirnov, save_sensors, flux_loop, plt,\
    mds, np, json, Mirnov_Geometry_C_Mod, os

from get_Cmod_Data import BP

def gen_Node_Dict():
    nodes = {}
    tree = mds.Tree('MAGX',-1,'readonly')
    # BN
    for node_name in ['BN','BP','FLUX_PARTIAL']:
        nodes[node_name]={}
        for phi in [10,110,190,290]:
            name_phi='TOR_SET_%03d'%phi
            nodes[node_name][name_phi]={}
            pos_list = ['DT23','DT5B','DT5C','DVT1','DVT4']
            if node_name != 'FLUX_PARTIAL':pos_list.extend(['OLIM'])
            if node_name != 'BN':pos_list.extend(['IOL1','IOL2','OOLM','VSCM'])
            if node_name =='FLUX_PARTIAL':pos_list.extend(['DVT6','IMID','OMPS'])
            for pos in pos_list:
                for ul in ['_L','_U']:
                    nodes[node_name]['TOR_SET_%03d'%phi]['%s%s'%(pos,ul)]={}
                    sig_list = ['R1','R2','Z1','Z2','PHI1','PHI2','POLARITY'] \
                        if node_name == 'FLUX_PARTIAL' else ['ANGLE','NA','R','Z','PHI','POLARITY']
                    for sig in sig_list:
                        tag = 'SIGNALS.%s.%s.%s%s.%s'%(node_name,name_phi,pos,ul,sig)
                        try:nodes[node_name][name_phi]['%s%s'%(pos,ul)][sig] = \
                            float(tree.getNode(tag).getFloatArray())
                        except:
                            print('Failure at: %s'%tag)
                            raise SyntaxError
            if node_name=='FLUX_PARTIAL':
                for pos in ['IMID','OMID']:
                    for ul in ['_M']:
                        nodes[node_name][name_phi]['%s%s'%(pos,ul)]={}
                        for sig in ['R1','R2','Z1','Z2','PHI1','PHI2','POLARITY']:
                            tag = 'SIGNALS.%s.%s.%s%s.%s'%(node_name,name_phi,pos,ul,sig)
                            try:nodes[node_name][name_phi]['%s%s'%(pos,ul)][sig] = \
                                float(tree.getNode(tag).getFloatArray())
                            except:
                                print('Failure at: %s'%tag)
                                raise SyntaxError
                    
    # Flux_Full
    for node_name in ['FLUX_FULL']:
        nodes[node_name]={}
        for pos in ['IDIV','ODIV','OMID','OOLM']:
            for ul in ['_L','_U']:
                nodes[node_name]['%s%s'%(pos,ul)]={}
                for sig in ['R','Z','POLARITY']:
                    tag = 'SIGNALS.%s.%s%s.%s'%(node_name,pos,ul,sig)
                    try:nodes[node_name]['%s%s'%(pos,ul)][sig] = \
                        float(tree.getNode(tag).getFloatArray())
                    except:
                        print('Failure at: %s'%tag)
                        raise SyntaxError
    
    # Mirnov
    for node_name in ['MIRNOV']:
        nodes[node_name]={}
        for phi in [160,340]:
            name_phi='TOR_SET_%03d'%phi
            nodes[node_name][name_phi]={}
            for hv in ['H','V']:
                for i in np.arange(1,7 if hv=='H' else 10):
                    nodes[node_name][name_phi]['%s%d'%(hv,i)]={}
                    for sig in ['GAIN','R','Z','PHI','POLARITY']:
                        tag = 'SIGNALS.%s.%s.%s%d.%s'%(node_name,name_phi,hv,i,sig)
                        try:nodes[node_name][name_phi]['%s%d'%(hv,i)][sig] = \
                            float(tree.getNode(tag).getFloatArray())
                        except:
                            print('Failure at: %s'%tag)
                            raise SyntaxError
                    sig='NA'
                    nodes[node_name][name_phi]['%s%d'%(hv,i)][sig]={}
                    tag = 'SIGNALS.%s.%s.%s%d.%s'%(node_name,name_phi,hv,i,sig)
                    tmp=tree.getNode(tag)
                    nodes[node_name][name_phi]['%s%d'%(hv,i)][sig]['NA']=\
                        np.array(tmp.data()).squeeze().tolist()
                    nodes[node_name][name_phi]['%s%d'%(hv,i)][sig]['NA_freq']=\
                        np.array(tmp.dim_of().data()).squeeze().tolist()
                    
            
    with open('input_data/MAGX_Coordinates.json','w') as f:json.dump(nodes,f)
        
#######################################
def gen_Sensors(coord_file='input_data/MAGX_Coordinates.json'):
    coords = json.load(open(coord_file,'r'))
    
    # Assuming "angle" is references from horizontal, and represents normal vector
    # Mirnov's assumed to have z-hat normal vec
    # Port plug not included yet
    # Save sensors separately, and together
    sensors_BP=[];sensors_BN=[];sensors_Flux_Partial=[];sensors_Flux_Full=[];sensors_Mirnov=[]
    sensors_all=[]
    coords_BP=[];coords_BN=[]
    
    # B normal, poloidal

    for set_ in ['BP','BN']:
        for phi in [10,110,190,290]:
            name_phi='TOR_SET_%03d'%phi
            pos_list = ['DT23','DT5B','DT5C','DVT1','DVT4']
            if set_ == 'BP':pos_list.extend(['IOL1','IOL2','OOLM','VSCM'])
            for pos_ in pos_list:
                for ul in ['_L','_U']:
                    sens = Mirnov(*__coords_xyz_BP_BN(coords[set_][name_phi]['%s%s'%(pos_,ul)],\
                  coords[set_][name_phi]['%s%s'%(pos_,'_L' if ul == '_U' else '_U')]),
                                   '%s_%s_%s%s'%(set_,name_phi,pos_,ul))
                    if set_ == 'BP': 
                        sensors_BP.append(sens);
                        #coords_BP.append(__coords_xyz_BP_BN(coords[set_][name_phi]['%s%s'%(pos_,ul)]))
                    if set_ == 'BN': 
                        sensors_BN.append(sens)
                        #coords_BN.append(__coords_xyz_BP_BN(coords[set_][name_phi]['%s%s'%(pos_,ul)]))
     
    # Flux partial
    for set_ in ['FLUX_PARTIAL']:
        for phi in [10,110,190,290]:
            name_phi='TOR_SET_%03d'%phi
            pos_list = ['DT23','DT5B','DT5C','DVT1','DVT4',\
                        'IOL1','IOL2','OOLM','VSCM',
                        'DVT6','IMID','OMPS','IMID','OMID']
            for pos in pos_list:
                for ul in ['_M'] if pos in ['IMID','OMID'] else ['_L','_U']:
                    sens = flux_loop(__coords_xyz_Flux_P(coords[set_][name_phi]['%s%s'%(pos,ul)]),
                     '%s_%s_%s%s'%(set_,name_phi,pos,ul) )
                    sensors_Flux_Partial.append(sens)
    
    # Flux Full (these loops are for integrating Bz, for EFIT reconstruction)
    for set_ in ['FLUX_FULL']:
        for pos in ['IDIV','ODIV','OMID','OOLM']:
            for ul in ['_L','_U']:
                # Format is different for these: centered on 0,0,Z, radius is R
                sens = Mirnov([0,0,coords[set_]['%s%s'%(pos,ul)]['Z']],[0,0,1],
                     '%s_%s%s'%(set_,pos,ul), coords[set_]['%s%s'%(pos,ul)]['R'])
                sensors_Flux_Full.append(sens)
                
    # Mirnov
    for set_ in ['MIRNOV']:
        for phi in [160,340]:
            name_phi='TOR_SET_%03d'%phi
            for hv in ['H','V']:
                for i in np.arange(1,7 if hv=='H' else 10):
                    sens = Mirnov(*__coords_xyz_Mirnov(coords[set_][name_phi]['%s%d'%(hv,i)]),
                                   '%s_%s_%s%d'%(set_,name_phi,hv,i))
                    sensors_Mirnov.append(sens)
    

    # Save in ThinCurr readable format
    # Mirnov object itself is directly readable: can extract location
    save_sensors(sensors_BP,'input_data/floops_BP.loc')
    save_sensors(sensors_BN,'input_data/floops_BN.loc')
    save_sensors(sensors_Flux_Partial,'input_data/floops_Flux_Partial.loc')
    save_sensors(sensors_Flux_Full,'input_data/floops_Flux_Full.loc')
    save_sensors(sensors_Mirnov,'finput_data/loops_MIRNOV.loc')
    
    
    sensors_all.extend(sensors_BP)
    sensors_all.extend(sensors_BN)
    sensors_all.extend(sensors_Flux_Partial)
    sensors_all.extend(sensors_Flux_Full)
    sensors_all.extend(sensors_Mirnov)
    
    return sensors_all, sensors_BP, sensors_BN, sensors_Flux_Partial, sensors_Flux_Full, sensors_Mirnov
    
def __coords_xyz_BP_BN(node,node_angle): # separately select node_angle due to upper/lower swap from CFS
    return ( [node['R']*np.cos(node['PHI']*np.pi/180),
                                 node['R']*np.sin(node['PHI']*np.pi/180),
                                 node['Z']] , \
                [np.cos(node_angle['ANGLE']*np.pi/180)*np.cos(node['PHI']*np.pi/180),
                 np.cos(node_angle['ANGLE']*np.pi/180)*np.sin(node['PHI']*np.pi/180),
                 np.sin(node_angle['ANGLE']*np.pi/180)] )
    
def __coords_xyz_Flux_P(node):
    
    pt1 = [node['R1']*np.cos(node['PHI1']*np.pi/180),
           node['R1']*np.sin(node['PHI1']*np.pi/180),
           node['Z1']]
    pt2 = [node['R1']*np.cos(node['PHI2']*np.pi/180),
           node['R1']*np.sin(node['PHI2']*np.pi/180),
           node['Z1']]
    pt3 = [node['R2']*np.cos(node['PHI2']*np.pi/180),
           node['R2']*np.sin(node['PHI2']*np.pi/180),
           node['Z2']]
    pt4 = [node['R2']*np.cos(node['PHI1']*np.pi/180),
           node['R2']*np.sin(node['PHI1']*np.pi/180),
           node['Z2']]
    
    pts=[];pts.append(pt1);pts.append(pt2);pts.append(pt3);pts.append(pt4)
    return np.array(pts)
        
def __coords_xyz_Mirnov(node): 
    # All mirnov sensors pointed vertically
    return ( [node['R']*np.cos(node['PHI']*np.pi/180),
                                 node['R']*np.sin(node['PHI']*np.pi/180),
                                 node['Z']] , [0,0,1] )

###############################################################################
def __cords_xyz_C_Mod(phi,R,Z,theta_pol,theta_tor):
    # Notice: Theta_Tor (toroidal orientation shift) <=2deg for all sections
    # Ignoring here
    pt = (R*np.cos(phi*np.pi/180), R*np.sin(phi*np.pi/180), Z)
    norm = [np.cos(theta_pol*np.pi/180)*np.cos(phi*np.pi/180),
     np.cos(theta_pol*np.pi/180)*np.sin(phi*np.pi/180),
     np.sin(theta_pol*np.pi/180)]
    return pt, norm
    
def __coords_xyz_Mirnov_C_Mod(phi, theta_pol, R, Z, name):
    pt = (R[name]*np.cos(phi[name]*np.pi/180), R[name]*np.sin(phi[name]*np.pi/180), Z[name])
    norm = [np.cos(theta_pol[name]*np.pi/180)*np.cos(phi[name]*np.pi/180),
     np.cos(theta_pol[name]*np.pi/180)*np.sin(phi[name]*np.pi/180),
     np.sin(theta_pol[name]*np.pi/180)]
    return pt, norm

##################################################
def confluence_spreadsheet_coords(coord_file='input_data/MAGX_Coordinates_CFS.json',
      coord_file_OG='input_data/MAGX_Coordinates.json', comparison='input_data/MAGX_Equilibrium_XYZ.csv'):
    #coords_j = json.load(open(coord_file,'r'))
    comp = np.loadtxt(comparison,delimiter=',',dtype=object,skiprows=1)
    
    #BNMX/BNMX
    coords = {}
    sets = ['BP','BN','SL','FL','MRNV']
    for s in sets:coords[s]={}
    
    for sensor in comp:# loop over entries
        sens_name = sensor[1]
        if sens_name[:2] in sets[:2]:
            coords[sens_name[:2]][sens_name]={'PHI':float('%3.3f'%(float(sensor[3]))),
              'R':float('%3.3f'%(float(sensor[4])*1e-3)),
           'X':float('%3.3f'%(float(sensor[5])*1e-3)),'Y':float('%3.3f'%(float(sensor[6])*1e-3)),
           'Z':float(('%3.3f'%(float(sensor[7])*1e-3))),
           'ANGLE':float('%3.3f'%(float(sensor[8]))),
           'NA':float('%3.3f'%(float(sensor[-1])*float(sensor[-2])*1e-6))}
            
        else:
            if sens_name[:2] == 'SL': # Saddle loop
                coords[sens_name[:2]][sens_name] = {'R1':float('%3.3f'%(float(sensor[4])*1e-3)),
                'Z1':float('%3.3f'%(float(sensor[7])*1e-3)),'PHI1':float('%3.3f'%(float(sensor[3]))),
                'PHI2':float('%3.3f'%(float(sensor[10]))),'R2':float('%3.3f'%(float(sensor[11])*1e-3)),
                'Z2':float('%3.3f'%(float(sensor[14])*1e-3))}
            if sens_name[:2] == 'FL': # Flux loop
                coords[sens_name[:2]][sens_name] = {'R':float('%3.3f'%(float(sensor[4])*1e-3)),
                                                    'Z':float('%3.3f'%(float(sensor[7])*1e-3))}
    # Separately include Mirnov Arrays
    coords_old = json.load(open(coord_file_OG,'r'))
    for set_ in ['MIRNOV']:
        for phi in [160,340]:
            name_phi='TOR_SET_%03d'%phi
            for hv in ['H','V']:
                for i in np.arange(1,7 if hv=='H' else 10):
                    sens_name = 'MRNV_%dM_%s%d'%(phi,hv,i)
                    sens = coords_old[set_][name_phi]['%s%d'%(hv,i)]
                    coords['MRNV'][sens_name] = {'R':float('%3.3f'%sens['R']),
                                                 'Z':float('%3.3f'%sens['Z']),
                                     'PHI':float('%3.3f'%sens['PHI']),
                                     'NA':sens['NA']['NA'][0]}
    
    
    with open(coord_file,'w') as f:json.dump(coords,f)
    return coords
####################################    
def gen_Sensors_Updated(coord_file='input_data/MAGX_Coordinates_CFS.json',
                        select_sensor='MRNV',cmod_shot=1051202011):
    # Try loading predfined sensor sets
    try: return __load_sensor_data(select_sensor)
    
    except: # Manually calculate sensor locations
        
        # Unified function for SPARC, C-Mod 
        if'C_MOD' not in select_sensor:
            coords = json.load(open(coord_file,'r'))
            
            sensors_BP=[];sensors_BN=[];sensors_Flux_Partial=[];sensors_Flux_Full=[];sensors_Mirnov=[]
            sensors_all=[]
            for set_ in coords:
                if set_ in ['BP','BN']:
                    for sensor in coords[set_]:
                        sens = Mirnov(*__coords_xyz_BP_BN(coords[set_][sensor],\
                              coords[set_][sensor]), sensor,7e-3 if set_ == 'BP' else 17e-3)
                        if set_ == 'BP': sensors_BP.append(sens)
                        if set_ == 'BN': sensors_BN.append(sens)
                if set_ == 'SL': # Saddle loop
                    for sensor in coords[set_]:
                        sens = flux_loop(__coords_xyz_Flux_P(coords[set_][sensor]),
                             sensor )
                        sensors_Flux_Partial.append(sens)
                if set_ == 'FL':
                    for sensor in coords[set_]:
                        sens = Mirnov([0,0,coords[set_][sensor]['Z']],[0,0,1],
                             sensor, coords[set_][sensor]['R'])
                        sensors_Flux_Full.append(sens)
                if set_ == 'MRNV':
                    for sensor in coords[set_]:
                        sens = Mirnov(*__coords_xyz_Mirnov(coords[set_][sensor]),
                                      name=sensor,dx=3e-2)
                        sensors_Mirnov.append(sens)
            
            # Save in ThinCurr readable format
            
            # Mirnov object itself is directly readable: can extract location
            save_sensors(sensors_BP,'input_data/floops_BP.loc')
            save_sensors(sensors_BN,'input_data/floops_BN.loc')
            save_sensors(sensors_Flux_Partial,'input_data/floops_SL.loc')
            save_sensors(sensors_Flux_Full,'input_data/floops_FL.loc')
            save_sensors(sensors_Mirnov,'input_data/floops_MRNV.loc')
    
            
            sensors_all.extend(sensors_BP)
            sensors_all.extend(sensors_BN)
            sensors_all.extend(sensors_Flux_Partial)
            sensors_all.extend(sensors_Flux_Full)
            sensors_all.extend(sensors_Mirnov)
            
            if select_sensor == 'BP': return sensors_BP
            if select_sensor == 'BN': return sensors_BN
            if select_sensor == 'SL': return sensors_Flux_Partial
            if select_sensor == 'FL': return sensors_Flux_Full
            if select_sensor == 'MRNV': return sensors_Mirnov
            if select_sensor == 'ALL': return sensors_all
            
            return sensors_all, sensors_BP, sensors_BN, sensors_Flux_Partial, sensors_Flux_Full, sensors_Mirnov
       
        #########################################################################
        
        else:# C-Mod side
            #shot is necessary for minor differences in TOP/BOT sensors
            # Pulls data for limiters only
            phi, theta_pol, R, Z = Mirnov_Geometry_C_Mod(cmod_shot)
            BP_Data = BP(cmod_shot) # Pull data for Low freq sensors
            dx = 4e-3 # True at minimum for BP probes, from 'Magnetic diagnostics in Alcator C‐MOD', R. Granetz 1990
           
            sensor_Mirnov_T = []
            sensor_Mirnov = []
            sensor_BP = []
            sensor_Lim = []
            sensor_all = []
            # doing fast Mirnovs CMOD_T
            for phi_num in ['AB','GH']:
                for sens_num in np.arange(1,7):
                    name = 'BP%dT_%sK'%(sens_num,phi_num)
                    pt, norm = __coords_xyz_Mirnov_C_Mod(phi, theta_pol, R, Z, name)
                    sens = Mirnov(pt, norm, name,dx)
                    sensor_Mirnov_T.append(sens)
                for sens in np.arange(1,29):
                    name = 'BP%d_%sK'%(sens_num,phi_num)
                    if name not in phi:continue
                    pt, norm = __cords_xyz_C_Mod(phi[name], \
                             R[name], Z[name], theta_pol[name], 0)
                    sens = Mirnov(pt, norm, name,dx)
                    sensor_Mirnov.append(sens)
            # Slower BP sensors
            cad_shift = 61.389 # Toroidal rotational shift to match CAD
            for phi_num in ['bc','de','gh','jk']:
                if phi_num=='bc':set_ = BP_Data.BC
                if phi_num=='de':set_ = BP_Data.DE
                if phi_num=='gh':set_ = BP_Data.GH
                if phi_num=='jk':set_ = BP_Data.JK
                for ind,name in enumerate(set_['NAMES']):
                    # Radial, Z shift to avoid cad surface
                    r_shift = 0.01 if 12 <= int(name[2:4]) <= 15 else 0
                    z_shift = 0.01*np.sign(set_['Z'][ind]) if \
                        (10 <= int(name[2:4]) <= 11 ) or \
                        (16 <= int(name[2:4]) <= 17 ) else 0
                    pt,norm = __cords_xyz_C_Mod(set_['PHI'][ind]+cad_shift,\
                        set_['R'][ind] + r_shift,\
                        set_['Z'][ind] + z_shift,\
                        set_['THETA_POL'][ind],set_['THETA_TOR'][ind])
                    sens = Mirnov(pt, norm, name, dx)
                    sensor_BP.append(sens)
            
            ############################################
            sensor_all.extend(sensor_Mirnov_T)
            sensor_all.extend(sensor_Mirnov)
            sensor_all.extend(sensor_BP)
            
            sensor_Lim.extend(sensor_Mirnov_T)
            sensor_Lim.extend(sensor_Mirnov)
            
            # Save in ThinCurr readable format
            # Mirnov object itself is directly readable: can extract location
            save_sensors(sensor_Mirnov_T,'input_data/floops_C_MOD_MIRNOV_T.loc')
            save_sensors(sensor_Lim,'input_data/floops_C_MOD_LIM.loc')
            save_sensors(sensor_BP,'input_data/floops_C_MOD_BP.loc')
            save_sensors(sensor_all,'input_data/floops_C_MOD_ALL.loc')
            
            # Save (x,y,z) coordinates for BP sensor for CAD comparison
            if select_sensor == 'Synth-C_MOD_BP':
                __save_C_Mod_BP_xyz(sensor_BP)
                return sensor_BP
            if select_sensor == 'Synth-C_MOD_BP_T': return sensor_Mirnov_T
            if select_sensor == 'C_MOD_LIM': return sensor_Lim
            if select_sensor == 'C_MOD_BP': return sensor_BP
            if select_sensor == 'C_MOD_ALL': return sensor_all
        
################################################################################
def __save_C_Mod_BP_xyz(sensor_BP):
    X={};Y={};Z={}
    for s in sensor_BP:
        coords = (s._pts[0] + s._pts[1])/2
        X[str(s._name)] = coords[0]
        Y[str(s._name)] = coords[1]
        Z[str(s._name)] = coords[2]
    
    with open('../C-Mod/C_Mod_BP_Geometry_X.json','w', encoding='utf-8') as f: 
        json.dump(X,f, ensure_ascii=False, indent=4)
    with open('../C-Mod/C_Mod_BP_Geometry_Y.json','w', encoding='utf-8') as f: 
        json.dump(Y,f, ensure_ascii=False, indent=4)
    with open('../C-Mod/C_Mod_BP_Geometry_Z.json','w', encoding='utf-8') as f: 
        json.dump(Z,f, ensure_ascii=False, indent=4)
################################################################################
def __load_sensor_data(select_sensor):
    fName = 'input_data/floops_%s.loc'%select_sensor
    if os.path.isfile(fName):
        with open(fName,'r') as f:
            dat=f.readlines()
        
        # Step through the set of sensors
        sensorList = []
        ind = 2
        while ind < len(dat):
            numPts, scale, name = dat[ind].split(' ')
            numPts=int(numPts);scale=float(scale);name=name[:-2]
            sensorPts = []
            for i in range(numPts):
                ind += 1
                ptx,pty,ptz = dat[ind].split(' ')
                ptx=float(ptx);pty=float(pty);ptz=float(ptz)
                sensorPts.append([ptx,pty,ptz])
            
            sensorList.append(flux_loop(np.array(sensorPts), name, scale))
            ind += 2
        return sensorList
    else: raise SyntaxError('File: %s Not Found'%fName)
################################################################################
def debug_plots(set_='BP',tor=0,coord_file='input_data/MAGX_Coordinates_CFS.json'):
    coords = json.load(open(coord_file,'r')) 
    
    plt.close('debug')
    fig,ax=plt.subplots(1,1,num='debug',tight_layout=True)
    for sensor in coords[set_]:
        if -15 + tor < coords[set_][sensor]['PHI'] < 15 + tor:
            x0, norm = __coords_xyz_BP_BN(coords[set_][sensor],\
                  coords[set_][sensor])
            #print(x0,norm)
            x0=np.array(x0)[[0,2]];norm=np.array(norm)[[0,2]]
            #x0=np.array([coords[set_][sensor]['X'],coords[set_][sensor]['Z']])
            ax.plot(x0[0],x0[1],'k*')
        
            ax.plot([x0[0],(x0+np.array(norm)*100)[0]],
                    [x0[1],(x0+np.array(norm)*100)[1]])
    
    plt.grid();plt.show()
####################################
if __name__ == '__main__':gen_Sensors()