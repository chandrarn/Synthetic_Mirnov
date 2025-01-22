#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:47:17 2025
    Pull sensor coordinates from Granetz tree
@author: rian
"""

try:import MDSplus as mds
except:import mdsthin as mds
import numpy as np
import json
from header import Mirnov, save_sensors, flux_loop

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
                    
            
    with open('MAGX_Coordinates.json','w') as f:json.dump(nodes,f)
        
#######################################
def gen_Sensors(coord_file='MAGX_Coordinates.json'):
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
                              coords[set_]['%s%s'%(pos,ul)]['R'])
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
    save_sensors(sensors_BP,'floops_BP.loc')
    save_sensors(sensors_BN,'floops_BN.loc')
    save_sensors(sensors_Flux_Partial,'floops_Flux_Partial.loc')
    save_sensors(sensors_Flux_Full,'floops_Flux_Full.loc')
    save_sensors(sensors_Mirnov,'floops_Mirnov.loc')
    
    
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
        

####################################
if __name__ == '__main__':gen_Sensors()