#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:47:17 2025
    Pull sensor coordinates from Granetz tree
@author: rian
"""

import MDSplus as mds
import numpy as np
import json

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
            if node_name != ' FLUX_PARTIAL':pos_list.extend(['OLIM'])
            if node_name != 'BN':pos_list.extend(['IOL1','IOL2','OOLM','VSCM'])
            if node_name =='FLUX_PARTIAL':pos_list.extend(['DVT6','IMID','OMPS'])
            for pos in pos_list:
                for ul in ['_L','_U']:
                    nodes[node_name]['TOR_SET_%03d'%phi]['%s%s'%(pos,ul)]={}
                    for sig in ['ANGLE','NA','R','Z','PHI','POLARITY']:
                        tag = 'SIGNALS.%s.%s.%s%s.%s'%(node_name,name_phi,pos,ul,sig)
                        try:nodes[node_name][name_phi]['%s%s'%(pos,ul)][sig] = \
                            float(tree.getNode(tag).getFloatArray())
                        except:
                            print('Failure at: %s'%tag)
                            raise SyntaxError
            if node_name=='FLUX_PARTIAL':
                for pos in ['IMID','OMID']:
                    for ul in ['_M']:
                        nodes[node_name]['TOR_SET_%03d'%phi]['%s%s'%(pos,ul)]={}
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
                    for sig in ['ANGLE','NA','R','Z','PHI','POLARITY']:
                        tag = 'SIGNALS.%s.%s.%s%d.%s'%(node_name,name_phi,hv,i,sig)
                        try:nodes[node_name][name_phi]['%s%d'%(hv,i)][sig] = \
                            float(tree.getNode(tag).getFloatArray())
                        except:
                            print('Failure at: %s'%tag)
                            raise SyntaxError
            
    with open('MAGX_Coordinates.json','w') as f:json.dump(nodes,f)
        

if __name__ == '__main__':gen_Node_Dict()