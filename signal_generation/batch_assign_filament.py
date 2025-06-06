#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 18:38:14 2025

@author: rianc
"""
from Synthetic_Mirnov import gen_synthetic_Mirnov,np, json,os
from subprocess import check_output, STDOUT

def record_storage(m,n,f,save_ext,sensor_set,archiveExt):
    fName = 'floops_filament_%s_m-n_%d-%d_f_%d%s.hist'%\
        (sensor_set,m,n,f,save_ext)
    if os.path.exists('../data_output/%sSimulation_Params.json'%archiveExt):
        with open('../data_output/%sSimulation_Params.json'%archiveExt,'r') as f:
            params = json.load(f)
    else:params={}
    
    params[fName]={'m':m,'n':n,'f':f}
    with open('../data_output/%sSimulation_Params.json'%archiveExt,'w') as f:
        json.dump(params,f)
if __name__ == '__main__':
    archiveExt='training_data/'
    print( check_output('rm ../data_output/%sSimulation_Params.json'%archiveExt,\
            shell=True, stderr=STDOUT).decode('utf-8') )
        
    mesh_file='C_Mod_ThinCurr_Combined-homology.h5'

    file_geqdsk='g1051202011.1000'
    sensor_set='Synth-C_MOD_BP_T';cmod_shot=1051202011
    #mesh_file='SPARC_Sept2023_noPR.h5'
    # mesh_file='thincurr_ex-torus.h5'
    #sensor_set='MRNV'
    #mesh_file='vacuum_mesh.h5'
    save_ext=''
    doSave='../output_plots/'*False
    #params={'m':18,'n':16,'r':.25,'R':1,'n_pts':70,'m_pts':60,'f':500e3,'dt':1e-7,'periods':3,'n_threads':64,'I':10}

    
    mode_list = []
    f = 10e3
    for n in np.arange(1,13,dtype=int):
        for m in np.arange(n,13,dtype=int):
            mode_list.append([m,n])
            params={'m':m,'n':n,'r':.25,'R':1,'n_pts':100,'m_pts':70,\
                'f':f,'dt':1e-6,'T':3e-4,'periods':3,'n_threads':64,'I':10}

            record_storage(m,n,f,save_ext,sensor_set,archiveExt)
            gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_set=sensor_set,params=params,
                     save_ext=save_ext,doSave=doSave,archiveExt=archiveExt,
                     doPlot=False)
            
