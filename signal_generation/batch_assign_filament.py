#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 18:38:14 2025

@author: rianc
"""
from Synthetic_Mirnov import gen_synthetic_Mirnov,np

if __name__ == '__main__':
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
    f = 1e3
    for n in np.arange(1,13):
        for m in np.arange(n,13):
            mode_list.append([m,n])
            params={'m':m,'n':n,'r':.25,'R':1,'n_pts':100,'m_pts':70,\
                'f':10e3,'dt':1e-6,'T':3e-4,'periods':3,'n_threads':64,'I':10}
            gen_synthetic_Mirnov(mesh_file=mesh_file,sensor_set=sensor_set,params=params,
                     save_ext=save_ext,doSave=doSave,archiveExt='training_data/',
                     doPlot=False)