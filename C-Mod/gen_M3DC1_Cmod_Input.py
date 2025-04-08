#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 19:12:03 2025
    Wrapper for generating M3D-C1 input files from a given C-Mod shot, timepoint
    Should both output and save: gEqdsk, Te(psi_n), ne(psi_n), aEqdsk coil currents
    Issues with: time alignment: thomson timebase doesn't align with EFIT (?)
    May have discrepency due to sawtooth cycle? (e.g. EFIT, Thomson at different
     phases of the crash)
@author: rian
"""

from header_Cmod import np, sys, plt
from get_Cmod_Data import YAG, A_EQDSK_CCBRSP
sys.path.append('/home/rianc/Documents/eqtools-1.0/')
import eqtools as eq

########################################

def gen_M3DC1_CMod_Input(shotno,timePoint,
             saveDataFile='/home/rianc/Documents/Synthetic_Mirnov/data_output/',
                         doPlot=False, doSavePlot='',dropChansTS=[3],
                         dropChansTS_Edge=[0,1,2,3]):
    
    
    # Get gEQDSK file from C-Mod MDSplus server
    eqfile = eq.CModEFITTree(shotno)
    if saveDataFile:eq.filewriter.gfile(eqfile,timePoint,
                    name=saveDataFile+'g%d.%d'%(shotno,timePoint*1000))
    if doPlot: __verify_Psi(eqfile,shotno,timePoint,doSavePlot)
    
    # Get Equilbirium coil currents
    curr = A_EQDSK_CCBRSP(shotno)
    curr_out = curr.saveOutput(timePoint,saveDataFile)
    if doPlot:curr.makePlots(doSavePlot)
    
    # Get Thomson Ne, Te
    yag = YAG(shotno)
    Te,Ne,R_ts = yag.return_Profile(timePoint,dropChansTS,dropChansTS_Edge)
    if doPlot:yag.makePlot(timePoint,dropChansTS,dropChansTS_Edge,doSavePlot)
    
    # Convert R to psi_N
    R_ts_psi = eqfile.rz2psinorm(R_ts,0,timePoint,sqrt=False)
    
    if saveDataFile: np.savetxt(saveDataFile+'TS_%d_%1.1f'%(shotno,timePoint),
                                [Te,Ne,R_ts_psi])
    
    return eqfile, curr_out,Te, Ne, R_ts_psi, R_ts

################################################
def __verify_Psi(eqfile,shotno,timePoint,doSavePlot):
    r = np.linspace(.4,1,50)
    z = np.linspace(-.25,.25,50)
    psi = eqfile.rz2psinorm(r,z,timePoint,make_grid=True,sqrt=True)
    
    plt.close('Psi_%d_%1.1f'%(shotno,timePoint))
    fig,ax=plt.subplots(1,1,num='Psi_%d_%1.1f'%(shotno,timePoint),figsize=(3,2),
                        tight_layout=True)
    cs = ax.contour(r,z,psi)
    ax.clabel(cs)
    ax.set_xlabel('R [m]')
    ax.set_Ylabel('Z [m]')
    ax.grid()
    
    if doSavePlot: fig.savefig(doSavePlot+fig.canvas.manager.get_window_title()+\
                               '.pdf',transparent=True)
    plt.show()
    
################################################
if __name__ == '__main__':
    shotno = int(sys.argv[1])
    timePoint = float(sys.argv[2])
    
    gen_M3DC1_CMod_Input(shotno, timePoint, doPlot=True,doSavePlot='../output_plots/')
    

