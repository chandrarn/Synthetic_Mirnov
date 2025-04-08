#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 19:12:03 2025
    Wrapper for generating M3D-C1 input files from a given C-Mod shot, timepoint
    Should both output and save: gEqdsk, Te(psi_n), ne(psi_n), aEqdsk coil currents
    
    Issues with timebase: EFIT, Thomson data is selected based on nerest-neighbor
    interpolation. This could potentially lead to inconsistencies, particularly
    with respect to the sawtooth crash cycle (e.g. TS laser could end up reporting
    from top of cycle, EFIT from bottom)
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
    R_ts_psi = eqfile.rz2psinorm(R_ts,0,timePoint,sqrt=True,make_grid=True)[0]
    if saveDataFile: np.savetxt(saveDataFile+'TS_%d_%1.1f'%(shotno,timePoint),
                                [Te,Ne,R_ts_psi])
    
    
    return eqfile, curr_out,Te, Ne, R_ts_psi, R_ts

################################################
def __verify_Psi(eqfile,shotno,timePoint,doSavePlot):
    # Verification plot for Psi_n contours
    
    r = np.linspace(.4,1,50)
    z = np.linspace(-.25,.25,50)
    psi = eqfile.rz2psinorm(r,z,timePoint,make_grid=True,sqrt=True)
    
    plt.close('Psi_%d_%1.1f'%(shotno,timePoint))
    fig,ax=plt.subplots(1,1,num='Psi_%d_%1.1f'%(shotno,timePoint),figsize=(3,4),
                        tight_layout=True)
    cs = ax.contour(r,z,psi)
    ax.clabel(cs)
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    l1=ax.plot([.7,.71],[0,0],'-k',label=r'$\sqrt{\psi_n}$')
    ax.legend(loc='upper left',fontsize=8,handlelength=1.5,
              title='%d: %1.1fs'%(shotno,timePoint),title_fontsize=8)
    l1[0].remove()
    ax.grid()
    
    if doSavePlot: fig.savefig(doSavePlot+fig.canvas.manager.get_window_title()+\
                               '.pdf',transparent=True)
    plt.show()
    
################################################
if __name__ == '__main__': #Command line operations
    shotno = int(sys.argv[1])
    timePoint = float(sys.argv[2])
    
    gen_M3DC1_CMod_Input(shotno, timePoint, doPlot=True,
             doSavePlot='/home/rianc/Documents/Synthetic_Mirnov/output_plots/')
    

