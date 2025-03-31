#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 23:07:36 2025
tmp contour plot code for BP sensors
@author: rian
"""
from header_Cmod import plt, np, __doFilter

        
def makePlot(dat,HP=10,LP=20e3,tLim=[1,1.2]):
    plt.close('BP_Contour')
    fig,ax=plt.subplots(2,1,tight_layout=True,num='BP_Contour',sharex=True)
    
    out= np.array(__doFilter(dat.BC['SIGNAL'], dat.time, HP, LP))
    
    
    dt = dat.time[1]-dat.time[0]
    tInds = np.arange(*[np.argmin((dat.time-t)**2) for t in tLim],dtype=int)
    
    #return out
    theta = np.arctan2(dat.BC['Z'],np.array(dat.BC['R'])-.67)*180/np.pi
    ax[0].contourf(dat.time[tInds],theta,out[:,tInds],levels=50)
    
    ax[0].set_ylabel(r'$\hat{\theta}$ [deg]')
    ax[1].set_xlabel('Time [s]')
    
    
    plt.show()