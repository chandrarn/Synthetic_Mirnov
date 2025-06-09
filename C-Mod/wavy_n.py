#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 16:27:44 2025
    Load in WavyStar filtered spectrogram
    option one: contour detect, get single n number for each blob
    option two: step through in time, allowing for time varying n
    
    ToDo: get tLim, fLim automatically
@author: rian
"""

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2

def mode_id_n(wavy_file='Spectrogram_C_Mod_Data_BP_1051202011_Wavy.png',
              tLim_orig=[.75,1.1],fLim_orig=[0,625]):
    
    # Load in png
    wavy, contour, hierarchy, c_out = gen_contours(wavy_file)
    
    # step through contours, set filtering range
    filter_limits = gen_filter_ranges(c_out,tLim_orig,fLim_orig,wavy.shape[:2])
    
    # Loop through contours
    for 
        # load in signal data, filter accordingly
        
        
        # run n-script
        n_opt = run_n(shotno=1160930034, tLim=[0.82,0.82008], fLim=None, 
              HP_Freq=400e3, LP_Freq=600e3,n=[2], doSave='',save_Ext='',
              directLoad=True,tLim_plot=[],z_levels = [8,10,14], doBode=True,
              bp_k=None,data=None,f_samp=None,R=None,Z=None,phi=None,
              inds=None, time=None, doPlot=True)[-1]
    
    # color contours (fill in? step through contour horizontally, fill vertically?)
    
    return wavy, contour, hierarchy, c_out
############################################################
def gen_filter_ranges(c_out,tLim_orig,fLim_orig,dimImage):
    # Convert coordinates from pixels to time, freq
    tRange = np.linspace(*tLim_orig,dimImage[0])
    fRange = np.linspace(*fLim_orig,dimImage[1])
    
    filter_limits = [] # structured as time, spatial limits, contour#
    for c in c_out:
        filter_limits.append(\
             {'Time':[tRange[np.argmin(c[:,0])], tRange[np.argmax(c[:,0])]],\
              'Freq':[fRange[np.argmin(c[:,0])], fRange[np.argmax(c[:,0])]],
              'Contours':c})
    
    return filter_limits
############################################################
def gen_contours(wavy_file,min_contour_w=4,min_contour_h=4,doPlot=True):
    wavy = iio.imread('../output_plots/'+wavy_file)
    
    if doPlot:
        plt.close('Input')
        fig,ax = plt.subplots(1,4,tight_layout=True,sharex=True,sharey=True,num='Input')
        for i in range(4):ax[i].imshow(wavy[:,:,i])
    
    # Contour detect a la geqdsk cv2.findContours
    contour,hierarchy=cv2.findContours(np.array(wavy[:,:,0]==253,dtype=np.uint8),
                                       cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    c_out=[]
    for c in contour: 
        c=c.squeeze()
        # minimum size checks for valid contour
        if len(c)<=2:continue
        if np.max(c[:,0])-np.min(c[:,0])<min_contour_w: continue
        if np.max(c[:,1])-np.min(c[:,1])<min_contour_w: continue
    
        if doPlot: ax[0].plot(c[:,0],c[:,1],'r-',lw=1)
        c_out.append(c)
        
    return wavy, contour, hierarchy, c_out
###############################################3
if __name__ == '__main__':mode_id_n()