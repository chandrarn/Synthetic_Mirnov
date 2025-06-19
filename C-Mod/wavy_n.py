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

from header_Cmod import plt, np, cv2, iio, sys
from get_Cmod_Data import __loadData
sys.path.append('../signal_analysis/')
from estimate_n_improved import run_n
import matplotlib as mpl

def mode_id_n(wavy_file='Spectrogram_C_Mod_Data_BP_1051202011_Wavy.png',
              tLim_orig=[.75,1.1],fLim_orig=[0,625],shotno=1051202011,
              f_band=[],doSave='',saveExt='',cLim=[]):
    
    # Load in png
    wavy, contour, hierarchy, c_out = gen_contours(wavy_file,tLim_orig,fLim_orig,
                                                   f_band)
    
    # step through contours, set filtering range
    #return wavy, contour
    filter_limits = gen_filter_ranges(c_out,tLim_orig,fLim_orig,wavy.shape[:2])
    
    # Loop through contours
    bp_k = __loadData(shotno,pullData=['bp_k'],forceReload=['bp_k'*False])['bp_k']
    
    #return wavy, contour, filter_limits,bp_k
    n_opt_out=[]
    for ind,lims in enumerate(filter_limits):
        print('Time: %3.3f-%3.3f, Freq: %3.3f-%3.3f'%(*lims['Time'],*lims['Freq']))
        
        # run n-script
        n_opt = run_n(shotno=shotno, tLim=lims['Time'], fLim=None, 
              HP_Freq=lims['Freq'][0]*1e3, LP_Freq=lims['Freq'][1]*1e3,
              n=[lims['Freq'][0]**(1/3)], doSave='',save_Ext='',
              directLoad=True,tLim_plot=[],z_levels = [8,10,14], doBode=True,
              bp_k=bp_k, doPlot=False)[-1]
        filter_limits[ind]['n_opt'] = n_opt
        n_opt_out.append(n_opt)
        #if ind>10:break
    
    # color contours (fill in? step through contour horizontally, fill vertically?)
    overplot_n(filter_limits,wavy,tLim_orig,fLim_orig,n_opt_out,doSave,
               saveExt,shotno,cLim)
    
    return wavy, contour, hierarchy, c_out,filter_limits,n_opt_out
############################################################
def overplot_n(filter_limits,wavy,tLim_orig,fLim_orig,n_opt,doSave,saveExt,
               shotno,cLim):
    title = 'WavyStar_n_Estimator_%d%s'%(shotno,saveExt)
    plt.close(title)
    fig,ax = plt.subplots(1,1,num=title,tight_layout=True)
    
    tRange = np.linspace(*tLim_orig,wavy.shape[0])
    fRange = np.linspace(*fLim_orig,wavy.shape[1])
    wavy = wavy[:,:,0]
    
    ax.contourf(tRange,fRange,wavy,cmap=plt.get_cmap('Greys'),zorder=-5)
    
    cmap = mpl.cm.viridis
    bounds = np.arange( *(cLim if cLim else [np.nanmin(n_opt),np.nanmax(n_opt)+1]) )

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

    for ind,lims in enumerate(filter_limits):
        if np.isnan(n_opt[ind]):continue
        contours = lims['Contours']
        contours = np.vstack((contours,contours[0]))
        
        ax.plot(tRange[contours[:,0]],fRange[contours[:,1]],c=cmap(norm(n_opt[ind])) )
    
    ax.text(.04,.97,'%d'%shotno,transform=ax.transAxes,fontsize=8,
            verticalalignment='top',bbox={'boxstyle':'round','alpha':.7,
                                          'facecolor':'white'})
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Frequency [kHz]')
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             label='n\#',ax=ax)
    ax.set_rasterization_zorder(-1)
    plt.show()
    
    if doSave: 
        fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',transparent=True)
############################################################
def gen_filter_ranges(c_out,tLim_orig,fLim_orig,dimImage):
    # Convert coordinates from pixels to time, freq
    tRange = np.linspace(*tLim_orig,dimImage[0])
    fRange = np.linspace(*fLim_orig,dimImage[1])
    
    filter_limits = [] # structured as time, spatial limits, contour#
    for c in c_out:
        try:
            filter_limits.append(\
             {'Time':[tRange[np.min(c[:,0])], tRange[np.max(c[:,0])]],\
              'Freq':[fRange[np.min(c[:,1])], fRange[np.max(c[:,1])]],
              'Contours':c})
        except: 
            print(c)
            raise SyntaxError
    return filter_limits
############################################################
def gen_contours(wavy_file,tLim_orig,fLim_orig,f_band = [],
                 min_contour_w=10,min_contour_h=7,
                 doPlot=True,):
    wavy = iio.imread('../output_plots/'+wavy_file)
    wavy=wavy[::-1,:,:]
    
    tLim = np.linspace(*tLim_orig,wavy.shape[0])
    fLim = np.linspace(*fLim_orig,wavy.shape[1])
    if doPlot:
        plt.close('Input')
        fig,ax = plt.subplots(1,4,tight_layout=True,sharex=True,sharey=True,num='Input')

        for i in range(4):ax[i].contourf(tLim,fLim,wavy[:,:,i])
    
    # Contour detect a la geqdsk cv2.findContours
    contour,hierarchy=cv2.findContours(np.array(wavy[:,:,0]==253,dtype=np.uint8),
                                       cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    c_out=[]
    for c in contour: 
        c=c.squeeze()
        # minimum size checks for valid contour
        if len(c)<=2:continue
        if np.max(c[:,0])-np.min(c[:,0])<min_contour_w: continue
        if np.max(c[:,1])-np.min(c[:,1])<min_contour_h: continue
        
        # frequency limit
        if f_band:
            if fLim[np.min(c[:,1])] < f_band[0]: continue
            if fLim[np.max(c[:,1])] > f_band[1]: continue
    
        if doPlot: 
            ax[0].plot(tLim[c[:,0]],fLim[c[:,1]],'r-',lw=1)
            plt.show()
        c_out.append(c)
        
    return wavy, contour, hierarchy, c_out
###############################################3
if __name__ == '__main__':mode_id_n()