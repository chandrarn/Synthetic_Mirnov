#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:25:07 2025
    Equilbirium evolution signal trace code
@author: rianc
"""

from header_Cmod import plt, np, Rectangle, cm
from get_Cmod_Data import __loadData

###############################################################################
def plot_basic_quantities(shots, LP_Freq = 30e3, doSave=False, tLim=None,
                save_Ext='', highlightTimes = None, 
                makePlots = ['ip', 'p_rf', 'gpc', 'bp_t'],
                plotChan={},data_archive='',\
                    doCPCI_Zeroing = False, yLims = {},large=False,\
                        overlayOptical=False,reload=[],\
                        indicateTimes=[],narrow=False,\
                        headless=False,debug=False,\
                        suppressY=False,suppressX=False,plotTempRm=False,
                        annotateTimePoints=False,\
                        doTempRm=True,doSaveData=True,shotLabelsExt=[],
                        limit_legend=True,manual_t_align=[],xLim=None,
                        showLegend=True,overrideImport={},rw=0,\
                        correctMag_Rm_q_Rw=False,labelShotNumber=True,\
                        leg_loc='upper left',tall=False,\
                        tLim_Manual={},mAmp_inds=np.arange(14,19),doAspect=False):
    """
    Plot equilibrium and diagnostic signals, higlight time regions if necessary
    
    Selected arguments:
    ---------
    shots/tLim:
        Shot number and tLim self explainitory.
        Can handle multiple shots, with differing time windows
        Negative time values are referenced from disruption, unless manual_t_align
        is set
    makePlots:
        The diagnostics to be plotted are set by makePlots, options can be found
        in makePlots subfunction
    yLims: 
        dictionary labled by diagnostic names, to manually set specific
        time limits, e.g. yLims = {'ip':[10,20]}
    tLim_Manual:
        Similar to prev. Both functions can also handle diagnostic specific, 
        shot specific limits if nested lists of limits are given
    annotateTimePoints: 
            if single values, draws dashed lines, if tuple, draws
            a box
    indicateTimes: 
            the standard traces will have a lighter transparency, 
            the time window in indicateTimes will be bolded
    
    
    """
    
    
    # Make Figure
    fig, ax, figTitle, shots = __buildFigure(shots,makePlots,save_Ext,tall,suppressY,narrow)
    
    for shot_ind,shot in enumerate(shots):
        
        # Only get the data necssary
        rawData = __loadData(shot,pullData=makePlots,data_archive=data_archive)
        
        
        # Plots
        for plot_ind,signal in enumerate(makePlots):
            for line in range(1+bool(highlightTimes) ):
                
                # Get signal, timebase, label
                data,time,yLabel = __selectSignal(rawData,signal,plotChan,
                      tLim_Manual,shot_ind,manual_t_align,
                                   tLim,line,highlightTimes)
                
                
                # Plot signal
                __plotSignal(line,highlightTimes,shot_ind,shots,labelShotNumber,shot,
                                 shotLabelsExt,plot_ind,time,data,ax,limit_legend,
                                 yLabel,signal,yLims,annotateTimePoints,xLim)
               
    # Clean up plots, save
    __doCleanup(ax,fig,leg_loc,makePlots,shots,suppressY,doSave,figTitle,
                   indicateTimes,manual_t_align,suppressX,tLim,showLegend)
    
    return ax, rawData
###############################################################################
def __buildFigure(shots,makePlots,save_Ext,tall,suppressY,narrow):
    
    shots = [shots] if type(shots) == int else shots
    
    figTitle = 'Equilibrium Signals Shot%s'%(' %d'%shots[0] if len(shots)==1 else \
                                 's %d-%d'%(shots[0],shots[-1]) ) + save_Ext
    plt.close(figTitle)
    height = len(makePlots)*((1. if tall else .9) if len(makePlots) >=3 else 1.25)
    fig,ax = plt.subplots(len(makePlots),1,\
        num=figTitle,sharex=True,tight_layout=True,\
          figsize=(((2.5-0*suppressY) if narrow else 3.404)-.5*suppressY,height),\
              squeeze=True)
    if np.size(ax) == 1: ax = [ax]
    
    return fig,ax,figTitle,shots

###############################################################################
def __selectSignal(rawData,signal,plotChan,tLim_Manual,shot_ind,manual_t_align,
                   tLim,line,highlightTimes):
    
    ##### Select signal subset
    node = rawData[signal]
    if signal == 'ip':
        data = node.ip*1e-3
        time = node.time
        label = r'$\mathrm{I_p}$ [kA]'
    elif signal == 'bp':
        # outboard wall sensor
        data = node.BC['SIGNAL'][plotChan[signal] if signal in plotChan else 14]
        time = node.time
        label = r'$\mathrm{B}_\theta$ [G]'
    elif signal == 'bp_t':
        data = node.ab_data[plotChan[signal] if signal in plotChan else 3]
        time = node.time
        label =r'$\partial_t\mathrm{B}_\theta$ [G/S]'
    elif signal == 'gpc':
        data = node.Te[plotChan[signal] if signal in plotChan else 4]
        time = node.time
        label = r'$\mathrm{T_e}$ [keV]'
    elif signal == 'gpc_2':
        data = node.Te[plotChan[signal] if signal in plotChan else -1]
        time = node.time
        label = r'$\mathrm{T_e}$ [keV]'
    elif signal =='p_rf':
        data = node.pwr
        time = node.time
        label = r'$\mathrm{P_{rf}}$ [MW]'
        
        
    ####### Trim to desired time range
    if manual_t_align: shifttime = manual_t_align[shot_ind]*1e-3
    else: shifttime=0
        
    timeBound =  np.array ( \
   ((tLim_Manual[signal][shot_ind] if np.size(tLim_Manual[signal][0]) > 1\
     else tLim_Manual[signal]) if signal in tLim_Manual else \
           ( [-.25,2.5] if not np.any(tLim) else \
        (tLim[shot_ind] if np.size(tLim[0])==2 else tLim) ) ) \
            if line == 0 else highlightTimes[shot_ind]  ) + shifttime
        
    tLim_ = np.arange(*[np.argmin((time-t)**2) \
                      for t in timeBound ] ) 
    time = ( time[tLim_]- shifttime )
    data = data[tLim_]
    
    
    return data,time,label
    
###############################################################################
def __doCleanup(ax,fig,leg_loc,makePlots,shots,suppressY,doSave,figTitle,
               indicateTimes,manual_t_align,suppressX,tLim,showLegend):
    # Cleanup
    if showLegend:ax[0].legend(fontsize=8,handlelength=1,loc=leg_loc,\
                               ncol=(1+len(shots)//4) )
    if suppressX:
        ax[-1].set_xticklabels([])
        ax[-1].format_coord = lambda x, y: 'x={:g}, y={:g}'.format(x, y)
    else:
        ax[-1].set_xlabel(r'$\mathrm{t}-\tau_\mathrm{o}$ [s]' \
          if manual_t_align else 'Time [s]')

    if suppressY:
        for axes in ax:
            axes.set_ylabel('');
            axes.set_yticklabels([])
            
    for i in range(len(makePlots)):ax[i].grid()
    for axes in ax:
        for t in indicateTimes:
                    axes.plot([t,t],np.array(axes.get_ylim())*1,'k--',alpha=.5)
        
    __plotSignal
    if doSave:
        fig.savefig(doSave + figTitle + '.pdf', transparent=True)
        print('Saving: '+doSave + figTitle + '.pdf')
    #return ax

###############################################################################
def __plotSignal(line,highlightTimes,shot_ind,shots,labelShotNumber,shot,
                 shotLabelsExt,plot_ind,time,data,ax,limit_legend,
                 yLabel,signal,yLims,annotateTimePoints,xLim):
    # shot group labeling
    alpha = .4 + .6*line if bool(highlightTimes) else \
        1 - .6*(shot_ind)/len(shots)
        
    label = (('%d'%shot if labelShotNumber else '')+ \
         ((': '+shotLabelsExt[shot_ind]) if shotLabelsExt else '')) \
        if plot_ind==0 and \
        (line == (1 if bool(highlightTimes) else 0)) else None
        
    # Plot Signal
    ax[plot_ind].plot(time, data,alpha=alpha,\
          label=None if (limit_legend and shot_ind>6) else label,\
              color = cm.get_cmap('tab10')(shot_ind))
    
    # Clean up axes
    ax[plot_ind].set_ylabel(yLabel)
    if signal in yLims: 
        ax[plot_ind].set_ylim(yLims[signal])   
    elif line==1:# Highlighted time case
        ax[plot_ind].set_ylim([.9*min(data),1.1*max(data)])
    if annotateTimePoints and shot_ind==line==0:
        annotatePlot(ax[plot_ind],annotateTimePoints,time)
    if xLim:ax[plot_ind].set_xlim(xLim)

###############################################################################    
def annotatePlot(ax,annotateTimePoints,time):
    for t in annotateTimePoints:
        if type(t)!=list:
            t=[t,t+(time[-1]-time[0])*2.5e-2]
            alpha=.2
        else:alpha=.2
        #r=Rectangle( (rectLeft,rectBot),rectWidth,rectHeight,alpha=.5)
        yLimTmp = ax.get_ylim()
        r=Rectangle( (t[0],-100),t[1]-t[0],2000,alpha=alpha,\
                    color='k',zorder=-10)
        ax.add_patch(r)
        ax.set_ylim(yLimTmp)
        ax.set_rasterization_zorder(-1)