#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 16:21:32 2025
    C-Mod data spectrogram
    
@author: rianc
"""

from header_Cmod import np, plt, Normalize, cm, rolling_spectrogram, __doFilter, \
    gaussianHighPassFilter


import get_Cmod_Data as gC

###############################################################################

def signal_spectrogram_C_Mod(shotno=1051202011,sensor_set='BP',diag=None,
                            doSave='',pad=1000,fft_window=500,tLim=[.75,1.1],
                            signal_reduce=2,f_lim=None,sensor_name='BP2T_ABK',
                            debug=True,plot_reduce=3,doColorbar=True,
                            clabel='',HP_Freq=100,cLim=None,
                            doSave_Extractor=True,block_reduce=[400,20000],
                            data_archive=''):
    '''
    

    Parameters
    ----------
    shotno : INT
        Shot number. The default is 1051202011.
    sensor_set : STR
        Which diagnostic to operate on. The default is 'BP'.
    diag : OBJ, optional
        Data object for sensor, can be reinput for faster loading. The default is None.
    doSave : Str, optional
        File in which to save plots. The default is ''.
    pad : INT
        Pad signal with zeros (increases FFT resolution). The default is 1000.
    fft_window : INT
        FFT smoothing window width. The default is 500.
    tLim : [INT,INT]
        Time range to operate on, in [s]. The default is [.75,1.1].
    signal_reduce : INT, optional
        Downsample diagnostic signal. The default is 2.
    f_lim : [FLOAT, FLOAT], optional
        Frequency range for plot. The default is None.
    sensor_name : STR, optional
        Name of specific sensor within diagnostic if needed. The default is 'BP2T_ABK'.
    debug : BOOL, optional
        Debug output printing. The default is True.
    plot_reduce : INT, optional
        Downsample spectrogram for plotting. The default is 3.
    doColorbar : BOOL, optional
        Add colorbar. The default is True.
    clabel : STR, optional
        Colorbar label. The default is r'$\tilde{\mathrm{B}}_\theta$ [G]'.
    doSave_Extractor : BOOL, optional
        Save .png without axes labels/etc, for feature extraction. The default is True.
    block_reduce : [INT, INT]
        Reduce sample in blocks (preserves high frequency signals, at the cost o
         introducing aliasing "noise"): cut the signal in blocks of [keep samples, drop samples]

    Returns
    -------
    diag : OBJ
        Diagnostic data output, returned for faster access.
    signals : ARRAY
        Raw signals.
    time : ARRAY
        Time vector.
    out_spect : ARRAY
        Spectrogram.
    freq : ARRAY
        Frequency vector.

    '''
    
    if tLim[1]-tLim[0]>.3: 
        print('Warning! The requested time range may be too' + \
            ' large to plot, consider increasing signal or plot reduction term')
    
    # Pull C-Mod data from server for a given diagnostic
    if sensor_set == 'BP_T':
        if diag is None: diag = gC.__loadData(shotno,pullData=['bp_t'],
                                   data_archive=data_archive)['bp_t']
            
        # Extract specific sensor, if desired. Otherwise FFTs will be averaged
        signals = diag.ab_data if sensor_name == '' else \
            diag.ab_data[(np.array(diag.ab_names) ==  sensor_name)]
        time = diag.time
        
        clabel=r'$\partial_t{\mathrm{B}}_\theta$ [T/s]'
        
    elif sensor_set == 'FRCECE':
        if diag is None: diag = diag = gC.__loadData(shotno,pullData=['frcece'],
                                   data_archive=data_archive)['frcece']
        signals = diag.ECE*1e3 # convert from keV to eV
        # For the FRC-ECE, sensor_name is just a channel number
        signals = signals[sensor_name] if sensor_name else signals
        if np.ndim(signals) == 1: signals = signals[np.newaxis,:]
        time = diag.time
        clabel = r'$\tilde{\mathrm{T}}_e$ [eV]'
        
    elif sensor_set == 'GPC':
        if diag is None: diag = diag = gC.__loadData(shotno,pullData=['gpc'],
                                   data_archive=data_archive)['gpc']
        signals = diag.Te*1e3 
        time = diag.time
        clabel = r'$\tilde{\mathrm{T}}_e$ [eV]'
        
    elif sensor_set == 'BP':
        if diag is None: diag = diag = gC.__loadData(shotno,pullData=['bp'],
                                   data_archive=data_archive)['bp']
        signals = diag.BC['SIGNAL'] if sensor_name == '' else\
            diag.BC['SIGNAL'][diag.BC['NAMES']==sensor_name]
        time = diag.time
        clabel=r'$\tilde{\mathrm{B}}_\theta$ [G]'
        
    else: raise SyntaxError('Selected Diagnostic Not Yet Implemented')
    
    if debug: print('Loaded: %s'%sensor_set)
    
    # # Desired time range to operate on
    # t_inds = np.arange((tLim[0]-time[0])/(time[1]-time[0]),\
    #           (tLim[1]-time[0])/(time[1]-time[0]),dtype=int)[::signal_reduce]
        
    t_inds = np.arange(*[np.argmin((time-t)**2) for t in tLim],dtype=int)[::signal_reduce]
    
    
    if block_reduce:
        del_inds=np.array([],dtype=int)
        for ind in np.arange(0,len(t_inds)-block_reduce[0],np.sum(block_reduce)):
            del_inds = np.append(del_inds,np.arange(ind,ind+block_reduce[0],dtype=int))
        #return t_inds,del_inds
        t_inds = np.delete(t_inds,del_inds)
        
    # Run filtering 
    signals = __doFilter(signals, time, HP_Freq, None)
    
    #return t_inds,time
    # Run spectrogram
    time, freq, out_spect = rolling_spectrogram(time[t_inds], signals[:,t_inds],pad=pad,
                                                fft_window=fft_window)

    # Average across channels
    out_spect=np.mean(out_spect,axis=0)
    
    if debug: print('Computed Spectrogram, Size ',out_spect.shape)    
        
    # Plot Spectrogram
    # Reduce plot resolution:
    time = time[::plot_reduce]
    out_spect = out_spect[:,::plot_reduce]
    plot_spectrogram(time,freq,out_spect,doSave,sensor_set,None,
                 None,'',f_lim,shotno,doSave_Extractor=doSave_Extractor,tScale=1,
                 doColorbar=doColorbar,clabel=clabel,cLim=cLim)
    
    if debug: print('Finished Plot')
    
    return diag,signals,time,out_spect, freq

###############################################################################

def plot_spectrogram(time,freq,out_spect,doSave,sensor_set,params,filament,
                     save_Ext,f_lim,shotno=None,
                     doSave_Extractor=False,tScale=1e3,doColorbar=False,
                     clabel='',cLim=None):
    
    name = 'Spectrogram_%s'%'-'.join(sensor_set) if type(sensor_set) is list else 'Spectrogram_%s'%sensor_set
    name='%s%s'%(name,save_Ext)
    fName = __gen_fName(params,sensor_set,save_Ext,filament,shotno)   
    plt.close(name)
    
    fig,ax=plt.subplots(1,1,tight_layout=True,num=name)
    
    # Get fLim
    f_inds = np.arange(*[np.argmin((freq*1e-3-f)**2) for f in f_lim]) if f_lim\
        else np.arange(*[0,len(freq)])
    
    vmin,vmax = cLim if cLim else [np.min(out_spect[f_inds]),np.max(out_spect[f_inds])]
    ax.pcolormesh(time*tScale,freq[f_inds]*1e-3,out_spect[f_inds],
                  shading='auto',rasterized=True,vmin=vmin,vmax=vmax)
    
    if doSave_Extractor and doSave: 
        # For mode extractor code, remove axes labels/etc
        plt.tick_params(left = False, bottom = False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig(doSave+'Spectrogram_%s.png'%fName,
                    bbox_inches='tight',pad_inches=0)
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.tick_params(left = True, bottom = True)
    
    ax.set_xlabel('Time [%s]'%('s' if tScale==1 else 'ms'))
    ax.set_ylabel('Freq [kHz]')
    
    if doColorbar:
        norm = Normalize(vmin,vmax)
        fig.colorbar(cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,
                     label=clabel)
        
    if shotno:ax.text(.05,.95,'%d'%shotno,transform=ax.transAxes,fontsize=8,
            verticalalignment='top',bbox={'boxstyle':'round','alpha':.7,
                                          'facecolor':'white'})
    
    #if f_lim:ax.set_ylim(f_lim)
    
    plt.show()

    if doSave: fig.savefig(doSave+'Spectrogram_%s.pdf'%fName,transparent=True)
        
###############################################################################
def __gen_fName(params,sensor_set,save_Ext,filament,shotno):
    if params:
        fName = 'floops_%s_m-n_%s-%s_f_%s%s'%\
             ( '-'.join(sensor_set),'-'.join(['%d'%param['m'] for param in params]),
              '-'.join(['%d'%param['n'] for param in params]),
              '-'.join(['%d'%(param['f']*1e-3) for param in params]),save_Ext) if \
                 type(params) is list else 'floops_%s_%s_m-n_%d-%d_f_%d%s.pdf'%\
             ('filament' if filament else 'surface', sensor_set,params['m'],
              params['n'],params['f']*1e-3,save_Ext)
    else:
        fName = 'C_Mod_Data_%s_%d%s'%(sensor_set,shotno,save_Ext)
    return fName