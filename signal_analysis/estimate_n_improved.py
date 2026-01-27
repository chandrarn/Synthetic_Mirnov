#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 22:57:33 2025
 Rewrite of phase analysis code for n#
 Use Ted's data pulling functinoality
 May run into issues for large data frames
 
 - select rings of sensors
 - 
 
 - Eventually: Interface with fft spectrogram selection code:
     Given fourier signal, return n# within frequencyt peak
  Eventually: needs to be able to pull in from senosrs ( need to add all HF sensors)
  
  rawData=gc.__loadData(1051202011,forceReload=['bp_k'],params={'skipInteger':0,'tLim':[.921,.923]},pullData=['bp_k'])
@author: rian
"""


try:from header_signal_analysis import CModEFITTree, doFFT, butter, sosfilt
except:pass
from sys import path; path.append('../C-Mod/')
from header_Cmod import __doFilter, correct_Bode, np, hilbert, minimize, plt, remove_freq_band
from get_Cmod_Data import __loadData

from scipy.stats import mode
# Assume that we have some way of grouping in frequency/time
def run_n(shotno=1160930034, tLim=[1.1,1.11], fLim=None, 
          HP_Freq=670e3, LP_Freq=730e3, n=[12], doSave='', save_Ext='',
          directLoad=True, tLim_plot=[], z_levels=[8,10,14],
          doBode=True, bp_k=None, doPlot=True,
          phase_mode="fft_peak"):
    """
    phase_mode:
        "hilbert"  -> per-channel time series phase via analytic signal (default)
        "fft_peak" -> constant per-channel phase taken at strongest FFT peak
                      within [HP_Freq, LP_Freq]; broadcast over time axis
    """
    # Load in data
    bp_k,data,f_samp,R,Z,phi,inds,time = \
            load_in_data(shotno,directLoad,tLim,HP_Freq,LP_Freq,bp_k)

    R_map=R; Z_map=Z; phi_map=phi

    data,R,Z,phi,mag,mag_Freq,names,fft_freq,fft_out,time = \
        doFilterSignal(time,data,HP_Freq, LP_Freq,R,Z,phi,bp_k.names)

    if doBode:
        for ind, sig in enumerate(data):
            data[ind] = correct_Bode(sig,time,names[ind])[0]

    if phase_mode.lower() == "hilbert":
        phase = np.unwrap(np.angle(hilbert(data,axis=1)), axis=1)*180/np.pi
    elif phase_mode.lower() == "fft_peak":
        # Find peak FFT phase within band for each channel
        band = (fft_freq >= HP_Freq) & (fft_freq <= LP_Freq)
        if not np.any(band):
            while not np.any(band):
                LP_Freq += 1e3  # expand search band
                band = (fft_freq >= HP_Freq) & (fft_freq <= LP_Freq)
            # raise ValueError("No FFT frequencies inside HP/LP window for phase_mode='fft_peak'.")
        band_fft = fft_out[:, band]                 # (Nch, Nband)
        band_mag = np.abs(band_fft)
        peak_idx = np.argmax(np.mean(band_mag, axis=0)) # (Nch,)
        print('Maximum Frequency: ', fft_freq[band][peak_idx]/1e3, 'kHz')     
        peak_phase_deg = np.angle(band_fft[:, peak_idx])*180/np.pi
        peak_phase_deg = peak_phase_deg % 360
        # Broadcast to (Nch, Nt) to keep downstream averaging logic unchanged
        phase = np.tile(peak_phase_deg[:, None], (1, data.shape[1]))
    else:
        raise ValueError(f"Unknown phase_mode '{phase_mode}'. Use 'hilbert' or 'fft_peak'.")

    # Calculate relative toroidal phase angle vs poloidal group of sensors
    angles_group, phase_group, z_inds_out = get_rel_phase(Z,data,phi,phase,z_levels)
    
    if doPlot:
        # Print filter output
        __doPlotFilter(mag,mag_Freq,z_inds_out,angles_group,data,time,Z,fft_freq,fft_out)
        
        # Print map of used sensors
        __Coord_Map(R,Z,phi,R_map,Z_map,phi_map,doSave,shotno,z_levels,z_inds_out,save_Ext)


    
    ## run fit for n
    n_opt, res = optimize_n(angles_group.copy(),phase_group,n,doPlot=doPlot)
    
    if doPlot: doPlot_n(angles_group,phase_group,n,n_opt,data[z_inds_out[0,0]],time,
             doSave,save_Ext,tLim_plot)
    
    return angles_group, phase_group, phi, phase, bp_k, data,inds,time, z_inds_out,n_opt
################################################################################
def load_in_data(shotno,directLoad,tLim,HP_Freq,LP_Freq,bp_k=None):
    if bp_k is None: bp_k = __loadData(shotno,pullData='bp_k',debug=True,
                      data_archive='',forceReload=['bp_k'*False])['bp_k']
    

    # Get data
    data = bp_k.data #TODO: unclear how frequency calibration is or is not being done
    
    f_samp = bp_k.f_samp# digitizier sampling frequency
    
    # select time block, or just use entire signal if its stored in short form
    if directLoad:
        # Check if tLim is a list of time ranges or a single time range
        if isinstance(tLim[0], (list, tuple)):
            # Multiple time ranges
            inds_list = []
            for t_range in tLim:
                t_start, t_end = t_range
                # Get indices for this time range
                i_start = np.argmin(np.abs(bp_k.time - t_start))
                i_end = np.argmin(np.abs(bp_k.time - t_end))
                inds_list.append(np.arange(i_start, i_end + 1))
            # Concatenate all indices
            inds = np.concatenate(inds_list)
        else:
            # Single time range
            if tLim[1] - tLim[0] > 4e-3:
                tLim[0] += 1e-3
                tLim[1] -= 1e-3
            else:
                len_ = tLim[1] - tLim[0]
                tLim[0] += len_ * 0.05
                tLim[1] -= len_ * 0.05
            inds = np.arange(*[np.argmin(np.abs(bp_k.time - t)) for t in tLim])
    else:
        # get Time range
        time_block = bp_k.blockStart/f_samp+bp_k.tLim[0]
        
        ind_1 = np.argmin(np.abs(time_block-tLim[0])) # index within block of start times for blocked data
        ind_1 = np.argmin(np.abs(bp_k.time-time_block[ind_1])) # starting index in terms of time vector
        ind_2 = int(ind_1+(tLim[1]-tLim[0])*f_samp)
        inds = np.arange(ind_1,ind_2,dtype=int)
     
    
    # trim data
    data = data[:,inds]
    
    # Apply windowing at edges of time ranges for multiple ranges
    if directLoad and isinstance(tLim[0], (list, tuple)):
        window_length = int(0.5e-3 * f_samp)  # 0.5 ms taper on each edge
        cumulative_idx = 0
        for i, t_range in enumerate(tLim):
            segment_length = len(inds_list[i])
            # Create Tukey window (cosine taper)
            if segment_length > 2 * window_length:
                window = np.ones(segment_length)
                # Taper at start
                window[:window_length] = 0.5 * (1 - np.cos(np.pi * np.arange(window_length) / window_length))
                # Taper at end
                window[-window_length:] = 0.5 * (1 - np.cos(np.pi * np.arange(window_length, 0, -1) / window_length))
                # Apply window to this segment
                data[:, cumulative_idx:cumulative_idx + segment_length] *= window[np.newaxis, :]
            cumulative_idx += segment_length
    
    phi = np.array(bp_k.Phi)
    
    R = np.array(bp_k.R)
    Z = np.array(bp_k.Z)
    
    return bp_k,data,f_samp,R,Z,phi,inds, bp_k.time[inds]
###############################################################################
def filterAveMag(data,lim=[]):pass
    
def doFilterSignal(time,data,HP_Freq, LP_Freq,R,Z,phi,names,magLim=.5,freqLim=0,
                   hist_ind=3,hist_lim=100):
    
    # Do frequency Filtering 
    # data=__doFilter(data.copy(), time,HP_Freq, LP_Freq)
    
    # Trying with Butterworth filter
    sos = butter(2, [HP_Freq,LP_Freq], 'bandpass', fs=1/(time[1]-time[0]), output='sos')
    data = sosfilt(sos, data,axis=1)

    # Remove spurious frequencies
    data = remove_freq_band(data, time, freq_rm=500e3, freq_sigma=10e3)
    
    # check if signal is finite
    mag = np.mean(np.abs(hilbert(data,axis=1)),axis=1)
 
    mag_inds = np.argwhere(mag<magLim).squeeze()
    
    # data distribution check
    hist = np.array([np.histogram(np.abs(hilbert(d)))[0] for d in data])
    hist_inds = np.argwhere(hist[:,0]/hist[:,hist_ind]>hist_lim)
    
    # Check if signal has reasonable frequeny peak
    fft_freq, fft_out = doFFT(time,data)
    mag_Freq=np.max(np.abs(fft_out),axis=1)/\
           np.mean(np.abs(fft_out),axis=1)
    freq_inds = np.argwhere(mag_Freq<freqLim).squeeze() 
    
#    print(mag_inds,freq_inds)
#    inds = np.([])
#    inds = np.append(inds,mag_inds)
    inds = np.append(mag_inds,freq_inds)
    inds = np.append(inds,hist_inds)
    
    #if HP_Freq<100e3:inds = np.append(inds,[4]) # known bad channel for this shot
    
    inds = np.unique(inds).astype(int)
    # print(inds,mag_inds,freq_inds,hist_inds)
    if np.size(inds)>0:
        data = np.delete(data,inds,axis=0)
        Z = np.delete(Z,inds,axis=0)
        R = np.delete(R,inds,axis=0)
        phi = np.delete(phi,inds,axis=0)
        names = np.delete(names,inds,axis=0)
    
    # check to ensure data and time sizes match
    if data.shape[1]!=len(time):  time = time[:data.shape[1]]

    return data, R, Z, phi,mag,mag_Freq,names,fft_freq,fft_out, time
###############################################################################
def __doPlotFilter(mag,mag_Freq,z_inds_out,angles_group,data,time,Z,fft_freq,fft_out,\
                   freq_lim=[0,800],plotAll=True):
    
    
    title='Filter_Output'
    plt.close(title)
    fig,ax=plt.subplots(2,len(z_inds_out),num=title,tight_layout=True)
    
    title_1 = 'Filter_All'
    plt.close(title_1)
    fig_1,ax_1=plt.subplots(len(z_inds_out),2,num=title_1,tight_layout=True)
    
    for ind,z_inds in enumerate(z_inds_out[:,0]):
        for ind_1,z_ind in enumerate(z_inds):
            ax[0,ind].plot(angles_group[ind][ind_1],mag[z_ind],'*')
            ax[1,ind].plot(angles_group[ind][ind_1],mag_Freq[z_ind],'*')
            
        
            # All signals
            ax_1[ind,0].plot(time,data[z_ind].T)
            ax_1[ind,0].plot(time,np.abs(hilbert(data[z_ind])),
                alpha=.6,c=plt.get_cmap('tab10')(ind_1))
            ax_1[ind,1].plot(fft_freq*1e-3,np.abs(fft_out[z_ind].T))
            ax_1[ind,1].set_xlim(freq_lim)
        for i in range(2):ax[i,ind].grid()
        for i in range(2):ax_1[ind,i].grid()
        if ind > 0: ax_1[ind,1].sharex(ax_1[0,1])
        
        ax_1[ind,0].set_ylabel('Z = %1.1f cm'% (Z[ind]) )
    
    # Clean up
    ax[0,0].set_ylabel('Magnitude')
    ax[1,0].set_ylabel('PSD')
    for i in range(len(z_inds_out)):
        ax[1,i].set_xlabel(r'$\phi$ [deg]')
        ax[0,i].set_title('Z = %1.1f cm'% (z_inds_out[i,1]) )
    
    
    for ax_ in ax_1[:-1,1]: ax_.label_outer()


    ax_1[-1,0].set_xlabel('Time [s]')
    ax_1[-1,1].set_xlabel('Frequency [kHz]')

    plt.show()
#################################################################################
def optimize_n(angles_group,phase_group,n,doPlot=True):
    if len(angles_group) == 0: return np.nan,[]
    # print(angles_group)

    for i in range(len(angles_group)):
        angles_group[i] -=angles_group[i][0]
        angles_group[i][angles_group[i]<0]+=360
    angles = np.concatenate(angles_group).ravel()
    phase = np.concatenate(phase_group).ravel()
    fn = lambda n: np.mean(np.abs(phase-(n[0]*(angles[:]-angles[0]))%360 )**2)
    fn = lambda n: __fn_opt(n,angles,phase)
    #res = minimize(fn,[n[-1]],bounds=[[-15,15]])
    #n_opt = res.x
    x_range = np.arange(-30,31)
    objective = np.argmin([fn([n_]) for n_ in x_range])
    n_opt = x_range[objective]
    res=None
    # distance based optimization
    
    if doPlot:
        plt.close('Optimizer')
        fig,ax=plt.subplots(1,1,num='Optimizer',tight_layout=True,figsize=(3.4,2))
        ax.plot(x_range,[fn([n_]) for n_ in x_range],'-*')
        ax.set_xlabel(r'$n\#$')
        ax.set_ylabel(r'$\phi_{obj}(n)$ [arb]')
        ax.grid()
        plt.show()
    return n_opt, res

def __fn_opt(n,angles,phase):
    # at every angle, calculate the distance to the corresponding predicted angle
    return np.mean(np.sqrt( (np.cos(phase*np.pi/180)-np.cos(n[0]*angles*np.pi/180))**2 +\
                   (np.sin(phase*np.pi/180)-np.sin(n[0]*angles*np.pi/180))**2 ) )
    
#################################################################################
def get_rel_phase(Z,data,angles,phase, z_levels,Z_tol=1):
    angles_group = []
    phase_group=[]
    # normalize off phase by rings in poloidal plane?
    #+/-, in cm
    # SWitch this to theta
    z_inds_out=[]
    for z in z_levels:
        for s in [-1,1]:
            z_inds_ = np.argwhere((Z*1e2<=s*z+Z_tol)  &  (Z*1e2>=s*z-Z_tol)).squeeze()


            if np.size(z_inds_)<=1:continue
            else:z_inds=z_inds_
            z_inds_out.append([z_inds,s*z])
            tmp=angles[z_inds]#-angles[z_inds][0]
            tmp[tmp<0]+=360
            angles_group.append(tmp)
            tmp = [np.mean(phase[z_] - phase[z_inds[0]])%360 for z_ in z_inds]
            phase_group.append(tmp)
    
    z_inds_out = np.array(z_inds_out,dtype=object)
    return angles_group, phase_group, z_inds_out
#################################################################################
def get_theta(R,Z,shotno,tLim):
        # Assume using the upper row always
        try:
            try:eq = CModEFITTree(shotno)
            except:eq = CModEFITTree(1160930034)
            time = eq.getTimeBase()
            tInd = np.arange(*[np.argmin((time-t)**2) for t in tLim ] ) 
            # if tLim points are closer than dt-EFIT, it won't work
            if np.size(tInd)==0:tInd = np.argmin((time-tLim[0])**2) 
            zmagx = np.mean(eq.getMagZ()[tInd])
            rmagx = np.mean(eq.getMagR()[tInd])
        except: zmagx=0;rmagx=0.79
        
        #sensor = __loadData(shotno,pullData=['bp_t'])['bp_t']
        theta = np.arctan2(Z-zmagx,R-rmagx)[0]*180/np.pi
        
        return theta
   
################################################################################
def doPlot_n(angle,phases,n,n_opt,data,time,doSave,save_Ext,tLim_plot):
    # Figure
    plt.close("Linear Phase Comparison%s"%save_Ext)
    fig,ax = plt.subplots(2,1,tight_layout=True,num='Linear Phase Comparison%s'%save_Ext)
    #angle=np.insert(angle,3,np.nan)
    #phases=np.insert(phases,2,np.nan)
    for ind,a in enumerate(angle):ax[0].plot(a,phases[ind],'*',label='Signal Phase'*(ind==0))
    angle = np.concatenate(angle).ravel()
    phases = np.concatenate(phases).ravel()
    x_ = np.linspace(0,np.min(angle),200) if np.min(angle) < -100 else np.linspace(0,np.max(angle),200)
    for n_ in n: ax[0].plot(x_,(n_*x_+n_*angle[0])%360,'--',label=r'$n$=%d'%n_)
    for n_ in [n_opt]: ax[0].plot(x_,(n_*x_ +n_*angle[0] )%360,'--',label=r'$\hat{n}$=%2.2f'%n_)
    ax[0].set_xlabel(r'Geometric Phase $\Delta\phi$ [deg]')
    ax[0].set_ylabel(r'Signal Phase $\bar{\Delta\phi}$ [deg]')
    ax[0].legend(fontsize=8,handlelength=1)
    
    #ax[1].plot(X[0],Z[0].T) 
    ax[1].plot(time,data.T)
    ax[1].legend([r'$\phi=%1.1f^\circ$'%a for a in angle],fontsize=8,handlelength=1,ncols=2)
    
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('B [T/s]')
    for i in range(2):ax[i].grid()
    if tLim_plot:ax[1].set_xlim(*tLim_plot)
    plt.show()
    
    if doSave: fig.savefig(doSave+'Linear_Phase_Comparison%s.pdf'%save_Ext,transparent=True)
################################################################################
def __Coord_Map(R,Z,phi,R_map,Z_map,phi_map,doSave,shotno,z_levels,z_inds_out,save_Ext):
    plt.close('Z-Phi_Map_%d%s'%(shotno,save_Ext))
    fig,ax=plt.subplots(1,1,num='Z-Phi_Map_%d%s'%(shotno,save_Ext),tight_layout=True,figsize=(3.4,2))
    fn_norm = lambda r: (r-np.min(R))/(np.max(R)-np.min(R)) *.5 +.5
    if np.mean(phi) < 0: phi += 360; phi_map += 360
    
    ax.plot(phi_map,Z_map*1e2,'k*',label='Unused')
    z_inds_out=np.array(z_inds_out,dtype=object)
    for ind,z_inds in enumerate(z_inds_out[:,0]):
        flag=True
        for z_ind in z_inds:
            plt.plot(phi[z_ind], Z[z_ind]*1e2,'*', c=plt.get_cmap('tab10')(ind),\
                 alpha=fn_norm(R[z_ind]), label=('%1.1f cm'%z_inds_out[ind,1])*flag,ms=5)
            flag=False
    ax.set_xlabel(r'$\phi$ [deg]')
    ax.set_ylabel(r'Z [cm]')
    ax.legend(fontsize=9,handlelength=1.5,title_fontsize=8,ncols=3,)
     #         title=r'R$in${%1.1f-%1.1f}'%(np.min(R),np.max(R)))
    ax.grid()
    
    if doSave:
        fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.png',transparent=True)
    plt.show() 
########################################################
if __name__ == '__main__': 
    shotno = 1160930034
    tLim = [[1.1,1.11],[1.128,1.141], [1.164,1.175], [1.191,1.205], [1.223,1.237]]
    
    HP_Freq=725e3; LP_Freq=750e3
    n = [12]   
    z_levels = [8,10,14]
    
    doSave='../output_plots/'
    save_Ext='_Test_n12_Calibrated'
    run_n(shotno=shotno, tLim=tLim, HP_Freq=HP_Freq, LP_Freq=LP_Freq,
          n=n, z_levels=z_levels,
          doSave=doSave, save_Ext=save_Ext,
          doPlot=True, doBode=True)
    
    print('Done')