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


import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize
import matplotlib.pyplot as plt
try:from header_signal_analysis import CModEFITTree, doFFT
except:pass
from sys import path; path.append('../C-Mod/')
from header_Cmod import __doFilter, correct_Bode
from get_Cmod_Data import __loadData


# Assume that we have some way of grouping in frequency/time
def run_n(shotno=1160930034, tLim=[0.82,0.82008], fLim=None, 
          HP_Freq=400e3, LP_Freq=600e3,n=[2], doSave='',save_Ext='',
          directLoad=True,tLim_plot=[],z_levels = [8,10,14], doBode=True,
          bp_k=None,doPlot=True):
    
    # Load in data
    bp_k,data,f_samp,R,Z,phi,inds,time = \
            load_in_data(shotno,directLoad,tLim,HP_Freq,LP_Freq,bp_k)

    R_map=R;Z_map=Z;phi_map =phi# Save for later plotting
    # FFT processing: more than just Hilbert
    # Use dominant frequency within bandpassfilteresed region for now
    #return data,time,HP_Freq,LP_Freq,R,Z,phi,bp_k
    data,R,Z,phi,mag,mag_Freq,names,fft_freq,fft_out = \
        doFilterSignal(time,data,HP_Freq, LP_Freq,R,Z,phi,bp_k.names)
    
    #return bp_k, data, f_samp,R,Z,phi,mag,mag_Freq
    # Bode filtering
    if doBode:
        for ind, sig in enumerate(data):
            data[ind]=correct_Bode(sig,time,names[ind])[0]
    
    phase = np.unwrap(np.angle(hilbert(data,axis=1)))*180/np.pi # all phases, unfilterd
    
    #return data,phase,phi,bp_k,inds
    
    # Calculate relative toroidal phase angle vs poloidal group of sensors
    angles_group, phase_group, z_inds_out = get_rel_phase(Z,data,phi,phase,z_levels)

    if doPlot:
        # Print filter output
        __doPlotFilter(mag,mag_Freq,z_inds_out,angles_group,data,time,Z,fft_freq,fft_out,)
        
        # Print map of used sensors
        __Coord_Map(R,Z,phi,R_map,Z_map,phi_map,doSave,shotno,z_levels,z_inds_out,save_Ext)


    
    ## run fit for n
    n_opt, res = optimize_n(angles_group,phase_group,n)
    
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
        inds = np.arange(len(bp_k.time))
        inds = np.arange(*[np.argmin(np.abs(bp_k.time-t)) for t in tLim])
    else:
        # get Time range
        time_block = bp_k.blockStart/f_samp+bp_k.tLim[0]
        
        ind_1 = np.argmin(np.abs(time_block-tLim[0])) # index within block of start times for blocked data
        ind_1 = np.argmin(np.abs(bp_k.time-time_block[ind_1])) # starting index in terms of time vector
        ind_2 = int(ind_1+(tLim[1]-tLim[0])*f_samp)
        inds = np.arange(ind_1,ind_2,dtype=int)
        print(ind_1,ind_2)
        
    #return R,Z,data, inds,bp_k
    
    #return bp_k
    # trim data
    data = data[:,inds]
    
    print(bp_k.Phi)
    phi = np.array(bp_k.Phi)
    
    R = np.array(bp_k.R)
    Z = np.array(bp_k.Z)
    
    #theta = get_theta(R,Z,shotno,tLim)
    
    return bp_k,data,f_samp,R,Z,phi,inds, bp_k.time[inds]

def doFilterSignal(time,data,HP_Freq, LP_Freq,R,Z,phi,names,magLim=1,freqLim=10):
    
    print(data.shape)
    # Do frequency Filtering 
    data=__doFilter(data.copy(), time,HP_Freq, LP_Freq)
    
    # check if signal is finite
    mag = np.mean(np.abs(hilbert(data,axis=1)),axis=0)
    mag_inds = np.argwhere(mag<magLim).squeeze()
    
    # Check if signal has reasonable frequeny peak
    fft_freq, fft_out = doFFT(time,data)
    mag_Freq=np.max(np.abs(fft_out),axis=1)/\
           np.mean(np.abs(fft_out),axis=1)
    freq_inds = np.argwhere(mag_Freq<freqLim).squeeze() 
    
    inds = np.append(mag_inds,freq_inds)
    inds = np.unique(inds)
    
#    return data,inds
    data = np.delete(data,inds,axis=0)
    Z = np.delete(Z,inds,axis=0)
    R = np.delete(R,inds,axis=0)
    phi = np.delete(phi,inds,axis=0)
    names = np.delete(names,inds,axis=0)
    
    return data, R, Z, phi,mag,mag_Freq,names,fft_freq,fft_out
###############################################################################
def __doPlotFilter(mag,mag_Freq,z_inds_out,angles_group,data,time,Z,fft_freq,fft_out,plotAll=True):
    
    
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
            ax_1[ind,1].plot(fft_freq,np.abs(fft_out[z_ind].T))
        for i in range(2):ax[ind,i].grid()
        for i in range(2):ax_1[ind,i].grid()
    ax[0,0].set_ylabel('Magnitude')
    ax[1,0].set_ylabel('PSD')
    
    plt.show()
#################################################################################
def optimize_n(angles_group,phase_group,n,doPlot=True):
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
def get_rel_phase(Z,data,angles,phase, z_levels):
    angles_group = []
    phase_group=[]
    # normalize off phase by rings in poloidal plane?
    #+/-, in cm
    # SWitch this to theta
    z_inds_out=[]
    for z in z_levels:
        for s in [-1,1]:
            print(s*z)
            z_inds_ = np.argwhere((Z*1e2<=s*z+0.5)  &  (Z*1e2>=s*z-0.5)).squeeze()

            print(z_inds_)
            if np.size(z_inds_)<=1:continue
            else:z_inds=z_inds_
            z_inds_out.append([z_inds,s*z])
            tmp=angles[z_inds]-angles[z_inds][0]
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
    ax.legend(fontsize=9,handlelength=1.5,title_fontsize=8,)
     #         title=r'R$in${%1.1f-%1.1f}'%(np.min(R),np.max(R)))
    ax.grid()
    
    if doSave:
        fig.savefig(doSave+fig.canvas.manager.get_window_title()+'.pdf',transparent=True)
    plt.show() 
########################################################
if __name__ == '__main__': run_n()