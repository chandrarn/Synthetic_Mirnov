#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:06:49 2025

@author: rianc
"""

from header_signal_analysis import loadmat, np, plt

def correct_Bode(signal,time,sensor_name):
    '''
    Bode correction
    Based on calibrate_xpose.m by Jason Sears
    '''
    
    rfft = np.fft.rfft(signal)
    fftFreq = np.fft.rfftfreq(len(signal), time[1]-time[0])
    
    #f,H_spline,fine_cal,fine_caln = __cal_Correction(sensor_name,fftFreq)
    #return  f,H_spline,fine_cal,fine_caln
    f, H = __cal_Correction(sensor_name,fftFreq)

    sig_new = np.fft.irfft( rfft / H ) 
    
    return sig_new,  H, f, fftFreq,rfft
   
def __cal_Correction(sensor_name,freq):
    CAL_PATH = '/mnt/home/sears/Matlab/Calibration/cal_v2/'
    mat = loadmat(CAL_PATH+'451_responses/'+sensor_name +'_cal.mat')
    f = mat['f'][0]; H_spline = mat['H_spline'][0]
    mat = loadmat(CAL_PATH+'fine_tuning/'+sensor_name +'_cal+.mat')
    fine_cal= mat['fine_cal'][0,0]
    mat = loadmat(CAL_PATH+'fine_tuning/'+sensor_name +'_caln.mat')
    fine_caln = mat['fine_caln'][0,0]
    
    #return f,H_spline,fine_cal,fine_caln
    
    return  f, np.interp(freq,f,H_spline) * fine_cal * fine_caln
    