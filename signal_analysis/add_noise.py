#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:53:37 2025

@author: rianc
"""
from header_signal_analysis import np

def __noise_component(signals):
    sigma=2.5;A=25000*6.5
    fn_norm = lambda x: A*np.exp(-x**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)
    x_=np.linspace(-1,1,100)
    return 5*(np.max(signals)/15)*np.random.choice(x_,len(signals),True,fn_norm(x_)/np.sum(fn_norm(x_)))