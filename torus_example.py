import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista
pyvista.set_jupyter_backend('static') # Comment to enable interactive PyVista plots
plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2

import sys;sys.path.append('/home/rianc/OpenFUSIONToolkit/build_release/python/')
from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.sensor import Mirnov, save_sensors
from OpenFUSIONToolkit.util import build_XDMF
from OpenFUSIONToolkit.io import histfile

tw_torus = ThinCurr(nthreads=4)
tw_torus.setup_model(mesh_file='thincurr_ex-torus.h5',xml_filename='oft_in.xml')
tw_torus.setup_io()

sensors = [
    Mirnov([1.45,0.0,0.0], [0.0,0.0,1.0], 'Bz_inner'),
    Mirnov([1.55,0.0,0.0], [0.0,0.0,1.0], 'Bz_outer'),
    Mirnov([1.45,0.0,0.0], [1.0,0.0,0.0], 'Br_inner'),
    Mirnov([1.55,0.0,0.0], [1.0,0.0,0.0], 'Br_outer'),
]
save_sensors(sensors)
Msensor, Msc, sensor_obj = tw_torus.compute_Msensor('floops.loc')

Mc = tw_torus.compute_Mcoil()
tw_torus.compute_Lmat()
tw_torus.compute_Rmat()

tw_mode = ThinCurr(nthreads=4)
tw_mode.setup_model(mesh_file='thincurr_mode.h5')
with h5py.File('thincurr_mode.h5', 'r+') as h5_file:
    mode_drive = np.asarray(h5_file['thincurr/driver'])

Msensor_plasma, _, _ = tw_mode.compute_Msensor('floops.loc')
# Significance? 
mode_driver = tw_mode.cross_eval(tw_torus,mode_drive)
sensor_mode = np.dot(mode_drive,Msensor_plasma)

mode_freq = 1.E3
mode_growth = 2.E3
dt = (1.0/1.E3)/50.0
nsteps = 200
timebase_current = np.arange(0.0,dt*nsteps+1,dt/4.0); timebase_voltage = (timebase_current[1:]+timebase_current[:-1])/2.0
cos_current = timebase_current/mode_growth*np.cos(mode_freq*2.0*np.pi*timebase_current); cos_voltage = np.diff(cos_current)/np.diff(timebase_current)
sin_current = timebase_current/mode_growth*np.sin(mode_freq*2.0*np.pi*timebase_current); sin_voltage = np.diff(sin_current)/np.diff(timebase_current)
volt_full = np.zeros((nsteps+2,tw_torus.nelems+1))
sensor_signals = np.zeros((nsteps+2,sensor_mode.shape[1]+1))
# Unclear 
for i in range(nsteps+2):
    volt_full[i,0] = dt*i
    sensor_signals[i,0] = dt*i
    if i > 0:
        volt_full[i,1:] = mode_driver[0,:]*np.interp(volt_full[i,0],timebase_voltage,cos_voltage) \
          + mode_driver[1,:]*np.interp(volt_full[i,0],timebase_voltage,sin_voltage)
          
        sensor_signals[i,1:] = sensor_mode[0,:]*np.interp(volt_full[i,0],timebase_current,cos_current) \
          + sensor_mode[1,:]*np.interp(volt_full[i,0],timebase_current,sin_current)
          
tw_torus.run_td(dt,nsteps,status_freq=10,full_volts=volt_full,sensor_obj=sensor_obj,direct=True,sensor_values=sensor_signals)

hist_file = histfile('floops.hist')
print(hist_file)

fig, ax = plt.subplots(1,1)
ax.plot(hist_file['time'],hist_file['Bz_inner'],color='tab:blue')
ax.plot(hist_file['time'],hist_file['Bz_outer'],color='tab:red')
ax.plot(hist_file['time'],sensor_signals[:-1,1],'--',color='tab:blue')
ax.plot(hist_file['time'],sensor_signals[:-1,2],'--',color='tab:red')
_ = ax.set_xlim(left=0.0)

fig, ax = plt.subplots(1,1)
ax.plot(hist_file['time'],hist_file['Br_inner'],color='tab:blue')
ax.plot(hist_file['time'],hist_file['Br_outer'],color='tab:red')
ax.plot(hist_file['time'],sensor_signals[:-1,3],'--',color='tab:blue')
ax.plot(hist_file['time'],sensor_signals[:-1,4],'--',color='tab:red')
_ = ax.set_xlim(left=0.0)
