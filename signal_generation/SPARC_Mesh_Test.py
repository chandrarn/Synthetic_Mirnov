#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:48:10 2024

@author: rian
"""

"""
Created on Wed Dec  4 16:45:23 2024
 Example ThinCurr
@author: rian
"""

import struct
import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista
from prep_sensors import conv_sensor
from gen_MAGX_Coords import gen_Sensors, gen_Sensors_Updated
import Synthetic_Mirnov as sM
import geqdsk_filament_generator as gF

plt.rcParams['figure.figsize']=(6,6)
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.labelweight']='bold'
plt.rcParams['lines.linewidth']=2
plt.rcParams['lines.markeredgewidth']=2
#%matplotlib inline
#%config InlineBackend.figure_format = "retina"

from OpenFUSIONToolkit.ThinCurr import ThinCurr
from OpenFUSIONToolkit.ThinCurr.meshing import write_ThinCurr_mesh, build_torus_bnorm_grid, build_periodic_mesh
from OpenFUSIONToolkit.util import build_XDMF



'''
Set up model object, with mesh and set of coil positions. Note that the mesh
information isn't accessable locally in the object (?) only in the hd5 file (?)

'''
# Gen Currents
params={'m':3,'n':1,'r':.25,'R':1,'n_pts':70,'T':2e-3,\
        'm_pts':10,'f':1e3,'dt':1e-4,'periods':3,'n_threads':64,'I':10}
theta,phi = gF.gen_filament_coords(params)
#filament_coords = gF.calc_filament_coords_geqdsk('input_data/geqdsk', theta, phi, params)
filament_coords = gF.calc_filament_coords_geqdsk('g1051202011.1000', theta, phi, params)
coil_currs = sM.gen_coil_currs(params)
sM.gen_filaments('oft_in.xml',params,filament_coords,eta='1E-5, 1E-5 , 1E-5 , 1E-5, 1E-5, 1E-5, 1E-5, 1E-5' )

tw_plate = ThinCurr(nthreads=8)
# Mesh file contains dict name 'mesh', and sub attributes LC (mesh cells, n_cells),
# R (npts x 2) (Resistances? matches number of points), REG  [npts x 1] (? just ones?)
# .xml file defines coils 
# 'SPARC_Sept2023_noPR.h5'
#tw_plate.setup_model(mesh_file='input_data/SPARC_Sept2023_noPR.h5',xml_filename='input_data/Soft_in.xml')
#tw_plate.setup_model(mesh_file='vacuum_mesh.h5',xml_filename='oft_in.xml')
#tw_plate.setup_model(mesh_file='input_data/thincurr_ex-plate.h5',xml_filename='input_data/oft_in.xml')

# tw_plate.setup_model(mesh_file='input_data/C_Mod_ThinCurr_Limiters-homology.h5',xml_filename='input_data/oft_in.xml')
tw_plate.setup_model(mesh_file='input_data/C_Mod_ThinCurr_Limiters_Combined-homology.h5',xml_filename='input_data/oft_in.xml')
# tw_plate.setup_model(mesh_file='input_data/C_Mod_ThinCurr_VV-homology.h5',xml_filename='input_data/oft_in.xml')
tw_plate.setup_io()

# Coupling for plot
Mc = tw_plate.compute_Mcoil()
tw_plate.compute_Lmat()#(use_hodlr=True,cache_file='input_data/HOLDR_L_%s.save'%'Mesh_Test_VV')
tw_plate.compute_Rmat()

print("Building XMDF")
tw_plate.build_XDMF()
print('Built')
print(tw_plate.Lmat.shape)
with h5py.File('mesh.0001.h5','r') as h5_file:
    r = np.asarray(h5_file['R_surf']) # x,y,z coords of surface
    lc = np.asarray(h5_file['LC_surf']) # This is the mesh itself ["mesh triangles"]

    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
cells = np.insert(lc, [0,], 3, axis=1) # appends the value '3' in first coll of the cells
grid = pyvista.UnstructuredGrid(cells, celltypes, r) # Why is r necessary for the grid?

# Gen sensors
#sensors = conv_sensor('sensorLoc.xyz')[0]
sensors = gen_Sensors_Updated(select_sensor='C_MOD_MIRNOV_T')
# Msensor, Msc, sensor_obj = tw_plate.compute_Msensor('input_data/floops_BP_CFS.loc')


# Color grid cells
shading = np.dot(np.linalg.pinv(tw_plate.Lmat),np.dot(Mc.T,np.ones((10,)) ) )


# Launch Plotter
p = pyvista.Plotter()
#p2=pyvista.Plotter()

p.add_mesh(grid, color="white", opacity=.6, show_edges=True,label='Mesh')
#p.add_mesh(grid, scalars=shading, opacity=.6, show_edges=True,label='Mesh')
p.show_bounds()

slice_coords=[np.linspace(0,3,10),[0]*10,np.linspace(-3.5,3.5,10)]
slice_line = pyvista.Spline(np.c_[slice_coords].T,10)
slices = grid.slice_along_line(slice_line)

#p.add_mesh(slice_line,line_width=5)
#p2.add_mesh(slices,line_width=5)


# Plot Sensors
for ind,s in enumerate(sensors):
    p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                 render_points_as_spheres=True,
                 label='Sensor' if ind==0 else None)
   
# Plot Filaments
t_pt=0
for ind,filament in enumerate(filament_coords):
    pts = np.array(filament).T
    #print(np.array(calc_filament_coord(m,n,r,R,theta,phi)).shape)
    pts=np.array(pts)
    #print(pts.shape,theta)
    spl=pyvista.Spline(pts,len(pts))
    #p.add_(spline,render_lines_as_tubes=True,line_width=5,show_scalar_bar=False)
    #p.add_mesh(spl,opacity=1,line_width=6,color=plt.get_cmap('viridis')(theta*m/(2*np.pi)))
    slices_spl=spl.slice_along_line(slice_line)#spl.slice_orthogonal()
    color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2)
    p.add_mesh(spl,color=color,
               line_width=10,render_points_as_spheres=True,
               label='Filament' if ind==0 else None)
    #p.add_points(pts,render_points_as_spheres=True,opaity=1,point_size=20,color=colors[ind])
    #tmp.append(pts)
    
    # # Place on frame outline
    # p2.add_points(pts[0],render_points_as_spheres=True,point_size=10,
    #               color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2))
    # half_way = np.argmin((phi-np.pi)**2)
    # p2.add_points(pts[half_way],render_points_as_spheres=True,point_size=10,
    #               color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2))
    
p.add_legend()



p.save_graphic('../output_plots/SPARC_Cad_Sensors_Mirnov_12-10.pdf')
# p2.save_graphic('../output_plots/SPARC_Cad_Sensors_Frame_Mirnov_12-10.pdf')
p.show()
# p2.show()

