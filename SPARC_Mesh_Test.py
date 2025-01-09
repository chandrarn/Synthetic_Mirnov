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
tw_plate = ThinCurr(nthreads=4)
# Mesh file contains dict name 'mesh', and sub attributes LC (mesh cells, n_cells),
# R (npts x 2) (Resistances? matches number of points), REG  [npts x 1] (? just ones?)
# .xml file defines coils 
tw_plate.setup_model(mesh_file='SPARC_Sept2023_noPR.h5',xml_filename='oft_in.xml')
tw_plate.setup_io()

print("Building XMDF")
tw_plate.build_XDMF()
print('Built')

with h5py.File('mesh.0001.h5','r') as h5_file:
    r = np.asarray(h5_file['R_surf']) # x,y,z coords of surface
    lc = np.asarray(h5_file['LC_surf']) # This is the mesh itself ["mesh triangles"]
    
    celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
cells = np.insert(lc, [0,], 3, axis=1) # appends the value '3' in first coll of the cells
grid = pyvista.UnstructuredGrid(cells, celltypes, r) # Why is r necessary for the grid?

# Gen sensors
sensors = conv_sensor('sensorLoc.xyz')[0]

p = pyvista.Plotter()
p2=pyvista.Plotter()

p.add_mesh(grid, color="white", opacity=.2, show_edges=True,label='Mesh')
p.show_bounds()

slice_coords=[np.linspace(0,3,10),[0]*10,np.linspace(-3.5,3.5,10)]
slice_line = pyvista.Spline(np.c_[slice_coords].T,10)
slices = grid.slice_along_line(slice_line)
#p.add_mesh(slice_line,line_width=5)



# Plot Sensors
for ind,s in enumerate(sensors):
    p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                 render_points_as_spheres=True,
                 label='Sensor' if ind==0 else None)
    p2.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
                 render_points_as_spheres=True,
                 label='Sensor' if ind==0 else None)
p.add_legend()


p2.add_mesh(slices,line_width=5)
p.save_graphic('SPARC_Cad_Sensors.pdf')
p2.save_graphic('SPARC_Cad_Sensors_Frame.pdf')
p.show()
p2.show()

