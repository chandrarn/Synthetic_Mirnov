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

import sys

# import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyvista
from gen_MAGX_Coords import gen_Sensors_Updated, get_sensor_category

sys.path.append("/home/rianc/Documents/disruption-py/")

# from disruption_py.machine.d3d.mirnov import (
#     _BP_SENSOR_CATEGORIES,
# )

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markeredgewidth"] = 2
# %matplotlib inline
# %config InlineBackend.figure_format = "retina"

from OpenFUSIONToolkit.ThinCurr import ThinCurr

# from OpenFUSIONToolkit.util import build_XDMF
from OpenFUSIONToolkit import OFT_env


"""
Set up model object, with mesh and set of coil positions. Note that the mesh
information isn't accessable locally in the object (?) only in the hd5 file (?)

"""
# Gen Currents
params = {
    "m": 3,
    "n": 1,
    "r": 0.25,
    "R": 1,
    "n_pts": 70,
    "T": 2e-3,
    "m_pts": 10,
    "f": 1e3,
    "dt": 1e-4,
    "periods": 3,
    "n_threads": 64,
    "I": 10,
}

# theta,phi = gF.gen_filament_coords(params)
# #filament_coords = gF.calc_filament_coords_geqdsk('input_data/geqdsk', theta, phi, params)
# filament_coords = gF.calc_filament_coords_geqdsk('g1051202011.1000', theta, phi, params)
# coil_currs = sM.gen_coil_currs(params)

# sM.gen_filaments('oft_in.xml',params,filament_coords,eta='1E-5, 1E-5 , 1E-5 , 1E-5, 1E-5, 1E-5, 1E-5, 1E-5' )

oft_env = OFT_env(nthreads=20)
tw_plate = ThinCurr(oft_env)
# Mesh file contains dict name 'mesh', and sub attributes LC (mesh cells, n_cells),
# R (npts x 2) (Resistances? matches number of points), REG  [npts x 1] (? just ones?)
# .xml file defines coils
# 'SPARC_Sept2023_noPR.h5'
# tw_plate.setup_model(mesh_file='input_data/SPARC_Sept2023_noPR.h5',xml_filename='input_data/Soft_in.xml')
tw_plate.setup_model(mesh_file='input_data/SPARC_vv_prtmrv_noext.h5',xml_filename='input_data/oft_in.xml')
# tw_plate.setup_model(mesh_file='vacuum_mesh.h5',xml_filename='oft_in.xml')
# tw_plate.setup_model(mesh_file='input_data/thincurr_ex-plate.h5',xml_filename='input_data/oft_in.xml')

# tw_plate.setup_model(mesh_file='input_data/C_Mod_ThinCurr_Limiters-homology.h5',xml_filename='input_data/oft_in.xml')
# tw_plate.setup_model(mesh_file='input_data/C_Mod_ThinCurr_Limiters_Combined-homology.h5',xml_filename='input_data/oft_in.xml')
# tw_plate.setup_model(mesh_file='input_data/ThinCurr_DIIID-1_23_26-homology.h5',xml_filename='input_data/diiid_coils.xml')
# tw_plate.setup_model(mesh_file='input_data/ThinCurr_DIIID-1_23_26-homology.h5',xml_filename='input_data/diiid_coils.xml')
# tw_plate.setup_model(
#     mesh_file="input_data/TCV-homology.h5", xml_filename="input_data/oft_in.xml"
# )

tw_plate.setup_io()

# Coupling for plot
# Mc = tw_plate.compute_Mcoil()
# tw_plate.compute_Lmat()#(use_hodlr=True,cache_file='input_data/HOLDR_L_%s.save'%'Mesh_Test_VV')
# tw_plate.compute_Rmat()

print("Building XMDF")
plot_data = tw_plate.build_XDMF()
grid = plot_data["ThinCurr"]["smesh"].get_pyvista_grid()
print("Built Pyvista grid from ThinCurr mesh")

# print(tw_plate.Lmat.shape)
# with h5py.File('mesh.0001.h5','r') as h5_file:
#     r = np.asarray(h5_file['R_surf']) # x,y,z coords of surface
#     lc = np.asarray(h5_file['LC_surf']) # This is the mesh itself ["mesh triangles"]

#     celltypes = np.array([pyvista.CellType.TRIANGLE for _ in range(lc.shape[0])], dtype=np.int8)
# cells = np.insert(lc, [0,], 3, axis=1) # appends the value '3' in first coll of the cells
# grid = pyvista.UnstructuredGrid(cells, celltypes, r) # Why is r necessary for the grid?

# Gen sensors
# sensors = conv_sensor('sensorLoc.xyz')[0]
# sensors = gen_Sensors_Updated(select_sensor='DIII_D_BP')
# sensor_category_list = [category_name for category_name, _ in _BP_SENSOR_CATEGORIES]
select_sensor="SPARC_MRNV"

save_ext = '_V_only'
sensors = gen_Sensors_Updated(select_sensor=select_sensor)
sensor_category_list = get_sensor_category("SPARC", get_categories=True)

# Msensor, Msc, sensor_obj = tw_plate.compute_Msensor('input_data/floops_BP_CFS.loc')
# select_sensor_names = ['BP-DT23-290U', 'BP-IOL1-110U', 'BP-DVT1-010L', 'BP-IOL2-190L']
# select_sensor_names.extend(['MRNV_160M_H%d'%i for i in range(1,6)])
# select_sensor_names.extend(['MRNV_160M_V%d'%i for i in range(1,6)])
# select_sensor_names.extend(['MRNV_340M_H%d'%i for i in range(1,6)])
# select_sensor_names.extend(['MRNV_340M_V%d'%i for i in range(1,6)])
select_sensor_names =  [ 
                    # 'MRNV_160M_H1', 'MRNV_160M_H2', 'MRNV_160M_H3', 'MRNV_160M_H4', 'MRNV_160M_H5',\
                    'MRNV_160M_V1', 'MRNV_160M_V2', 'MRNV_160M_V3', 'MRNV_160M_V4', 'MRNV_160M_V5',\
                    # 'MRNV_340M_H1', 'MRNV_340M_H2', 'MRNV_340M_H3', 'MRNV_340M_H4',  'MRNV_340M_H5',\
                    'MRNV_340M_V1', 'MRNV_340M_V2', 'MRNV_340M_V3', 'MRNV_340M_V4', 'MRNV_340M_V5',
                    ]

sensors = [s for s in sensors if s._name in select_sensor_names]

# drop_sensors = ['MRNV_160M_H4', 'MRNV_160M_V3', 'MRNV_340M_H2', 'MRNV_340M_V5']

# sensors = [s for s in sensors if s._name not in drop_sensors]
# print(sensors)
# Color grid cells
# shading = np.dot(np.linalg.pinv(tw_plate.Lmat),np.dot(Mc.T,np.ones((10,)) ) )


# Launch Plotter
p = pyvista.Plotter(off_screen=True)
print("Launched Plotter")

# p2=pyvista.Plotter()

p.add_mesh(grid, color="white", opacity=0.3, show_edges=True, label="Mesh")
# p.add_mesh(grid, scalars=shading, opacity=.6, show_edges=True,label='Mesh')


p.camera.focal_point = (0, 0, 0)
p.camera.zoom(400)

p.show_bounds(
    bounds=(-5, 5, -5, 5, -5, 5),
    n_xlabels=20,
    n_ylabels=20,
    n_zlabels=20,
    location="outer",
    all_edges=True,
)

# p.save_graphic('../output_plots/SPARC_Cad_Mesh.png')
# print('Saved mesh to : ../output_plots/SPARC_Cad_Mesh.png')


# slice_coords=[np.linspace(0,3,10),[0]*10,np.linspace(-3.5,3.5,10)]
# slice_line = pyvista.Spline(np.c_[slice_coords].T,10)
# slices = grid.slice_along_line(slice_line)

# p.add_mesh(slice_line,line_width=5)
# p2.add_mesh(slices,line_width=5)


# Plot Sensors
cmap_name = "tab20"
used_categories = [] # track which sensor categories are used, to only add those to the legend
for ind, s in enumerate(sensors):
    # p.add_points(np.mean(s._pts,axis=0),color='k',point_size=10,
    #              render_points_as_spheres=True,
    #              label='Sensor' if ind==0 else None)
    # category_name = get_bp_sensor_category(s._name)
    category_name = get_sensor_category("SPARC", s._name)
    category_order = np.argwhere(np.array(sensor_category_list) == category_name)[0][0]
    if category_name not in used_categories:
        used_categories.append(category_name)
    colors = plt.get_cmap(cmap_name)
    color = colors(category_order / len(sensor_category_list))
    # color='k'
    p.add_points(
        np.mean(s._pts, axis=0),
        color=color,
        point_size=20,
        render_points_as_spheres=True,
        cmap=cmap_name,
    )


temp_pts = []
for ind, name in enumerate(sensor_category_list):
    if name not in used_categories:
        continue
    temp_pts.append(
        p.add_points(
            np.array([0, 0, 0]),
            color=plt.get_cmap(cmap_name)(ind / len(sensor_category_list)),
            point_size=0,
            render_points_as_spheres=True,
            label=name,
        )
    )
p.add_legend()

for tmp in temp_pts:
    tmp.visibility = False

# p.show()
# Plot Filaments
t_pt = 0
filament_coords = []
for ind, filament in enumerate(filament_coords):
    pts = np.array(filament).T
    # print(np.array(calc_filament_coord(m,n,r,R,theta,phi)).shape)
    pts = np.array(pts)
    # print(pts.shape,theta)
    spl = pyvista.Spline(pts, len(pts))
    # p.add_(spline,render_lines_as_tubes=True,line_width=5,show_scalar_bar=False)
    # p.add_mesh(spl,opacity=1,line_width=6,color=plt.get_cmap('viridis')(theta*m/(2*np.pi)))
    slices_spl = spl.slice_along_line(slice_line)  # spl.slice_orthogonal()
    color = plt.get_cmap("plasma")((coil_currs[t_pt, ind + 1] / params["I"] + 1) / 2)
    p.add_mesh(
        spl,
        color=color,
        line_width=10,
        render_points_as_spheres=True,
        label="Filament" if ind == 0 else None,
    )
    # p.add_points(pts,render_points_as_spheres=True,opaity=1,point_size=20,color=colors[ind])
    # tmp.append(pts)

    # # Place on frame outline
    # p2.add_points(pts[0],render_points_as_spheres=True,point_size=10,
    #               color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2))
    # half_way = np.argmin((phi-np.pi)**2)
    # p2.add_points(pts[half_way],render_points_as_spheres=True,point_size=10,
    #               color=plt.get_cmap('plasma')((coil_currs[t_pt,ind+1]/params['I']+1)/2))

p.add_legend()


# p.save_graphic("../output_plots/SPARC_Cad_Sensors_Mirnov_12-10.pdf")
# p2.save_graphic('../output_plots/SPARC_Cad_Sensors_Frame_Mirnov_12-10.pdf')
p.screenshot(f"../output_plots/SPARC_Cad_Sensors_Mirnov_{select_sensor}{save_ext}.png")
# p.save_graphic(f"../output_plots/SPARC_Cad_Sensors_Mirnov_{select_sensor}{save_ext}.png")
print("Saved: ", f"../output_plots/SPARC_Cad_Sensors_Mirnov_{select_sensor}{save_ext}.png")
# p.show()
# p2.show()
    
