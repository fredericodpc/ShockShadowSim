import vtk
import os
import pdb

import pyvista as pv
import numpy as np
# import cupy as cp
import time
# from pymeshfix import _meshfix
import tkinter as tk
from vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from vtk.util.numpy_support import vtk_to_numpy

import DataManager
import Mesh
import RKF
import RayTracer
import Light
import GuiIllumination

import cProfile
import time
import math
import scipy.integrate as integrate
from scipy.interpolate import Rbf
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
def wing_hit(t, state):
    
    x, y, z, dxdt, dydt, dzdt = state
    
    # findCellTol             = 10**(-20)
    # findCellGenCell         = vtk.vtkGenericCell()
    # findCellPcoords         = [0.0, 0.0, 0.0]
    # findCellWeights         = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    # inOutWingCheck          = wingMesh.cellLocator.FindCell(np.array([x,y,z]), findCellTol, findCellGenCell, findCellPcoords, findCellWeights)    

    # if (inOutWingCheck == -1):
        # return 1
    # else:
        # return 0
        
        
    if (wingMesh.locate_point_in_out_grid(np.array([x,y,z])) == 0):
        return 1
    else:
        return 0

wing_hit.terminal   = True
wing_hit.direction  = 0        
    
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #        
def ray_ode(t, state):
 
    x, y, z, dxdt, dydt, dzdt = state
    
    findCellTol            = 10**(-20)
    findCellGenCell        = vtk.vtkGenericCell()
    findCellPcoords        = [0.0, 0.0, 0.0]
    findCellWeights        = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    idx     = fluidMesh.cellLocator.FindCell(np.array([x,y,z]), findCellTol, findCellGenCell, findCellPcoords, findCellWeights)
    d2xdt2  = fluidMesh.grid['GradientCell'][idx,0]
    d2ydt2  = fluidMesh.grid['GradientCell'][idx,1]
    d2zdt2  = fluidMesh.grid['GradientCell'][idx,2]
    
    derCurrentState = [dxdt, dydt, dzdt, d2xdt2, d2ydt2, d2zdt2] 
    
    # print(t)
    return derCurrentState
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
def get_light_intersection(photonMapDict, sol):
    
    # ******************* #
    # Get interpolated normal at linear hit point - Gouraud/Phong Shading
    # ******************* #
    # Inside and outside points
    source  = np.array(sol.sol.interpolants[-1](sol.sol.interpolants[-1].t_min)[0:3])
    target  = np.array(sol.sol.interpolants[-1](sol.sol.interpolants[-1].t_max)[0:3])
    
    tol     = 1*(10**-20)
    t       = vtk.mutable(0)
    pcoords = [0.0,0.0,0.0]
    subId   = vtk.mutable(0)
    cellId  = vtk.mutable(0)
    
    linearHitPoint  = [0.0,0.0,0.0]
    hitTri    = vtk.vtkGenericCell()
    wingMesh.obbtree.IntersectWithLine(source, target, tol, t, linearHitPoint, pcoords, subId, cellId, hitTri)  

    # Vertices of the triangles intersected by the line connecting inside and outside points
    idxV0   = wingMesh.grid.cells_dict[5][cellId][0]
    idxV1   = wingMesh.grid.cells_dict[5][cellId][1]
    idxV2   = wingMesh.grid.cells_dict[5][cellId][2]
    v0      = wingMesh.grid.points[idxV0]
    v1      = wingMesh.grid.points[idxV1]
    v2      = wingMesh.grid.points[idxV2]    
    
    # Vertice normals
    n0      = wingMesh.grid['Normals'][idxV0]
    n1      = wingMesh.grid['Normals'][idxV1]
    n2      = wingMesh.grid['Normals'][idxV2]
    
    # Barycentric Coordinates
    areaT = np.linalg.norm((np.cross(v1-v0,v2-v0)))/2
    areaU = np.linalg.norm((np.cross(v0-linearHitPoint, v1-linearHitPoint)))/2
    areaV = np.linalg.norm((np.cross(v1-linearHitPoint, v2-linearHitPoint)))/2
    u = areaU/areaT
    v = areaV/areaT
    w = 1-u-v
    
    # Interpolated normal
    hitNormal      = u*n2 + v*n0 + w*n1
   
    # ******************* #
    # Get nonlinear hit point - Bisection on the 7th-order interpolant
    # ******************* #
    edge1           = v1-v0
    edge2           = v2-v0
    triNormal       = np.cross(edge1, edge2)
    
    
    tOut    = sol.sol.interpolants[-1].t_min
    pOut    = np.array(sol.sol.interpolants[-1](tOut)[0:3])
    evalOut = 90-np.arccos(np.dot((pOut-v0), triNormal)/(np.linalg.norm(pOut-v0)*np.linalg.norm(triNormal)))*180/np.pi 
    tIn     = sol.sol.interpolants[-1].t_max
    pIn     = np.array(sol.sol.interpolants[-1](tIn)[0:3])
    evalIn  = 90-np.arccos(np.dot((pIn-v0), triNormal)/(np.linalg.norm(pIn-v0)*np.linalg.norm(triNormal)))*180/np.pi
        
    tMid    = (tIn+tOut)/2
    pMid    = np.array(sol.sol.interpolants[-1](tMid)[0:3])
    evalMid = 90-np.arccos(np.dot((pMid-v0), triNormal)/(np.linalg.norm(pMid-v0)*np.linalg.norm(triNormal)))*180/np.pi
    
    while (np.abs(evalMid) >= 1*(10**-6)):
        if (evalOut*evalMid >= 0 and evalIn*evalMid < 0):
            tOut = tMid
            pOut = pMid
            evalOut = evalMid
        elif (evalOut*evalMid < 0 and evalIn*evalMid >= 0):
            tIn = tMid
            pIn = pMid
            evalIn = evalMid
            
        tMid    = (tIn+tOut)/2
        pMid    = np.array(sol.sol.interpolants[-1](tMid)[0:3])
        evalMid = 90-np.arccos(np.dot((pMid-v0), triNormal)/(np.linalg.norm(pMid-v0)*np.linalg.norm(triNormal)))*180/np.pi

    nonlinearHitPoint = pMid
    hitDirection      = np.array(sol.sol.interpolants[-1](tMid)[3:6])
    
    # plotter = pv.Plotter()
    # plotter.add_mesh(pv.PolyData(v0), color='red')
    # plotter.add_mesh(pv.PolyData(v1), color='red')
    # plotter.add_mesh(pv.PolyData(v2), color='red')
    # plotter.add_mesh(pv.PolyData(source), color='green')
    # plotter.add_mesh(pv.PolyData(target), color='blue')
    # plotter.add_mesh(pv.PolyData(linearHitPoint), color='magenta')
    # plotter.add_mesh(pv.PolyData(nonlinearHitPoint), color='cyan')
    # plotter.add_mesh(wingMesh.grid, opacity=0.2, show_edges=True)
    # plotter.set_background('white')
    # plotter.show()
    # pdb.set_trace()
    
    
    
    # stepOutWing  = sol.sol.interpolants[-1].t_min
    # stepInsWing  = sol.sol.interpolants[-1].t_max
    # pos = []
    # dir = []
    # t = np.linspace(stepOutWing, stepInsWing, 10000)
    # for i in range(0, t.shape[0], 1):
        # pos.append(sol.sol.interpolants[-1](t[i])[0:3])
        # dir.append(sol.sol.interpolants[-1](t[i])[3:6])
    # pos = np.array(pos)
    # dir = np.array(dir)
    # wingMesh.inOutChecker.SetInputData(pv.PolyData(pos))
    # wingMesh.inOutChecker.Update()

    # check = vtk_to_numpy(wingMesh.inOutChecker.GetOutput().GetPointData().GetArray('SelectedPoints'))
    # # idxHit = np.argwhere(np.diff(check[:]))+1
    # # hitPos = pos[idxHit[0][0],:]
    # # hitDir = dir[idxHit[0][0],:]
    # # pos     = hitPos
    # # dir     = hitDir/np.linalg.norm(hitDir)
     



    pos     = nonlinearHitPoint.reshape((1,3))
    dir     = hitDirection/np.linalg.norm(hitDirection)
    pow     = light.totalPower/light.photonsOrg.shape[0]
    if (np.linalg.norm(hitNormal) != 1):
        normal = hitNormal/np.linalg.norm(hitNormal)
    else:
        normal = hitNormal
        
    photonMapDict['pos'] = pos
    photonMapDict['dir'] = dir
    photonMapDict['pow'] = pow
    photonMapDict['normal'] = normal  

def nonlinear_ray_trace(photonMapDict, initialState, plotter):
    # Solve ODE
    sol = solve_ivp(fun=ray_ode, t_span=[0,10], y0=initialState, method='DOP853', dense_output=True, first_step=0.1, max_step=0.1, rtol=1*(10**-6), atol=1*(10**-6), events=wing_hit)
    
    # cProfile.runctx('sol = solve_ivp(fun=ray_ode, t_span=[0,10], y0=initialState, method="DOP853", dense_output=False,max_step=0.01, rtol=1*(10**-6), atol=1*(10**-6), events=wing_hit)',globals(), locals(), sort='tottime')        
    # tracedRay = pv.PolyData(sol.y[0:3,:].transpose())
    # plotter.add_mesh(tracedRay, color='yellow', render_points_as_spheres=True)
    # tracedDir = np.arctan(-sol.y[5,:]/sol.y[3,:])*180.0/np.pi
    
    # Point inside the wing (hit)
    if (sol.status == 1):
        get_light_intersection(photonMapDict, sol)


    light.add_photon_to_map(photonMapDict['pos'], photonMapDict['dir'], photonMapDict['pow'], photonMapDict['normal'])
    
    
    # outWingT = sol.sol.interpolants[-1].t_min
    # inWingT  = sol.sol.interpolants[-1].t_max
    # pos = []
    # dir = []
    # t = np.linspace(outWingT, inWingT, 1000)
    # for i in range(0, t.shape[0], 1):
        # pos.append(sol.sol.interpolants[-1](t[i])[0:3])
        # dir.append(sol.sol.interpolants[-1](t[i])[3:6])
    # pos = np.array(pos)
    # dir = np.array(dir)
  
    # wingMesh.inOutChecker.SetInputData(pv.PolyData(pos))
    # wingMesh.inOutChecker.Update()

    # check = vtk_to_numpy(wingMesh.inOutChecker.GetOutput().GetPointData().GetArray('SelectedPoints'))
    # idxHit = np.argwhere(np.diff(check[:]))+1
    # try:
        # hitPos = pos[idxHit[0][0],:]
        # hitDir = dir[idxHit[0][0],:]
        
        # pos     = hitPos
        # dir     = hitDir/np.linalg.norm(hitDir)
        # pow     = np.array([100,100,100])
        # normal  = np.array([1,1,1])
            
        # light.add_photon_to_map(pos, dir, pow, normal)
    # except:
        # pass
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #


# ------------------------------------------------------------------------ #
cwd = os.getcwd()
os.chdir(cwd + "\onera_solution")

# CFD volume solution and mesh loading
fileName = 'flow.vtu'
fluidMesh = Mesh.Mesh()
fluidMesh.grid = pv.read(fileName)

# Wing mesh loading
fileName = 'surface_flow.ply'
wingMesh = Mesh.Mesh()
pd = pv.read(fileName)
appendFilter = vtk.vtkAppendFilter()
appendFilter.AddInputData(pd)
appendFilter.Update()
uGrid = vtk.vtkUnstructuredGrid()
uGrid.ShallowCopy(appendFilter.GetOutput())
wingMesh.grid = pv.UnstructuredGrid(uGrid)


os.chdir(cwd)

# ------------------------------------------------------------------------ #
# plotter = pv.Plotter()
# # sliced  = fluidMesh.grid.slice(normal=[0,1,0], origin=(0,0.6,0), generate_triangles=False)
# # # clipped = fluidMesh.grid.clip(normal=[0,1,0], origin=(42,16,0.6))
# # plotter.add_mesh(sliced, scalars='GradientMagnitude',  clim=[0, 0.03], above_color='magenta', cmap='jet', show_edges=False)
# # plotter.add_mesh(fluidMesh.grid, scalars='GradientMagnitude', cmap='jet', show_edges=True, opacity=0.2)
# # # plotter.add_mesh(clipped, style='wireframe', show_edges=True)
# # # plotter.add_mesh(fluidMesh.grid, style='wireframe', color='white', opacity=0.3)
# # # plotter.add_mesh_slice(fluidMesh.grid, normal=(0,1,0), scalars='GradientCell',  clim=[0, 0.001], above_color='magenta', cmap='jet', show_edges=False)

# plotter.add_mesh(wingMesh.grid, show_edges=False, color='green')
# plotter.show_bounds(grid=True, location='back', color='black')
# plotter.show_axes()
# plotter.set_background('white')
# plotter.view_xz()
# plotter.show(auto_close=False)
# pdb.set_trace()
# ------------------------------------------------------------------------ #
# Cell locator for gradient interpolation
fluidMesh.build_cell_locator()
# Inside/outside wing checker for hit determination
wingMesh.build_grid_in_out_checker(1*(10**-20))
# OBB tree for linear intersection computation close to the wing surface
wingMesh.build_obb_tree()
# ------------------------------------------------------------------------ #
# Spectral light source setup
    # Sun positioning
sunAzimuth     = 30
sunElevation   = 90
sunRange       = 1
    # Spectral range
spectrumBin = [380, 780]

light = Light.Light()
light.set_number_of_photons(4000000)
light.set_direction(wingMesh, sunAzimuth, sunElevation, sunRange)
light.set_total_rgb_power(type='sun', lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)
light.set_origin(wingMesh)

light.set_gladstone_dale_constant(lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)
fluidMesh.compute_index_of_refraction(light.GDC)
# ------------------------------------------------------------------------ #
# Cell gradient computation
gradYN = 1
fluidMesh.compute_cell_gradient()
# ***************************# 
# gradYN = 0

# ------------------------------------------------------------------------ #
# Wing surface mesh normal computation
wingMesh.compute_surface_normals()
# ------------------------------------------------------------------------ #
# fluidMesh.grid['GradientMagnitude'] = ((fluidMesh.grid['GradientCell'][:,0]**2)+(fluidMesh.grid['GradientCell'][:,1]**2)+(fluidMesh.grid['GradientCell'][:,2]**2))**0.5
plotter = pv.Plotter()
# sliced  = fluidMesh.grid.slice(normal=[0,1,0], origin=(0,0.6,0), generate_triangles=False)
# plotter.add_mesh(sliced, scalars='GradientMagnitude',  clim=[0, 0.03], above_color='magenta', cmap='jet', show_edges=False)
# plotter.add_mesh(wingMesh.grid, show_edges=True, color='green')
# plotter.show_axes()
# plotter.set_background('white')
# plotter.view_xz()
# ------------------------------------------------------------------------ #
timer = vtk.vtkTimerLog()
timer.StartTimer()
pdb.set_trace()
for i in range(light.photonsOrg.shape[0]):
    print(i)
    
    photonMapDict = {}
    
    # Inside and outside points
    source  = light.photonsOrg[i,:]
    t       = 10**10
    target  = source + t*light.photonsDir[i,:]
        
    tol     = 1*(10**-20)
    t       = vtk.mutable(0)
    pcoords = [0.0,0.0,0.0]
    subId   = vtk.mutable(0)
    cellId  = vtk.mutable(0)
    
    linearHitPoint  = [0.0,0.0,0.0]
    hitTri    = vtk.vtkGenericCell()
    tmp = wingMesh.obbtree.IntersectWithLine(source, target, tol, t, linearHitPoint, pcoords, subId, cellId, hitTri)  
    
    idxV0   = wingMesh.grid.cells_dict[5][cellId][0]
    idxV1   = wingMesh.grid.cells_dict[5][cellId][1]
    idxV2   = wingMesh.grid.cells_dict[5][cellId][2]
    v0      = wingMesh.grid.points[idxV0]
    v1      = wingMesh.grid.points[idxV1]
    v2      = wingMesh.grid.points[idxV2]    
    
    n0      = wingMesh.grid['Normals'][idxV0]
    n1      = wingMesh.grid['Normals'][idxV1]
    n2      = wingMesh.grid['Normals'][idxV2]
    
    areaT = np.linalg.norm((np.cross(v1-v0,v2-v0)))/2
    areaU = np.linalg.norm((np.cross(v0-linearHitPoint, v1-linearHitPoint)))/2
    areaV = np.linalg.norm((np.cross(v1-linearHitPoint, v2-linearHitPoint)))/2
    u = areaU/areaT
    v = areaV/areaT
    w = 1-u-v
    
    hitNormal      = u*n2 + v*n0 + w*n1
    
    # source  = light.photonsOrg[i,:]
    # t       = 10**10
    # target  = source + t*light.photonsDir[i,:]
    # pt      = vtk.vtkPoints()
    # cellId  = vtk.vtkIdList()
    
    # tmp = wingMesh.obbtree.IntersectWithLine(source, target, pt, cellId)
    
    # Visibility Check
    if tmp != 0:
        pos     = np.array(linearHitPoint).reshape((1,3))
        dir     = light.photonsDir[i,:]
        dir     = dir/np.linalg.norm(dir)
        pow     = light.totalPower/light.photonsOrg.shape[0]
        if (np.linalg.norm(hitNormal) != 1):
            normal = hitNormal/np.linalg.norm(hitNormal)
        else:
            normal = hitNormal
        photonMapDict['pos']    = pos
        photonMapDict['dir']    = dir
        photonMapDict['pow']    = pow
        photonMapDict['normal'] = normal
    
        initialState    = (light.photonsOrg[i,0], light.photonsOrg[i,1], light.photonsOrg[i,2], light.photonsDir[i,0], light.photonsDir[i,1], light.photonsDir[i,2])
        nonlinear_ray_trace(photonMapDict, initialState, plotter)

pdb.set_trace()
# ************************************************************************** #
# count = 0
# for idx in range(light.photonsOrg.shape[0]):
    # origin      = light.photonsOrg[idx,:]
    # direction   = light.photonsDir[idx,:]
    
    # source      = origin
    # t           = 10**10
    # target      = source + t*direction
    
    # tol         = 1*(10**-20)
    # t           = vtk.mutable(0)
    # x           = [0.0,0.0,0.0]
    # pcoords     = [0.0,0.0,0.0]
    # subId       = vtk.mutable(0)
    # cellId      = vtk.mutable(0)
    # cell        = vtk.vtkGenericCell()
    # hit         = wingMesh.obbtree.IntersectWithLine(source, target, tol, t, x, pcoords, subId, cellId, cell)

    # if hit != 0:
        # idxV0 = wingMesh.grid.cells_dict[5][cellId][0]
        # idxV1 = wingMesh.grid.cells_dict[5][cellId][1]
        # idxV2 = wingMesh.grid.cells_dict[5][cellId][2]
        # v0    = wingMesh.grid.points[idxV0]
        # v1    = wingMesh.grid.points[idxV1]
        # v2    = wingMesh.grid.points[idxV2]
        # n0    = wingMesh.grid['Normals'][idxV0]
        # n1    = wingMesh.grid['Normals'][idxV1]
        # n2    = wingMesh.grid['Normals'][idxV2]
        
        # hitPoint = x
        
         # # Vertice normals
        # n0      = wingMesh.grid['Normals'][idxV0]
        # n1      = wingMesh.grid['Normals'][idxV1]
        # n2      = wingMesh.grid['Normals'][idxV2]
        
        # # Barycentric Coordinates
        # areaT = np.linalg.norm((np.cross(v1-v0,v2-v0)))/2
        # areaU = np.linalg.norm((np.cross(v0-hitPoint, v1-hitPoint)))/2
        # areaV = np.linalg.norm((np.cross(v1-hitPoint, v2-hitPoint)))/2
        # u = areaU/areaT
        # v = areaV/areaT
        # w = 1-u-v
        
        # # Interpolated normal
        # hitNormal      = u*n2 + v*n0 + w*n1

        # pos     = hitPoint
        # dir     = direction
        # dir     = dir/np.linalg.norm(dir)
        # pow     = light.totalPower/light.photonsOrg.shape[0]
        # if (np.linalg.norm(hitNormal) != 1):
            # normal = hitNormal/np.linalg.norm(hitNormal)
        # else:
            # normal = hitNormal
        
        # light.add_photon_to_map(pos, dir, pow, normal)
        
    # print(count)
    # count+=1



# timer.StopTimer()
# time = timer.GetElapsedTime()
# print("Time to ray trace: " + str(time))

# ------------------------------------------------------------------------ #
# plotter.add_mesh(light.photonMap, color ='red', render_points_as_spheres=True)
# plotter.show(auto_close=False)
# ------------------------------------------------------------------------ #
cwd         = os.getcwd()
simPath = '.\sim_results'
os.chdir(simPath)

illumPath = '.\elevation_' + str(sunElevation) + '_azimuth_' + str(sunAzimuth) + '_range_' +  str(sunRange)
if gradYN == 1:
    fileName    = 'm6_frt_flow_'  + str(light.photonsOrg.shape[0]) + '_photons.vtp'
elif gradYN == 0:
    fileName    = 'm6_frt_noflow_'  + str(light.photonsOrg.shape[0]) + '_photons.vtp'

if (os.path.isdir(illumPath) == False):
    os.mkdir(illumPath)
    os.chdir(illumPath)
        
    writer      = DataManager.DataManager(fileName)
    writer.write_vtp(light.photonMap)
    os.chdir(cwd)
    
elif (os.path.isdir(illumPath) == True):
    os.chdir(illumPath)
    
    if (os.path.exists(fileName) == True):
        overwriteCheck = input("Overwrite existing results? (Y/N)")
        if (overwriteCheck.lower() == 'y'):
            writer      = DataManager.DataManager(fileName)
            writer.write_vtp(light.photonMap)
        elif (overwriteCheck.lower() == 'n'):
            pdb.set_trace()
    elif (os.path.exists(fileName) == False):
        writer      = DataManager.DataManager(fileName)
        writer.write_vtp(light.photonMap)
    
os.chdir(cwd)
# ------------------------------------------------------------------------ #
pdb.set_trace()





