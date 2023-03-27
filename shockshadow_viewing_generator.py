import os
import pdb
import pickle

import vtk
import pyvista as pv

import numpy as np
import math
import scipy.io as sio

import DataManager
import Light
import Mesh
import Camera
import Material
import RKF
import RayTracer
import Brdf 
import GuiView

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
def get_camera_intersection(sol):
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
        
    cam.add_photon_to_map(pos, dir)
    
    cam.hitPoint        = pos
    cam.hitDirection    = dir
    pdb.set_trace()


def nonlinear_ray_trace(initialState, plotter):
    # Solve ODE
    sol = solve_ivp(fun=ray_ode, t_span=[0,10], y0=initialState, method='DOP853', dense_output=True, first_step=0.1, max_step=0.1, rtol=1*(10**-6), atol=1*(10**-6), events=wing_hit)
    
    # cProfile.runctx('sol = solve_ivp(fun=ray_ode, t_span=[0,10], y0=initialState, method="DOP853", dense_output=False,max_step=0.01, rtol=1*(10**-6), atol=1*(10**-6), events=wing_hit)',globals(), locals(), sort='tottime')        
    # tracedRay = pv.PolyData(sol.y[0:3,:].transpose())
    # plotter.add_mesh(tracedRay, color='yellow', render_points_as_spheres=True)
    # tracedDir = np.arctan(-sol.y[5,:]/sol.y[3,:])*180.0/np.pi

    # Point inside the wing (hit)
    if (sol.status == 1):
        get_camera_intersection(sol)
        
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
# Cell locator for gradient interpolation
fluidMesh.build_cell_locator()
# Inside/outside wing checker for hit determination
wingMesh.build_grid_in_out_checker(1*(10**-20))
# OBB tree for linear intersection computation close to the wing surface
wingMesh.build_obb_tree()
# ------------------------------------------------------------------------ #
# Spectral light source setup
    # Sun positioning
sunAzimuth     = 35
sunElevation   = 90
numOfPhotons   = 1043267
sunRange       = 1
    # Spectral range
spectrumBin = [380, 780]
light = Light.Light()
# light.set_number_of_photons(3500000)
# light.set_direction(wingMesh, sunAzimuth, sunElevation, sunRange)
# light.set_origin(wingMesh)
# light.set_total_rgb_power(type='sun', lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)
light.set_gladstone_dale_constant(lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)
fluidMesh.compute_index_of_refraction(light.GDC)
# ------------------------------------------------------------------------ #
# Cell gradient computation
gradYN = 1
fluidMesh.compute_cell_gradient()
# ------------------------------------------------------------------------ #
# Wing surface mesh normal computation
wingMesh.compute_surface_normals()
# ------------------------------------------------------------------------ #
# # # fluidMesh.grid['GradientMagnitude'] = ((fluidMesh.grid['GradientCell'][:,0]**2)+(fluidMesh.grid['GradientCell'][:,1]**2)+(fluidMesh.grid['GradientCell'][:,2]**2))**0.5
# # # plotter = pv.Plotter()
# # # sliced  = fluidMesh.grid.slice(normal=[0,1,0], origin=(0,0.6,0), generate_triangles=False)
# # # # # clipped = fluidMesh.grid.clip(normal=[0,1,0], origin=(42,16,0.6))
# # # plotter.add_mesh(sliced, scalars='GradientMagnitude',  clim=[0, 0.03], above_color='magenta', cmap='jet', show_edges=False)
# # # # # plotter.add_mesh(fluidMesh.grid, scalars='GradientMagnitude', cmap='jet', show_edges=True, opacity=0.2)
# # # # # plotter.add_mesh(clipped, style='wireframe', show_edges=True)
# # # plotter.add_mesh(fluidMesh.grid, style='wireframe', color='white', opacity=0.3)
# # # # # plotter.add_mesh_slice(fluidMesh.grid, normal=(0,1,0), scalars='GradientCell',  clim=[0, 0.001], above_color='magenta', cmap='jet', show_edges=False)
# # # plotter.add_mesh(wingMesh.grid, show_edges=False, color='green')
# # # # # plotter.show_bounds(grid=True, location='back', color='black')
# # # plotter.show_axes()
# # # plotter.set_background('white')
# # # plotter.view_xz()
# # # plotter.show(auto_close=False)
# # # pdb.set_trace()
# ------------------------------------------------------------------------ #
# cwd = os.getcwd()
# os.chdir(cwd + "\sim_results")
        
# dirPath = '.\elevation_' + str(sunElevation) + '_azimuth_' + str(sunAzimuth) + '_range_' +  str(sunRange)
# os.chdir(dirPath)
        
# fileName = 'm6_frt_flow_' + str(10201) + '_photons.vtp'
        
# reader      = DataManager.DataManager(fileName)
# photonMap   = reader.read_vtp()
# light.set_photon_map(photonMap)


cwd = os.getcwd()
os.chdir(cwd + "\photon_map_database")    

fileNameVTP = str(sunElevation) + '_ele_' + str(sunAzimuth) + '_azi_' + str(numOfPhotons) + '_photons.vtp'
reader      = DataManager.DataManager(fileNameVTP)
photonMap   = reader.read_vtp()
light       = Light.Light()
light.set_photon_map(photonMap)



os.chdir(cwd)
# ------------------------------------------------------------------------ #
light.photonMap['DirectionXComponent'] = light.photonMap['Direction'][:,0]
light.photonMap['DirectionYComponent'] = light.photonMap['Direction'][:,1]
light.photonMap['DirectionZComponent'] = light.photonMap['Direction'][:,2]
# mesh  = pv.StructuredGrid(light.photonMap.points[:,0],light.photonMap.points[:,1],light.photonMap.points[:,2])
# mesh["direction"] = -light.photonMap['Direction']
# mesh["magnitude"] = np.ones((light.photonMap.points.shape[0]))*0.1
# glyphs = mesh.glyph(orient="direction", scale="magnitude", geom=pv.Arrow(shaft_radius=0.01, tip_radius=0.03, tip_length=0.2, tip_resolution=100))
       

plotter    = pv.Plotter(notebook=False)
plotter.add_mesh(wingMesh.grid, color='green', show_edges=False)
# plotter.add_mesh(glyphs, color="yellow")
plotter.add_mesh(light.photonMap, scalars='DirectionZComponent', cmap='jet', render_points_as_spheres=True)
plotter.set_background('gray')
plotter.add_axes(color='black')
plotter.show(auto_close=False, return_img=True)
# cpos, image = plotter.show(auto_close=False, return_img=True)
pttpCam = plotter.camera


horRes             = 200
verRes             = 200
eye                = np.array(pttpCam.GetPosition())
lookAt             = np.array(pttpCam.GetFocalPoint())
up                 = np.array(pttpCam.GetViewUp())
pdb.set_trace()
coord = vtk.vtkCoordinate()
coord.SetCoordinateSystemToDisplay()
h = image.shape[0] # number of columns of pixels - horizontal
v = image.shape[1] # number of rows of pixels - vertical
viewPlane = np.zeros((h*v,3))
idx = 0
for i in range(0, h,1):
    for j in range(0, v, 1):
        coord.SetValue(i,j,0)
        viewPlane[idx,0] = coord.GetComputedWorldValue(plotter.renderer)[0]
        viewPlane[idx,1] = coord.GetComputedWorldValue(plotter.renderer)[1]
        viewPlane[idx,2] = coord.GetComputedWorldValue(plotter.renderer)[2]
        idx+=1
                
viewPlaneCenter        = np.array([np.mean(viewPlane[:,0]),np.mean(viewPlane[:,1]),np.mean(viewPlane[:,2])])
viewPlaneDistance      = np.linalg.norm(viewPlaneCenter-eye)
        
viewPlaneWidth         = np.linalg.norm(viewPlane[0,:]-viewPlane[h-1,:])
viewPlaneHeight        = np.linalg.norm(viewPlane[h-1,:]-viewPlane[(h-1)*(v-1),:])
        
horPixelSize           = viewPlaneWidth/horRes
verPixelSize           = viewPlaneHeight/verRes
        
numNN                  = 10000
maxRadius              = 0.008
numOfSamples           = 1
  
cam = Camera.Camera()
cam.set_eye(eye)
cam.set_look_at(lookAt)
cam.set_up_vector(up)
cam.set_view_plane_distance(viewPlaneDistance)
cam.set_horizontal_resolution(horRes)
cam.set_vertical_resolution(verRes)
cam.set_pixel_size(horPixelSize, verPixelSize)
cam.compute_uvw()
cam.set_canvas()
cam.set_photon_mapping_nn(numNN)
cam.set_photon_mapping_max_radius(maxRadius)
cam.set_vp_width(viewPlaneWidth)
cam.set_vp_height(viewPlaneHeight)
cam.set_number_of_samples(numOfSamples)
pdb.set_trace()
# ------------------------------------------------------------------------ #
# plotter = pv.Plotter(notebook=False)
# plotter.add_mesh(light.photonMap, render_points_as_spheres=True, color='yellow')
# plotter.add_mesh(wingMesh.grid, color='green', show_edges=False)
# plane = pv.PolyData(viewPlane)
# plotter.add_mesh(plane, color='magenta', point_size=10)
# plotter.add_mesh(pv.Sphere(radius=0.05, center=eye, theta_resolution=100, phi_resolution=100), color='green')
# plotter.add_mesh(pv.Sphere(radius=0.05, center=lookAt, theta_resolution=100, phi_resolution=100), color='orange')
# plotter.show(auto_close=False)
# ------------------------------------------------------------------------ #
wingMat     = Material.Material()
kd      = 0.0
cd      = [0.8,0.8,0.8]
rough   = 0.5
metal   = 1.0

wingMat.set_diffuse_coefficient(kd)
wingMat.set_diffuse_color(np.array([float(cd[0]),float(cd[1]),float(cd[2])]))
wingMat.set_roughness(rough)
wingMat.set_metalness(metal)
# ------------------------------------------------------------------------ #
plotter = pv.Plotter()
cam.HDRRadiance        = np.zeros((cam.horRes, cam.verRes, 3))
# cam.visualPoints = np.zeros((cam.horRes, cam.verRes, 2)) 

# Nonlinear Ray Tracing

count = 0
for v in range(0, cam.verRes-1, 1):
    for h in range(0, cam.horRes-1, 1):
        
        n = int(np.floor(np.sqrt(cam.nSamples)))
        for p in range(n):
            for q in range(n):
                count += 1
                print(count)
                
                origin     = np.zeros(3)
                direction  = np.zeros(3)
                
                # Viewing System Coord.
                # origin[0]   = cam.horPixSize * (h - (cam.horRes / 2) + 0.5)
                # origin[1]   = cam.verPixSize * (v - (cam.verRes / 2) + 0.5)
                origin[0]   = cam.horPixSize * (h - (cam.horRes / 2) + ((q+np.random.rand())/n))
                origin[1]   = cam.verPixSize * (v - (cam.verRes / 2) + ((p+np.random.rand())/n))
                origin[2]   = -cam.vpDist
                
                # World Coord.
                direction   = np.multiply(origin[0], cam.u) + np.multiply(origin[1], cam.v) + np.multiply(origin[2], cam.w)
                direction   = direction/np.linalg.norm(direction)

                l           = math.sqrt( (cam.vpDist**2) + (origin[0]**2 + origin[1]**2) )
                origin      = cam.eye + l*direction
                
                source      = origin
                t           = 10**10
                target      = origin + t*direction
                hitPoint    = vtk.vtkPoints()
                cellId      = vtk.vtkIdList()
                    
                hit         = wingMesh.obbtree.IntersectWithLine(source, target, hitPoint, cellId)
                
                cam.hitPoint = np.array([])
                # Visibility Check
                if hit != 0:
                    pdb.set_trace()
                    # Nonlinear Ray Trace
                    initialState    = (origin[0], origin[1], origin[2], direction[0], direction[1], direction[2])
                    nonlinear_ray_trace(initialState, plotter)
                        
                    # Linear Ray Trace    
                    # cam.hitPoint = np.array(hitPoint.GetData().GetTuple3(0))

                # Photon Mapping
                L = np.zeros(3)
                if (cam.hitPoint.size != 0):
                    nearPhotons     = vtk.vtkIdList()
                    idxNearPhotons  = []
                    distNearPhotons = []
                        
                    light.kdTreePhotonMap.FindClosestNPoints(cam.numOfNearPhotons, cam.hitPoint.reshape(3), nearPhotons)
                        
                    for i in range(nearPhotons.GetNumberOfIds()):
                        id          = nearPhotons.GetId(i)
                        nearPhoton  = np.array(light.kdTreePhotonMap.GetDataSet().GetPoint(id))
                        d           = np.linalg.norm(nearPhoton - cam.hitPoint)

                        if (d <= cam.maxDistance):
                            idxNearPhotons.append(id)
                            distNearPhotons.append(d)

                    if idxNearPhotons:
                        idxNearPhotons  = np.array(idxNearPhotons)
                        distNearPhotons = np.array(distNearPhotons)
                        # pmPlotter = pv.Plotter(notebook=False)
                        # pmPlotter.add_mesh(wingMesh.grid, color='black', show_edges=True)
                        # pmPlotter.add_mesh(light.photonMap, scalars='DirectionZComponent', cmap='jet', render_points_as_spheres=True)
                        # pmPlotter.add_axes(color='black')
                        # pmPlotter.set_background('white')
                        # pmPlotter.add_mesh(pv.PolyData(cam.hitPoint), color='magenta', render_points_as_spheres=True)
                        # pmPlotter.add_mesh(pv.PolyData(light.photonMap.points[idxNearPhotons,:]), color='cyan', render_points_as_spheres=True)
                        # # pmPlotter.add_mesh(pv.Arrow(start=cam.hitPoint, direction=-light.photonMap['Normal'][idx,:],     shaft_radius=0.01, tip_radius=0.03, tip_length=0.2, tip_resolution=100, scale=0.8), color='red')
                        # # pmPlotter.add_mesh(pv.Arrow(start=cam.hitPoint, direction=-light.photonMap['Direction'][idx,:], shaft_radius=0.01, tip_radius=0.03, tip_length=0.2, tip_resolution=100, scale=0.8), color='yellow')
                        # # pmPlotter.add_mesh(pv.Arrow(start=cam.hitPoint, direction=-direction,                           shaft_radius=0.01, tip_radius=0.03, tip_length=0.2, tip_resolution=100, scale=0.8), color='blue')
                        # pmPlotter.show(auto_close=False, interactive_update=True)
                        for j in range(idxNearPhotons.shape[0]):                 
                            idx = idxNearPhotons[j]
                            # # # microfacet      =   Brdf.BRDF(material=wingMat, surfaceNormal   =   light.photonMap['Normal'][idx,:],\
                                                                            # # # lightDirection  =   -light.photonMap['Direction'][idx,:],\
                                                                            # # # viewDirection   =   -direction)
                            # # # microfacet.set_value()
                            # # # fr = microfacet.fr
                            
                            # Lambertian surface
                            fr = (wingMat.diffuseColor)/np.pi
                            
                            # Gaussian filter
                            alpha   = 0.918
                            beta    = 1.953
                            term1   = 1-np.exp(-beta*(distNearPhotons[j]**2)/(2*(np.max(distNearPhotons)**2)))
                            term2   = 1-np.exp(-beta)
                            wpg     = alpha*(1-(term1/term2))
                            
                            L += (fr)*(light.photonMap['Power'][idx,:])*wpg

                        L /= (np.pi)*(np.max(distNearPhotons)**2)

                cam.HDRRadiance[h,v,:] += L
                
        cam.HDRRadiance[h,v,:] /= cam.nSamples

# ------------------------------------------------------------------------ #
pdb.set_trace()

cwd = os.getcwd()
os.chdir(cwd + "\sim_results")
       
dirPath = '.\elevation_' + str(sunElevation) + '_azimuth_' + str(sunAzimuth) + '_range_' +  str(sunRange)
os.chdir(dirPath)

fileNameNPY = 'onera_m6_flow_' + str(cam.horRes*cam.verRes*cam.nSamples) + '_pixels.npy'
np.save(fileNameNPY, cam.HDRRadiance, allow_pickle=True)


fileNamePKL = 'onera_m6_flow_' + str(cam.horRes*cam.verRes*cam.nSamples) + '_pixels.PKL'
del cam.photonMap
del cam.canvas
writer = open(fileNamePKL, "wb")
pickle.dump(cam, writer)
writer.close()

os.chdir(cwd)
pdb.set_trace()















# from PIL import Image


# pixColor = cam.HDRRadiance
# cam.srgb_to_xyz(pixColor)

# # Luminance channel
# im1 = Image.fromarray(pixColor[:,:,1], mode='L')
# im1.show()

# # Lightness channel
# whitePoint = np.array([95.047, 100.0, 108.883])
# cam.xyz_to_luv(pixColor, whitePoint)
# im2 = Image.fromarray(pixColor[:,:,0]-np.mean(pixColor[:,:,0]), mode='L')
# im2.show()

# pixColor = cam.HDRRadiance
# im3 = Image.fromarray(pixColor, mode='RGB')
# im3 = im3.convert('LAB')

# # Compressed and Gamma corrected RGB channels






























# cam.pixColor = np.divide(cam.HDRRadiance, np.mean(cam.HDRRadiance, axis=(0,1)))
# cam.pixColor = (cam.HDRRadiance - np.min(cam.HDRRadiance, axis=(0,1)))/(np.max(cam.HDRRadiance, axis=(0,1)) - np.min(cam.HDRRadiance, axis=(0,1)))
# cam.synthesize_image()
# cam.plot_synthetic_image()

# *************** #
cam.pixColor = cam.HDRRadiance
cam.srgb_to_xyz(cam.pixColor)

# Yn = np.max(cam.pixColor[:,:,1])
Yn = 100.0
Y  = cam.pixColor[:,:,1]

relLight            = Y/Yn
fRelLight           = np.where(relLight != 0, np.where(relLight <= 0.008856, (841.0/108.0)*relLight+(4.0/29.0), relLight**(1/3)), 0.0)
relLightStar        = np.where(fRelLight != 0, 116*(fRelLight) - 16, 0.0)
relLightStarDiff    = np.where(relLightStar != 0, relLightStar - np.mean(relLightStar, axis=(0,1), where=(relLightStar != 0)), 0.0)
cam.pixColor        = relLightStarDiff
# *************** #
# pdb.set_trace()
# cam.pixColor = cam.HDRRadiance
# cam.convert_to_xyz()
# cam.apply_tone_mapping(0.18)
# cam.apply_gamma_encoding()
# cam.pixColor = np.divide(cam.pixColor, np.mean(cam.pixColor, axis=(0,1)))[:,:,1]


lut = vtk.vtkLookupTable()
# 8-bit per channel
lut.SetNumberOfColors(256)
lut.SetSaturationRange(0.0,0.0)
lut.SetHueRange(0.0,0.0)
lut.SetValueRange(0,1)
lut.SetTableRange(np.min(cam.pixColor, axis=(0,1)), np.max(cam.pixColor, axis=(0,1)))
# lut.SetRampToLinear()
lut.SetRampToSCurve()
# lut.SetRampToSQRT()
lut.Build()

imgPlotter = pv.Plotter()
    
canvasDir   = ((cam.lookAt-cam.eye)/np.linalg.norm(cam.lookAt-cam.eye))
canvasOrg   = cam.eye + cam.vpDist*canvasDir
canvas      = pv.Plane(center=canvasOrg,direction=canvasDir,i_size=cam.vpWidth, j_size=cam.vpHeight,i_resolution=cam.horRes,j_resolution=cam.verRes)
    
pixelColor  = np.zeros(cam.horRes*cam.verRes)
idx = 0
for v in range(0, cam.verRes, 1):
    for h in range(0, cam.horRes, 1):
        color = [0.0, 0.0, 0.0]
        lut.GetColor(cam.pixColor[h,v], color)
        pixelColor[idx] = color[0]
        idx += 1

canvas.cell_arrays['PixelColor'] = pixelColor.reshape(pixelColor.shape[0],1)
canvasActor = imgPlotter.add_mesh(canvas, scalars='PixelColor', cmap='gray')
imgPlotter.add_axes(color='black')
imgPlotter.show(interactive=True, auto_close=False)
pdb.set_trace()


mainPath = os.getcwd()

simPath = mainPath + '\sim_results'
os.chdir(simPath)

illumPath = simPath + '\elevation_' + str(sunElevation) + '_azimuth_' + str(sunAzimuth) + '_range_' +  str(sunRange)
os.chdir(illumPath)

viewCount   = 0
imagePath   = illumPath + '\camera' + str(viewCount)

while (os.path.isdir(imagePath) == True):
    viewCount   += 1
    imagePath   = illumPath + '\camera' + str(viewCount)

imagePath = imagePath
if (os.path.isdir(imagePath) == False):
    os.mkdir(imagePath)




pdb.set_trace()

os.chdir(self.imagePath)

if self.grad == 1:
    fileNameNPY    = 'xrf_brt_flow.npy'
    fileNamePKL    = 'xrf_brt_flow.pkl'
    fileNameVTP    = 'xrf_brt_flow.vtp'
elif self.grad == 0:
    fileNameNPY    = 'xrf_brt_noflow.npy'
    fileNamePKL    = 'xrf_brt_noflow.pkl'
    fileNameVTP    = 'xrf_brt_noflow.vtp'
        
writer = DataManager.DataManager(fileNameVTP)
writer.write_vtp(self.cam.photonMap)
    
np.save(fileNameNPY, self.cam.HDRRadiance, allow_pickle=True)
    
del self.cam.photonMap
del self.cam.canvas
writer = open(fileNamePKL, 'wb')
pickle.dump(self.cam, writer)
writer.close()
self.cam.photonMap  = pv.PolyData()
self.cam.canvas     = vtk.vtkImageCanvasSource2D()
    
    
os.chdir(self.mainPath)   

























# # ------------------------------------------------------------------------ #
# fluidMesh   = Mesh.Mesh()
# wingMesh    = Mesh.Mesh()
# light       = Light.Light()
# rkf         = RKF.RKF()
# cam         = Camera.Camera()
# wingMat     = Material.Material()
# tracer      = RayTracer.RayTracer()

# gui = GuiView.GuiView(fluidMesh, wingMesh, light, rkf, cam, wingMat, tracer)
# gui.initialize_frames()
# gui.initialize_labels()
# gui.initialize_entries()
# gui.initialize_buttons()

# # ------------------------------------------------------------------------ #
# pdb.set_trace()
# cwd         = os.getcwd()

# simPath = '.\sim_results'
# os.chdir(simPath)

# illumPath = '.\elevation_' + str(gui.sunElevation) + '_azimuth_' + str(gui.sunAzimuth) + '_range_' +  str(gui.sunRange)
# os.chdir(illumPath)

# viewCount   = 0
# imagePath   = '.\camera' + str(viewCount)


# while (os.path.isdir(imagePath) == True):
    # viewCount   += 1
    # imagePath   = '.\camera' + str(viewCount)
    
# if (os.path.isdir(imagePath) == False):
    # os.mkdir(imagePath)
    # os.chdir(imagePath)

    # if gui.grad == 1:
        # fileNameNPY    = 'xrf_brt_flow_'  + str(cam.horRes*cam.verRes) + '_pixels.npy'
        # fileNamePKL    = 'xrf_brt_flow_'  + str(cam.horRes*cam.verRes) + '_pixels.pkl'
        # fileNameVTP    = 'xrf_brt_flow_'  + str(cam.horRes*cam.verRes) + '_pixels.vtp'
    # elif gui.grad == 0:
        # fileNameNPY    = 'xrf_brt_noflow_'  + str(cam.horRes*cam.verRes) + '_pixels.npy'
        # fileNamePKL    = 'xrf_brt_noflow_'  + str(cam.horRes*cam.verRes) + '_pixels.pkl'
        # fileNameVTP    = 'xrf_brt_noflow_'  + str(cam.horRes*cam.verRes) + '_pixels.vtp'
        

    # writer = DataManager.DataManager(fileNameVTP)
    # writer.write_vtp(cam.photonMap)
    

    # np.save(fileNameNPY, cam.HDRRadiance, allow_pickle=True)
    
    # del cam.photonMap
    # del cam.canvas
    # writer = open(fileNamePKL, 'wb')
    # pickle.dump(cam, writer)
    # writer.close()
    
# os.chdir(cwd)   
# pdb.set_trace()




# # ------------------------------------------------------------------------ #
# cwd = os.getcwd()
# os.chdir(cwd + "\processed_mesh")
# # ------------------------------------------------------------------------ #
# fileName = "XRF1_unrefined_tetra_domain_edited.dat"
# reader = DataManager.DataManager(fileName)
# reader.open_and_read_dat()

# cellTypes = np.empty(reader.numOfElements, dtype=np.uint8)
# cellTypes[:] = vtk.VTK_TETRA

# fluidMesh = Mesh.Mesh()
# fluidMesh.construct_unstructured_grid(reader.points, reader.attributes, reader.cells, cellTypes)
# # ------------------------------------------------------------------------ #
# fileName = "XRF1_unrefined_tetra_surface.obj"
# reader = DataManager.DataManager(fileName)
# reader.open_and_read_obj()

# cellTypes = np.empty(reader.numOfElements, dtype=np.uint8)
# cellTypes[:] = vtk.VTK_TRIANGLE

# wingMesh = Mesh.Mesh()
# wingMesh.construct_unstructured_grid(reader.points, reader.attributes, reader.cells, cellTypes)
# # ------------------------------------------------------------------------ #
# os.chdir(cwd)
# # ------------------------------------------------------------------------ #
# tol = 1*(10**-20)
# fluidMesh.build_grid_in_out_checker(tol)
# wingMesh.build_grid_in_out_checker(tol)
# wingMesh.build_bbox_in_out_checker(tol)
# ------------------------------------------------------------------------ #
# light   = Light.Light()
# spectrumBin = [380, 780]
# light.set_total_rgb_power(type='sun', lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)
# light.set_gladstone_dale_constant(lambdaLow=spectrumBin[0], lambdaHigh=spectrumBin[1], lambdaDisc=1)

# cwd = os.getcwd()
# os.chdir(cwd + "\illumination_results")
# fileName = "xrf_frt_azimuth_25_elevation_75_range_5_1000_photons.vtp"
# reader = DataManager.DataManager(fileName)
# photonMap = reader.read_vtp()
# light.set_photon_map(photonMap)
# ------------------------------------------------------------------------ #
# fluidMesh.compute_index_of_refraction(light.GDC)
# # ------------------------------------------------------------------------ #
# fluidMesh.compute_gradient()
# # ------------------------------------------------------------------------ #
# fluidMesh.build_octree()
# fluidMesh.set_gaussian_interpolator(sharpness=0.1, radius=0.1)
# # fluidMesh.set_voronoi_interpolator()
# # fluidMesh.set_shepard_interpolator(power=2.0, radius=0.05)
# # ------------------------------------------------------------------------ #
# wingMesh.build_obb_tree()
# wingMesh.compute_surface_normals()
# # ------------------------------------------------------------------------ #
# rkf = RKF.RKF()
# rkf.set_tolerance(1*(10**-9))
# rkf.set_fac_max(1.5)
# rkf.set_fac_min(0.1)
# rkf.set_h_init(0.001)
# rkf.set_h_max(0.1)
# rkf.set_h_min(0.0001)
# rkf.set_density_threshold(10)
# ------------------------------------------------------------------------ #
# tracer  = RayTracer.RayTracer()
# ------------------------------------------------------------------------ #
# plotter = pv.Plotter(notebook=False,lighting='three lights')
# plotter.add_mesh(photonMap, render_points_as_spheres=True, color='blue')
# plotter.add_mesh(wingMesh.grid, color=[168/256, 189/256, 219/256], show_edges=False, edge_color='black', lighting=True, specular=0.9)
# cpos, image = plotter.show(auto_close=False, return_img=True)
# pdb.set_trace()
# pttpCam         = plotter.camera
# eye             = np.array(pttpCam.GetPosition())
# lookAt          = np.array(pttpCam.GetFocalPoint())
# upVector        = np.array(pttpCam.GetViewUp())
# viewDirection   = np.array(pttpCam.GetDirectionOfProjection())


# coord = vtk.vtkCoordinate()
# coord.SetCoordinateSystemToDisplay()
# viewPlane = np.zeros((image.shape[0]*image.shape[1],3))
# idx = 0
# for i in range(0, image.shape[0],1):
    # for j in range(0, image.shape[1], 1):
        # coord.SetValue(i,j,0)
        # viewPlane[idx,0] = coord.GetComputedWorldValue(plotter.renderer)[0]
        # viewPlane[idx,1] = coord.GetComputedWorldValue(plotter.renderer)[1]
        # viewPlane[idx,2] = coord.GetComputedWorldValue(plotter.renderer)[2]
        # idx+=1
        
      
# viewPlaneCenter     = np.array([np.mean(viewPlane[:,0]),np.mean(viewPlane[:,1]),np.mean(viewPlane[:,2])])
# viewPlaneDistance   = np.linalg.norm(viewPlaneCenter-eye)
# pixelSize           = np.linalg.norm(viewPlane[1,:]-viewPlane[0,:])
# # PRESS Q AFTER DECIDING ON CAMERA POSITION
# plotter = pv.Plotter(notebook=False,lighting='three lights')
# plotter.add_mesh(photonMap, render_points_as_spheres=True, color='blue')
# plotter.add_mesh(wingMesh.grid, color=[168/256, 189/256, 219/256], show_edges=False, edge_color='black', lighting=True, specular=0.9)

# # plane = pv.Plane(center=lookAt,direction=viewDirection,i_size=4,j_size=4,i_resolution=10, j_resolution=10)
# # plotter.add_mesh(plane, color='white', show_edges=True, opacity=0.5)

# plane = pv.PolyData(viewPlane)
# plotter.add_mesh(plane, color='magenta', point_size=10)
# plotter.add_mesh(pv.Sphere(radius=0.01, center=eye, theta_resolution=100, phi_resolution=100), color='green')
# plotter.add_mesh(pv.Sphere(radius=0.5, center=lookAt, theta_resolution=100, phi_resolution=100), color='orange')
# plotter.show(auto_close=False)
# pdb.set_trace()




# ------------------------------------------------------------------------ #
# cam = Camera.Camera()
# cam.set_eye(eye)
# cam.set_look_at(lookAt)
# cam.set_up_vector(upVector)
# cam.set_view_plane_distance(viewPlaneDistance)
# cam.set_horizontal_resolution(image.shape[0])
# cam.set_vertical_resolution(image.shape[1])
# cam.set_pixel_size(pixelSize)
# cam.compute_uvw()
# ------------------------------------------------------------------------ #
# wingMat = Material.Material()
# wingMat.set_diffuse_coefficient(0)
# wingMat.set_diffuse_color(np.array([1,1,1]))
# wingMat.set_roughness(1)
# wingMat.set_metalness(1)
# ------------------------------------------------------------------------ #
# cam.set_canvas()
# cam.set_photon_mapping_nn(100)
# cam.set_photon_mapping_max_radius(1)
# # # # # # # cam.trace_primary_rays(tracer, rkf, wingMesh, fluidMesh, light, wingMat)
# cam.apply_tone_mapping()
# cam.apply_gamma_correction()
# cam.synthesize_image()
# cam.plot_synthetic_image()
# pdb.set_trace()














