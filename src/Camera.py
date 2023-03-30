import vtk
import numpy as np
import pdb
import math
import pyvista as pv
import time
import tkinter as tk

import RayTracer
import Light
import Mesh
import Material
import Brdf 


class Camera():
    def __init__(self):
        self.photonMap = pv.PolyData()
        self.canvas     = vtk.vtkImageCanvasSource2D()
        
    def set_view_plane_distance(self, d):
        self.vpDist = d
        
    def set_horizontal_resolution(self, num):
        self.horRes = num
        
    def set_vertical_resolution(self, num):
        self.verRes = num
    
    def set_number_of_samples(self, num):
        self.nSamples = num
    
    def set_pixel_size(self, horNum, verNum):
        self.horPixSize = horNum
        self.verPixSize = verNum
        
    def set_eye(self, point):
        self.eye = np.array(point)
    
    def set_look_at(self, point):
        self.lookAt = np.array(point)
    
    def set_up_vector(self, vector):
        self.up = np.array(vector)
    
    def compute_uvw(self):
        
        w       = self.eye - self.lookAt
        self.w  = w/np.linalg.norm(w)

        u       = np.cross(self.up, self.w)
        self.u  = u/np.linalg.norm(u)
        
        v       = np.cross(self.w, self.u)
        self.v  = v/np.linalg.norm(v)

        # # Camera looking vertically down
        # if ((self.eye[0] == self.lookAt[0]) and (self.eye[2] == self.lookAt[2]) and (self.eye[1] > self.lookAt[1])):
            # self.u = np.array([0, 0, 1])
            # self.v = np.array([1, 0, 0])
            # self.w = np.array([0, 1, 0])
            
        # # Camera looking vertically up
        # if ((self.eye[0] == self.lookAt[0]) and (self.eye[2] == self.lookAt[2]) and (self.eye[1] < self.lookAt[1])):
            # self.u = np.array([1, 0, 0])
            # self.v = np.array([0, 0, 1])
            # self.w = np.array([0, -1, 0])
            
    def set_canvas(self):
        self.canvas.SetScalarTypeToUnsignedChar()
        self.canvas.SetNumberOfScalarComponents(3)
        self.canvas.SetExtent(0, self.horRes, 0, self.verRes, 0, 0)

       
    def set_photon_mapping_nn(self, nn):
        self.numOfNearPhotons = nn
        
    def set_photon_mapping_max_radius(self, r):
        self.maxDistance = r
        
    def set_linearity_opt(self, opt):
        self.linearityOpt = opt
        
    def set_vp_width(self, w):
        self.vpWidth = w
        
    def set_vp_height(self, h):
        self.vpHeight = h

    def trace_primary_rays(self, tracer, rkf, wingMesh, fluidMesh, light, wingMat, gui):
        
        self.HDRRadiance        = np.zeros((self.horRes, self.verRes, 3))
        # self.visualPoints = np.zeros((self.horRes, self.verRes, 2)) 

        # Nonlinear Ray Tracing
        if (self.linearityOpt.lower() == "nonlinear"): 
            count = 0
            for v in range(0, self.verRes-1, 1):
                for h in range(0, self.horRes-1, 1):
                    count += 1
                    print(count)
                    # pdb.set_trace()
                    # gui.root.after(5000, gui.update_progress_bar)
                    
                    origin     = np.zeros(3)
                    direction  = np.zeros(3)
                    
                    # Viewing System Coord.
                    origin[0]   = self.horPixSize * (h - (self.horRes / 2) + 0.5)
                    origin[1]   = self.verPixSize * (v - (self.verRes / 2) + 0.5)
                    origin[2]   = -self.vpDist
                    
                    # World Coord.
                    direction   = np.multiply(origin[0], self.u) + np.multiply(origin[1], self.v) + np.multiply(origin[2], self.w)
                    direction   = direction/np.linalg.norm(direction)

                    l           = math.sqrt( (self.vpDist**2) + (origin[0]**2 + origin[1]**2) )
                    origin      = self.eye + l*direction
                    
                    source      = origin
                    t           = 10**10
                    target      = origin + t*direction
                    pt          = vtk.vtkPoints()
                    cellId      = vtk.vtkIdList()
                        
                    tmp         = wingMesh.obbtree.IntersectWithLine(source, target, pt, cellId)
                    
                    self.hitPoint = np.array([])
                    # Only nonlinearily trace if there is a great change of ray hitting the wing
                    if tmp != 0:
                        tracer.nonlinear_ray_trace("shockshadow", "backward", rkf, fluidMesh, wingMesh, origin, direction, camera=self)

                    # Photon Mapping
                    if (self.hitPoint.size != 0):
                        nearPhotons     = vtk.vtkIdList()
                        idxNearPhotons  = []
                        distNearPhotons = []
                            
                        light.kdTreePhotonMap.FindClosestNPoints(self.numOfNearPhotons, self.hitPoint.reshape(3), nearPhotons)
                            
                        for i in range(nearPhotons.GetNumberOfIds()):
                            id          = nearPhotons.GetId(i)
                            nearPhoton  = np.array(light.kdTreePhotonMap.GetDataSet().GetPoint(id))
                            d           = np.linalg.norm(nearPhoton - self.hitPoint)

                            if (d <= self.maxDistance):
                                idxNearPhotons.append(id)
                                distNearPhotons.append(d)
                                    
                        if idxNearPhotons:
                            for idx in idxNearPhotons:                              
                                microfacet      =   Brdf.BRDF(material=wingMat, surfaceNormal   =   light.photonMap['Normal'][idx,:],\
                                                                                lightDirection  =   -light.photonMap['Direction'][idx,:],\
                                                                                viewDirection   =   -direction)
                                microfacet.set_value()
                                
                                self.HDRRadiance[h,v,:] += (microfacet.fr)*(light.photonMap['Power'][idx,:])
                                
                            self.HDRRadiance[h,v,:] /= (math.pi)*(max(distNearPhotons)**2)

        # Linear Ray Tracing
        elif (self.linearityOpt.lower() == "linear"):
            count = 0
            for v in range(0, self.verRes-1, 1):
                for h in range(0, self.horRes-1, 1):
                    count += 1
                    print(count)
                    # gui.progressEntry.delete(0, 'end')
                    # gui.progressEntry.insert('end', str((count/(self.horRes*self.verRes))*100.0) + '%')
                
                    origin     = np.zeros(3)
                    direction  = np.zeros(3)
                    
                    # Viewing System Coord.
                    origin[0]   = self.horPixSize * (h - (self.horRes / 2) + 0.5)
                    origin[1]   = self.verPixSize * (v - (self.verRes / 2) + 0.5)
                    origin[2]   = -self.vpDist
                    
                    # World Coord.
                    direction   = np.multiply(origin[0], self.u) + np.multiply(origin[1], self.v) + np.multiply(origin[2], self.w)
                    direction   = direction/np.linalg.norm(direction)

                    l           = math.sqrt( (self.vpDist**2) + (origin[0]**2 + origin[1]**2) )
                    origin      = self.eye + l*direction
                    

                    # trace ray
                    self.hitPoint = np.array([])
                    tracer.linear_ray_trace("shockshadow", "backward", rkf, fluidMesh, wingMesh, origin, direction, camera=self)

                    # Photon Mapping
                    if (self.hitPoint.size != 0):
                        nearPhotons     = vtk.vtkIdList()
                        idxNearPhotons  = []
                        distNearPhotons = []
                            
                        light.kdTreePhotonMap.FindClosestNPoints(self.numOfNearPhotons, self.hitPoint.reshape(3), nearPhotons)
                            
                        for i in range(nearPhotons.GetNumberOfIds()):
                            id          = nearPhotons.GetId(i)
                            nearPhoton  = np.array(light.kdTreePhotonMap.GetDataSet().GetPoint(id))
                            d           = np.linalg.norm(nearPhoton - self.hitPoint)

                            if (d <= self.maxDistance):
                                idxNearPhotons.append(id)
                                distNearPhotons.append(d)
                                    
                        if idxNearPhotons:
                            for idx in idxNearPhotons:
                                microfacet      =   Brdf.BRDF(material=wingMat, surfaceNormal   =   light.photonMap['Normal'][idx,:],\
                                                                                lightDirection  =   -light.photonMap['Direction'][idx,:],\
                                                                                viewDirection   =   -direction)
                                microfacet.set_value()
                                
                                self.HDRRadiance[h,v,:] += (microfacet.fr)*(light.photonMap['Power'][idx,:])
                                
                            self.HDRRadiance[h,v,:] /= (math.pi)*(max(distNearPhotons)**2)

                        # self.visualPoints[h,v,0] = self.hitPoint[0]
                        # self.visualPoints[h,v,1] = self.hitPoint[1]
                        # self.visualPoints[h,v,2] = self.hitPoint[2]

        
    def add_photon_to_map(self, pos, dir):
        buff                = pv.PolyData(pos)
        buff['Direction']   = dir.reshape((1,3))
        
        apd = vtk.vtkAppendPolyData()
        apd.AddInputData(self.photonMap)
        apd.AddInputData(buff)
        apd.Update()
        
        self.photonMap = pv.PolyData(apd.GetOutput())    
    
    def srgb_to_xyz(self, triplet):
        # (linear) sRGB to CIE XYZ
        srgb2xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
        
        for i in range(triplet.shape[0]):
            for j in range(triplet.shape[1]):
                triplet[i,j,0] = np.matmul(srgb2xyz[0,:], triplet[i,j,:])
                triplet[i,j,1] = np.matmul(srgb2xyz[1,:], triplet[i,j,:])
                triplet[i,j,2] = np.matmul(srgb2xyz[2,:], triplet[i,j,:])

    def xyz_to_srgb(self, triplet):
        # (linear) CIE XYZ to sRGB
        xyz2srgb = np.array([[3.2405, -1.5371, -0.4985], [-0.9693, 1.8706, 0.0416], [0.0556, -0.2040, 1.0572]])
        
        for i in range(triplet.shape[0]):
            for j in range(triplet.shape[1]):
                triplet[i,j,0] = np.matmul(xyz2srgb[0,:], triplet[i,j,:])
                triplet[i,j,1] = np.matmul(xyz2srgb[1,:], triplet[i,j,:])
                triplet[i,j,2] = np.matmul(xyz2srgb[2,:], triplet[i,j,:])
                
    def xyz_to_luv(self, triplet, whitePoint):
        for i in range(triplet.shape[0]):
                for j in range(triplet.shape[1]):
                    X           = triplet[i,j,0]
                    Y           = triplet[i,j,1]
                    Z           = triplet[i,j,2]
                    Xn          = whitePoint[0]
                    Yn          = whitePoint[1]
                    Zn          = whitePoint[2]
                    ref         = Y/Yn
                    
                    if (X != 0 or Y != 0 or Z != 0):
                        u           = (4*X)/(X + (15*Y) + (3*Z))
                        v           = (9*Y)/(X + (15*Y) + (3*Z))
                        un          = (4*Xn)/(Xn + (15*Yn) + (3*Zn))
                        vn          = (9*Yn)/(Xn + (15*Yn) + (3*Zn))
                        
                        Lstar       = ( ((841.0/108.0)*ref+(4.0/29.0)) if (ref <= 0.008856) else (ref**(1/3)))
                        ustar       = 13*Lstar*(u-un)
                        vstar       = 13*Lstar*(v-vn)
                        
                        triplet[i,j,0] = Lstar
                        triplet[i,j,1] = ustar
                        triplet[i,j,2] = vstar
    
    def apply_tone_mapping(self, triplet, keyValue):
        # to avoid singularity of log(0)
        delta               = 0.00000001 
        logAverageRadiance  = np.exp(np.average(np.log(delta+triplet),axis=(0,1)))
        
        #Key-value: [0.09,0.045] low, 0.18 normal, [0.36,0.72] high
        scaledRadiance = (keyValue/logAverageRadiance)*triplet
        
        triplet  = (scaledRadiance) / (1+scaledRadiance)
    
    def apply_gamma_encoding(self, triplet):
        # (nonlinear) sRGB
        f       = 0.055
        gamma   = 1.0/2.4
        s       = 12.92
        t       = 0.0031308

        for i in range(triplet.shape[0]):
            for j in range(triplet.shape[1]):
                triplet[i,j,0] = ((1+f)*(triplet[i,j,0]**gamma)-f) if (triplet[i,j,0]>t) else (s*triplet[i,j,0])
                triplet[i,j,1] = ((1+f)*(triplet[i,j,1]**gamma)-f) if (triplet[i,j,1]>t) else (s*triplet[i,j,1])
                triplet[i,j,2] = ((1+f)*(triplet[i,j,2]**gamma)-f) if (triplet[i,j,2]>t) else (s*triplet[i,j,2])        
    
    def apply_gamma_correction(self, triplet):
        gamma = 2.2
        triplet = triplet**(1/gamma)            
        
    
                
    def synthesize_image(self):
        for v in range(0, self.verRes-1, 1):
            for h in range(0, self.horRes-1, 1):   
                self.canvas.SetDrawColor(self.pixColor[h,v,0]*255.0,self.pixColor[h,v,1]*255.0,self.pixColor[h,v,2]*255.0)
                self.canvas.FillBox(h, h+1, v, v+1)
        self.canvas.Update()
        
    def plot_synthetic_image(self):
       
        # Convert the image to a polydata
        imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
        imageDataGeometryFilter.SetInputConnection(self.canvas.GetOutputPort())
        imageDataGeometryFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(imageDataGeometryFilter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(1)

        # Setup rendering
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(1,1,1)
        renderer.ResetCamera()

        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(800, 800)

        renderWindowInteractor = vtk.vtkRenderWindowInteractor()

        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.Initialize()
        renderWindowInteractor.Start()
                
                




