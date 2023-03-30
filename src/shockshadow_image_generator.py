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
import Brdf 


from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.special import comb
import scipy.interpolate as interpolate

from PIL import Image
from PIL import ImageEnhance
from PIL import ImageCms

from colorspacious import cspace_convert as ColorSpaceConvert
from colorspacious import CAM02LCD
from colorspacious import CIECAM02Space
from colorspacious import LuoEtAl2006UniformSpace


import matplotlib.pyplot as plt
import cv2 as cv

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #
sunAzimuth     = 00
sunElevation   = 135
sunRange       = 1
numOfPhotons   = 1565101

numNN          = 10000
maxRadius      = 0.005

viewNumber     = 2
numOfPixel     = 10000


kd              = 1.0
cd              = [1.0,1.0,1.0]
rough           = 0.0
metal           = 1.0
# ------------------------------------------------------------------------ #
cwd = os.getcwd()

os.chdir(cwd + "\photon_map_database")
       
fileNamePKL = str(viewNumber) + '_view_' + str(numOfPixel) + '_pixels.PKL'
reader = open(fileNamePKL, 'rb')
cam = pickle.load(reader)
reader.close()

# ------------------------------------------------------------------------ #
cam.set_photon_mapping_nn(numNN)
cam.set_photon_mapping_max_radius(maxRadius)
# ------------------------------------------------------------------------ #

fileNameVTP = str(sunElevation) + '_ele_' + str(sunAzimuth) + '_azi_' + str(numOfPhotons) + '_photons.vtp'
reader      = DataManager.DataManager(fileNameVTP)
photonMap   = reader.read_vtp()
light       = Light.Light()
light.set_photon_map(photonMap)

os.chdir(cwd)

# ------------------------------------------------------------------------ #
wingMat     = Material.Material()
wingMat.set_diffuse_coefficient(kd)
wingMat.set_diffuse_color(np.array([float(cd[0]),float(cd[1]),float(cd[2])]))
wingMat.set_roughness(rough)
wingMat.set_metalness(metal)
# ------------------------------------------------------------------------ #
# Photon Mapping
cam.HDRRadiance = np.zeros((cam.horRes, cam.verRes, 3))
for v in range(0, cam.verRes, 1):
    for h in range(0, cam.horRes, 1):
        
        pixID = h + v*cam.horRes
        
        print(pixID)
        
        for n in range(len(cam.photonMap[pixID]['shotPos'])):
            L = np.zeros(3)

            if (len(cam.photonMap[pixID]['hitPos'][n]) != 0):
                nearPhotons     = vtk.vtkIdList()
                idxNearPhotons  = []
                distNearPhotons = []
                    
                try:
                    light.kdTreePhotonMap.FindClosestNPoints(cam.numOfNearPhotons, cam.photonMap[pixID]['hitPos'][n], nearPhotons)
                except:
                    light.kdTreePhotonMap.FindClosestNPoints(cam.numOfNearPhotons, cam.photonMap[pixID]['hitPos'][n][0], nearPhotons)
                    
                for i in range(nearPhotons.GetNumberOfIds()):
                    id          = nearPhotons.GetId(i)
                    nearPhoton  = np.array(light.kdTreePhotonMap.GetDataSet().GetPoint(id))
                    d           = np.linalg.norm(nearPhoton - cam.photonMap[pixID]['hitPos'][n])

                    if (d <= cam.maxDistance):
                        idxNearPhotons.append(id)
                        distNearPhotons.append(d)

                if idxNearPhotons:
                    idxNearPhotons  = np.array(idxNearPhotons)
                    distNearPhotons = np.array(distNearPhotons)

                    for j in range(idxNearPhotons.shape[0]):                 
                        idx = idxNearPhotons[j]
                        # # # microfacet      =   Brdf.BRDF(material=wingMat, surfaceNormal   =   light.photonMap['Normal'][idx,:],\
                                                                        # # # lightDirection  =   -light.photonMap['Direction'][idx,:],\
                                                                        # # # viewDirection   =   -direction)
                        # # # microfacet.set_value()
                        # # # fr = microfacet.fr
                        
                        # Lambertian surface
                        fr = (wingMat.diffuseColor)/np.pi
                        
                        # No filter
                        L += (fr)*(light.photonMap['Power'][idx,:])
                    L /= (np.pi)*(np.max(distNearPhotons)**2)

                        # # # # Gaussian filter
                        # # # alpha   = 0.918
                        # # # beta    = 1.953
                        # # # term1   = 1-np.exp(-beta*(distNearPhotons[j]**2)/(2*(np.max(distNearPhotons)**2)))
                        # # # term2   = 1-np.exp(-beta)
                        # # # wpg     = alpha*(1-(term1/term2))
                        
                        # # # L += (fr)*(light.photonMap['Power'][idx,:])*wpg

                    # # # L /= (np.pi)*(np.max(distNearPhotons)**2)

            cam.HDRRadiance[h,v,:] += L
            
        cam.HDRRadiance[h,v,:] /= cam.nSamples     



cwd = os.getcwd()
os.chdir(cwd + "\photon_map_database")
fileNameNPY = str(viewNumber) + '_view_' + str(numOfPixel) + '_pixels_' + str(sunElevation) + '_ele_' + str(sunAzimuth) + '_azi_' + str(numOfPhotons) + '_photons.npy'
np.save(fileNameNPY, cam.HDRRadiance)
pdb.set_trace()
pixColorSRGB            = np.load(fileNameNPY)
img                     = Image.fromarray(pixColorSRGB.astype(np.uint8), mode="RGB")
fileNameJPG = str(viewNumber) + '_view_' + str(numOfPixel) + '_pixels_' + str(sunElevation) + '_ele_' + str(sunAzimuth) + '_azi_' + str(numOfPhotons) + '_photons.jpg'
img.save(fileNameJPG)
pdb.set_trace()
# -------------------------------------------------------------------------------
























# srgb2xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
# pixColorXYZ = np.zeros((pixColorSRGB.shape[0],pixColorSRGB.shape[1],pixColorSRGB.shape[2]))
# for i in range(pixColorSRGB.shape[0]):
    # for j in range(pixColorSRGB.shape[1]):
        # pixColorXYZ[i,j,0] = np.matmul(srgb2xyz[0,:], pixColorSRGB[i,j,:])
        # pixColorXYZ[i,j,1] = np.matmul(srgb2xyz[1,:], pixColorSRGB[i,j,:])
        # pixColorXYZ[i,j,2] = np.matmul(srgb2xyz[2,:], pixColorSRGB[i,j,:])
        
       
# # hist, bin = np.histogram(pixColorXYZ[...,1], np.arange(int(np.min(pixColorXYZ[...,1])), int(np.max(pixColorXYZ[...,1]))+2))
# # ------------------- #
# # # 1 - RAW
# img         = Image.fromarray(pixColorSRGB.astype(np.uint8), mode="RGB")

# ss          = img.split()
# # select only the wing surface to generate the histogram from
# mask        = ss[2].point(lambda i: i != 255 and 255)
# # histogram   = img.histogram(mask)
# # l0          = histogram[0:256]
# # l0          = np.array(l0)
# # # select only existing colors in the histogram
# # l00         = np.where(l0 != 0)[0]
# # l0min       = np.min(l00)
# # l0max       = np.max(l00)
# # l1          = histogram[256:512]
# # l2          = histogram[512:768]


# out0 = ss[0].point(lambda i: i/2)
# out1 = ss[1].point(lambda i: i/2)
# out2 = ss[2].point(lambda i: i/2)
# ss[0].paste(out0, None, mask)
# ss[1].paste(out1, None, mask)
# ss[2].paste(out2, None, mask)
# img = Image.merge(img.mode, ss)
# img.show()
# pdb.set_trace() 
# # ------------------- #
# # 2 - LIGHTNESS
# srgbP       = ImageCms.createProfile("sRGB")
# labP        = ImageCms.createProfile("LAB")
# srgb2lab    = ImageCms.buildTransformFromOpenProfiles(srgbP, labP, "RGB", "LAB")
# lab2srgb    = ImageCms.buildTransformFromOpenProfiles(labP, srgbP, "LAB", "RGB")
# img2        = ImageCms.applyTransform(img, srgb2lab)
# histogram       = img2.histogram(mask)
# l0              = histogram[0:256]
# l0              = np.array(l0)
# # select only existing colors in the histogram
# l00             = np.where(l0 != 0)[0]
# l0min           = np.min(l00)
# l0max           = np.max(l00)
# # ------------------- #
# def bernstein_poly(i, n, t):
    # """
     # The Bernstein polynomial of n, i as a function of t
    # """

    # return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


# def bezier_curve(points, nTimes=1000):
    # """
       # Given a set of control points, return the
       # bezier curve defined by the control points.

       # points should be a list of lists, or list of tuples
       # such as [ [1,1], 
                 # [2,3], 
                 # [4,5], ..[Xn, Yn] ]
        # nTimes is the number of time steps, defaults to 1000

        # See http://processingjs.nihongoresources.com/bezierinfo/
    # """

    # nPoints = len(points)
    # xPoints = np.array([p[0] for p in points])
    # yPoints = np.array([p[1] for p in points])

    # t = np.linspace(0.0, 1.0, nTimes)

    # polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    # xvals = np.dot(xPoints, polynomial_array)
    # yvals = np.dot(yPoints, polynomial_array)

    # return xvals, yvals
    
# nPoints = 6
# points  = [[l0min,0], [l0min+10,0], [l0min+5, 255], [l0max-5, 0], [l0max-5, 255], [l0max, 255]]
# xpoints = [p[0] for p in points]
# ypoints = [p[1] for p in points]
# xvals, yvals = bezier_curve(points, nTimes=1000)


# plt.figure(0)
# ax0 = plt.gca()
# for i in range(0,256):
    # plt.bar(i, l0[i], color='r')
# ax0.set_yscale('log')
# ax0.set_xlim(l0min, l0max)    
# ax00 = ax0.twinx()
# ax00.plot(xvals, yvals, color='m')
# ax00.plot(xpoints, ypoints, 'ko')
# # ax00.plot(np.arange(l0min, l0max+1), 255*((l00-l0min)/(l0max-l0min)), color='m')
# ax00.grid()
# # plt.figure(1)
# # ax1 = plt.gca()
# # for i in range(0,256):
    # # plt.bar(i, l1[i], color='g')
# # ax1.set_yscale('log')
# # ax1.set_xlim(np.min(np.argwhere(np.array(l1) != 0)), np.max(np.argwhere(np.array(l1) != 0)))
    
# # plt.figure(2)
# # ax2 = plt.gca()
# # for i in range(0,256):
    # # plt.bar(i, l2[i], color='b')
# # ax2.set_yscale('log')
# # ax2.set_xlim(np.min(np.argwhere(np.array(l2) != 0)), np.max(np.argwhere(np.array(l2) != 0)))   
# plt.show()    

# toneCurve   = interpolate.interp1d(xvals, yvals, kind='cubic')

# channels    = img2.split()
# light       = channels[0]
# def try2map(inValue):
    # try:
        # out = toneCurve(inValue)
    # except:
        # out = 0
    
    # return out
# light   = light.point(lambda i: i/2)
# light   = light.point(lambda i: try2map(i))

# light.show()
# pdb.set_trace()
# # -------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------
# # -------------------------------------------------------------------------------
# # # # pdb.set_trace()


# pixColorSRGB        = np.load('test_radiance.npy')
# pixColorJCH         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "JCh")
# pixColorJ           = pixColorJCH[...,0]
# pixColorJ           = pixColorJ - np.mean(pixColorJ, axis=(0,1))
# pixColor            = pixColorJ
# pixColor            = (pixColor - np.min(pixColor, axis=(0,1)))/(np.max(pixColor, axis=(0,1)) - np.min(pixColor, axis=(0,1)))
# pixColor            = 255*(pixColor**(1/1.2))
# img                 = Image.fromarray(pixColor.astype(np.uint8), mode="L")
# img.show()


# # # # # imgJ                = Image.fromarray(pixColorJ.astype(np.uint8), mode="L")
# # # # # mask                = imgJ.point(lambda i: i != 0)
# # # # # enhc                = ImageEnhance.Contrast(mask)
# # # # # enhc.enhance(10).show()


# # # # # pixColor    = cam.HDRRadiance
# # # # # imgSRGB     = Image.fromarray(pixColor.astype(np.uint8), mode="RGB")
# # # # # mask        = imgSRGB.point(lambda i: i != 0)
# # # # # imgLAB      = mask.convert("LAB")






# # # # pixColorXYZ         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "XYZ100")
# # # # Y                   = pixColorXYZ[:,:,1]
# # # # Yn = 100.0
# # # # relLight            = Y/Yn
# # # # fRelLight           = np.where(relLight != 0, np.where(relLight <= 0.008856, (841.0/108.0)*relLight+(4.0/29.0), relLight**(1/3)), 0.0)
# # # # relLightStar        = np.where(relLight != 0, 116*(relLight) - 16, 0.0)
# # # # relLightStarDiff    = np.where(relLightStar != 0, relLightStar - np.mean(relLightStar, axis=(0,1), where=(relLightStar != 0)), 0.0)
# # # # pixColor            = relLightStarDiff
# # # # pixColor            = (pixColor - np.min(pixColor, axis=(0,1)))/(np.max(pixColor, axis=(0,1)) - np.min(pixColor, axis=(0,1)))
# # # # pixColor            = 255*(pixColor**(1/1.2))
# # # # img = Image.fromarray(pixColor.astype(np.uint8), mode="L")
# # # # img.show()





















# # # # pdb.set_trace()
# # # # # 
# # # # Yn = 100.0
# # # # pixColorSRGB        = cam.HDRRadiance
# # # # pixColorXYZ         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "XYZ100")
# # # # Y                   = pixColorXYZ[:,:,1]
# # # # # relLight            = (Y-np.mean(Y, axis=(0,1), where=(Y != 0)))/Yn
# # # # relLight            = Y/Yn
# # # # fRelLight           = np.where(relLight != 0, np.where(relLight <= 0.008856, (841.0/108.0)*relLight+(4.0/29.0), relLight**(1/3)), 0.0)
# # # # relLightStar        = np.where(relLight != 0, 116*(relLight) - 16, 0.0)
# # # # relLightStarDiff    = np.where(relLightStar != 0, relLightStar - np.mean(relLightStar, axis=(0,1), where=(relLightStar != 0)), 0.0)
# # # # # relLightStarDiff    = relLightStar
# # # # pixColor            = relLightStarDiff

# # # # pixColor            = (pixColor - np.min(pixColor, axis=(0,1)))/(np.max(pixColor, axis=(0,1)) - np.min(pixColor, axis=(0,1)))
# # # # pixColor            = 255*(pixColor**(1/1.2))
# # # # img = Image.fromarray(pixColor.astype(np.uint8), mode="L")
# # # # img = img.resize((800,800))
# # # # img.show()


# # # # pdb.set_trace()


# # # # # -
# # # # # pixColorSRGB        = cam.HDRRadiance
# # # # # pixColorXYZ         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "XYZ100")
# # # # # # pixColorXYZ[:,:,1] = pixColorXYZ[:,:,1] - np.mean(pixColorXYZ[:,:,1], axis=(0,1), where=(pixColorSRGB[:,:,1] != 0))
# # # # # pixColorSRGB       = ColorSpaceConvert(pixColorXYZ, "XYZ100", "sRGB255")
# # # # # pixColor = pixColorSRGB
# # # # # -
# # # # # pixColor = cam.HDRRadiance



# # # # pixColorSRGB        = cam.HDRRadiance
# # # # pixColorXYZ         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "XYZ100")
# # # # alpha               = 300
# # # # beta                = np.mean(pixColorXYZ[:,:,1], axis=(0,1), where=(pixColorSRGB[:,:,1] != 0))
# # # # sf                  = (1/alpha)*((1.219 + ((alpha/2)**0.4))/(1.219 + (beta**0.4)))**2.5
# # # # pixColorXYZ         = sf*pixColorXYZ
# # # # pixColorSRGB        = ColorSpaceConvert(pixColorXYZ, "XYZ100", "sRGB255")
# # # # pixColor            = pixColorSRGB
# # # # pixColor            = (pixColor - np.min(pixColor, axis=(0,1)))/(np.max(pixColor, axis=(0,1)) - np.min(pixColor, axis=(0,1)))
# # # # pixColor            = 255*(pixColor**(1/1.2))


# # # # pixColorJCH         = ColorSpaceConvert(pixColorSRGB, "sRGB255", "JCh")
# # # # pixColorJCH[:,:,0] *= 10
# # # # pixColorSRGB        = ColorSpaceConvert(pixColorJCH, "JCh", "sRGB255")
# # # # pixColor = pixColorSRGB

# # # # img = Image.fromarray(pixColor.astype(np.uint8), mode="L")
# # # # img = Image.fromarray(pixColor.astype(np.uint8), mode="RGB")
# # # # img = img.resize((800,800))
# # # # img.show()

# # # # pdb.set_trace()

# # # # # Representing Lightness (Lstar) as an sRGB gray scale 8bits
# # # # pixColor = (pixColor - np.min(pixColor, axis=(0,1)))/(np.max(pixColor, axis=(0,1)) - np.min(pixColor, axis=(0,1)))
# # # # pixColor = 255.0*(pixColor**(1/1.7))
# # # # img = Image.fromarray(pixColor.astype(np.uint8))
# # # # img = img.resize((800,800))
# # # # img.show()
# # # # pdb.set_trace()
# # # # # ------------------------------------------------------------------------ #
# # # # # # RGB 2 LAB
# # # # # srgbP = ImageCms.createProfile("sRGB")
# # # # # labP  = ImageCms.createProfile("LAB")
# # # # # srgb2lab = ImageCms.buildTransformFromOpenProfiles(srgbP, labP, "RGB", "LAB")
# # # # # lab2srgb = ImageCms.buildTransformFromOpenProfiles(labP, srgbP, "LAB", "RGB")
# # # # # imgLAB = ImageCms.applyTransform(img, srgb2lab)
# # # # # # ------------------------------------------------------------------------ #
# # # # # # Plot Lightness in gray scale
# # # # # L, A, B = imgLAB.split()
# # # # # # Pixel values
# # # # # pixMap = L.load()
# # # # # # Change lightness and get new pic 
# # # # # L.getextrema()
# # # # # Lnew = L.point(lambda l: l+10 if l>160 and l<200 else l)
# # # # # newImg = Image.merge("LAB", (Lnew, A, B))
# # # # # newImgRGB = ImageCms.applyTransform(newImg, lab2srgb)
# # # # # newImgRGB.show()
# # # # # ------------------------------------------------------------------------ #
# # # # # Contrast Enhancement
# # # # imgEnhcd = ImageEnhance.Contrast(img)
# # # # imgEnhcd.enhance(0.1).show()
# # # # pdb.set_trace()
# # # # # ------------------------------------------------------------------------ #
# # # # pixColorSRGB = cam.HDRRadiance
# # # # pixColorXYZ  = ColorSpaceConvert(pixColorSRGB, "sRGB255", "XYZ100")
# # # # pixColorLAB     = ColorSpaceConvert(pixColorSRGB, "XYZ100", {"name": "CIELab", "XYZ100_w": "D65"})
# # # # pixColorLAB[:,:,0] = pixColorLAB[:,:,0] - np.mean(pixColorLAB[:,:,0])
# # # # # pixColorSRGB    = ColorSpaceConvert(pixColorXYZ, "XYZ100", "sRGB1")
# # # # # pixColorSRGB    = (pixColorSRGB - np.min(pixColorSRGB, axis=(0,1)))/(np.max(pixColorSRGB, axis=(0,1)) - np.min(pixColorSRGB, axis=(0,1)))
# # # # # pixColorSRGB    = 255.0*(pixColorSRGB**(1/1.7))
# # # # img = Image.fromarray(pixColorLAB[:,:,0].astype(np.uint8), mode="L")
# # # # img = img.resize((800,800))
# # # # img.show()

# # # # pixColorLAB = ColorSpaceConvert(pixColorSRGB, "sRGB255", {"name": "CIELab", "XYZ100_w": "D65"})
# # # # pixColorCIECAM02 = ColorSpaceConvert(pixColorSRGB, "sRGB255", {"name": "CIECAM02"})
# # # # pixColorCAM02LCD = ColorSpaceConvert(pixColorSRGB, "sRGB255", {"name": "J'a'b'", "ciecam02_space": CIECAM02Space.sRGB, "luoetal2006_space": CAM02LCD})

# # # # # Lightness
# # # # img = Image.fromarray(pixColorCIECAM02.J.astype(np.uint8), mode="L")
# # # # img = img.point(lambda i: 10*i if i != 0 else i)
# # # # img.show()

# # # # # Brightness
# # # # img = Image.fromarray(pixColorCIECAM02.Q.astype(np.uint8), mode="L")
# # # # # img = img.point(lambda i: 10*i if i != 0 else i)
# # # # img.show()

# # # # # Luminance
# # # # img = Image.fromarray(pixColorXYZ[:,:,1].astype(np.uint8), mode="L")
# # # # img.show()

# # # # # Lightness (I guess
# # # # img = Image.fromarray(pixColorCAM02LCD[:,:,0].astype(np.uint8), mode="L")
# # # # img = img.point(lambda i: 10*i if i != 0 else i)
# # # # # ------------------------------------------------------------------------ #






















