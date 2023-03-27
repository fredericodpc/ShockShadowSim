import numpy as np
import math
import pdb

class BRDF():
    def __init__(self, material, surfaceNormal, lightDirection, viewDirection):
        self.material       = material
        self.surfaceNormal  = surfaceNormal
        self.lightDirection = lightDirection
        self.viewDirection  = viewDirection


    def get_halfway_direction(self):
        self.halfwayDirection = (self.lightDirection + self.viewDirection)/np.linalg.norm((self.lightDirection + self.viewDirection))

    def get_normal_distribution_function(self):
        # n     - surface normal vector
        n       = self.surfaceNormal
        # h     - halfway vector between light and viewing directions
        h       = self.halfwayDirection
        # alpha - surface roughness factor
        alpha   = self.material.roughness
        # D     - normal distribution function - Trowbridge-Reitz GGX
        
        alphaSqrd   = alpha**2
        dotNH       = abs(np.dot(n,h))
        dotNHSqrd   = dotNH**2

        nom         = alphaSqrd
        denom       =  ( (dotNHSqrd)*(alphaSqrd -1) +1)
        denom       = math.pi * denom * denom
        
        self.D      = nom/denom

    def get_geometry_function(self):
        # k     - remapping of roughness factor considering direct lighting
        # alpha - surface roughness factor
        alpha   = self.material.roughness
        # n     - surface normal vector
        n       = self.surfaceNormal
        # v     - viewing direction vector
        v       = self.viewDirection
        # l     - light direction vector
        l       = self.lightDirection
        # G     - Geometry function - Schlick-Beckmann approx. combined w/ GGX
        
        k       = ((alpha + 1)**2)/8

        dotNV   = abs(np.dot(n,v))
        dotNL   = abs(np.dot(n,l))

        
        nom1    = dotNV
        denom1  = dotNV*(1-k) + k
        
        nom2    = dotNL
        denom2  = dotNL*(1-k) + k

        G1      = nom1/denom1
        G2      = nom2/denom2
        
        self.G       = G1*G2

    def get_fresnel_function(self):
        # m     - "metalness" of a surface [0,1]
        m       = self.material.metalness
        # h     - halfway vector between light and viewing directions
        h       = self.halfwayDirection
        # v     - viewing direction vector
        v       = self.viewDirection
        # F     - Fresnel function - Fresnel-Schlick approx.
        
        # # RGB - linear
        # F0alluminium    = [0.91, 0.92, 0.92];
        # sRGB
        F0alluminium    = [0.96, 0.96, 0.97];
        # F0gold          = [1.00, 0.86, 0.57];
        F0dielectric    = [0.04, 0.04, 0.04];
        
        F0 = []
        for i in range(0,3,1):
            F0.append(np.interp(m, [0,1], [F0dielectric[i], F0alluminium[i]]))

        dotHV = np.dot(h,v)
        
        self.F = np.zeros((3))
        i = 0
        for element in F0:
            self.F[i] = (element + (1-element)*((1-dotHV)**5))
            i += 1

    def set_value(self):
        frDiffuse       = np.multiply(self.material.diffuseCoeff,self.material.diffuseColor)/math.pi

        self.get_halfway_direction()
        self.get_normal_distribution_function()
        self.get_geometry_function()
        self.get_fresnel_function()

        nom             = self.D*self.F*self.G
        dotWON          = abs(np.dot(self.viewDirection, self.surfaceNormal))
        dotWIN          = abs(np.dot(self.lightDirection, self.surfaceNormal))
        denom           = 4*dotWON*dotWIN
        frSpecular      = nom/denom
        
        self.fr         = frDiffuse + frSpecular
        
        






















