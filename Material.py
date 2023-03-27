import numpy as np


class Material():
    def __init__(self):
        self.diffuseCoeff   = 0
        self.diffuseColor   = np.array([1,1,1])
        self.roughness      = 1
        self.metalness      = 1


    def set_diffuse_coefficient(self, diffuseCoeff):
        # Ks (Kd + Ks = 1 | Ks - percentage of light "refracted" within surface and re-emitted)
        self.diffuseCoeff   = diffuseCoeff

    def set_diffuse_color(self, diffuseColor):
        self.diffuseColor   = diffuseColor

    def set_roughness(self, roughness):
        # Alpha (0 - rough | 1 - smooth)
        self.roughness = roughness
        
    def set_metalness(self, metalness):
        # M (0 - dielectric (non-metal) | 1 - metal
        self.metalness = metalness








