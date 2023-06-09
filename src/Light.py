# Import external libraries
import pyvista              as pv
import numpy                as np
import scipy.integrate      as integrate
import scipy.interpolate    as interpolate
from scipy                  import signal
import vtk
import pdb
import math

# Import classes
import Mesh

class Light():
# ------------------------------------------------------------------------ #
# Class constructor
# ------------------------------------------------------------------------ #  
    def __init__(self):
        self.photonMap      = pv.PolyData()
        self.totalPower     = 0
        self.numOfPhotons   = 0
        self.GDC            = 0.0002256
        
# ------------------------------------------------------------------------ #
# Light source spectral power distribution
# ------------------------------------------------------------------------ #          
    def sun_spectral_irradiance(self, wavelength):
        # nm to mum
        wavelength = wavelength*(10**-3)
        
        # Sun's color temperature
        T = 5777
        # Mean Sun-Earth distance
        r0 = 149547890000 # [m]
        # Mean radius of the solar disk
        rS = 6.9598*(10**8)
        
        # # Planck's constant [W.s^2]
        # h  = 6.6260693*(10**-34)
        # # speed of light in vacuum [m/s]
        # c  = 2.99792458*(10**8)
        # # Boltzmann's constant [J/K]
        # k  = 1.380658*(10**-23)
        # First radiation constant
        C1 = 3.7427*(10**8) # [W mum^4 m^-2]
        # C1 = 2*np.pi*h*(c**2)
        # Second radiation constant
        C2 = 1.4388*(10**4) # [mum K]
        # C2 = h*c/k
        
        num = C1
        den = (wavelength**5)*(np.exp(C2/(wavelength*T))-1)
        
        # (W/m2)/mum
        spectralPower = (num/den)*((rS/r0)**2)

        # (W/m2)/nm
        spectralPower = spectralPower*(10**-3)
        
        return spectralPower
        
# ------------------------------------------------------------------------ #
# XYZ color matching functions
# ------------------------------------------------------------------------ #           
    def get_x_color_matching_function(self, wavelength):
        wl = wavelength
        #gamma      beta             delta
        sx0=(0.0624) if (wl<442)   else (0.0374)
        sx1=(0.0264) if (wl<599.8) else (0.0323)
        sx2=(0.0490) if (wl<501.1) else (0.0382)
        #      alpha                  beta
        x0 =  0.362*np.exp(-0.5*((wl-442.0)*sx0)**2)
        x1 =  1.056*np.exp(-0.5*((wl-599.8)*sx1)**2)
        x2 = -0.065*np.exp(-0.5*((wl-501.1)*sx2)**2)
    
        x = x0 + x1 + x2
        return x
        
    def get_y_color_matching_function(self, wavelength):    
        wl = wavelength
        #gamma      beta             delta
        sy0=(0.0213) if (wl<568.8) else (0.0247)
        sy1=(0.0613) if (wl<530.9) else (0.0322)
        #      alpha                  beta
        y0 =   0.821*np.exp(-0.5*((wl-568.8)*sy0)**2)
        y1 =   0.286*np.exp(-0.5*((wl-530.9)*sy1)**2)

        y = y0 + y1
        return(y)
        
    def get_z_color_matching_function(self, wavelength):          
        wl = wavelength
        #gamma      beta             delta
        sz0=(0.0845) if (wl<437.0) else (0.0278)
        sz1=(0.0385) if (wl<459.0) else (0.0725)
        #      alpha                  beta
        z0 =   1.217*np.exp(-0.5*((wl-437.0)*sz0)**2)
        z1 =   0.681*np.exp(-0.5*((wl-459.0)*sz1)**2)
        
        z = z0 + z1
        return(z)
# ------------------------------------------------------------------------ #
# XYZ to sRGB color space conversion
# ------------------------------------------------------------------------ #                   
    def set_total_rgb_power(self, type='sun', lambdaLow=380, lambdaHigh=800, lambdaDisc=10):
        irradiance_function_name    = type + '_spectral_irradiance'
        irradiance_function         = getattr(self, irradiance_function_name)
        
        wavelength                  = np.linspace(int(lambdaLow), int(lambdaHigh), int((lambdaHigh-lambdaLow)/lambdaDisc))
        spectralPower               = np.zeros((wavelength.shape[0]))
        spectralX                   = np.zeros((wavelength.shape[0]))
        spectralY                   = np.zeros((wavelength.shape[0]))
        spectralZ                   = np.zeros((wavelength.shape[0]))
        for i in range(0, wavelength.shape[0]):
            spectralPower[i]        = irradiance_function(wavelength[i])
            spectralX[i]            = self.get_x_color_matching_function(wavelength[i])
            spectralY[i]            = self.get_y_color_matching_function(wavelength[i])
            spectralZ[i]            = self.get_z_color_matching_function(wavelength[i])
            
        xPower      = integrate.simps(spectralPower*spectralX, wavelength)
        yPower      = integrate.simps(spectralPower*spectralY, wavelength)
        zPower      = integrate.simps(spectralPower*spectralZ, wavelength)
        xyzPower    = np.array([xPower, yPower, zPower])

        xyz2srgb    = np.array([[3.2405, -1.5371, -0.4985], [-0.9693, 1.8706, 0.0416], [0.0556, -0.2040, 1.0572]])
        # xyz2rgb = np.array([[2.5623, -1.1661, -0.3962], [-1.0215, 1.9778, 0.0437], [0.0725, -0.2562, 1.1810]])

        rgbPower = np.matmul(xyz2srgb, xyzPower)*self.areaOfProj
        
        # clamp to zero
        for c in range(0, 3):
            if (rgbPower[c]<0.0):
                rgbPower[c]=0.0
        
        self.totalPower = rgbPower
        
# ------------------------------------------------------------------------ #
# Gladstone-Dale constant linear interpolation from tabulated spectral data
# ------------------------------------------------------------------------ #  
    def set_gladstone_dale_constant(self, lambdaLow=380, lambdaHigh=800, lambdaDisc=10):
        K           = np.array([0.2239, 0.2250, 0.2259, 0.2274, 0.2304, 0.2330])
        K           *= (10**-3)
        spectrum    = np.array([912.5, 703.4, 607.4, 509.7, 407.9, 356.2])
        
        gdc         = interpolate.interp1d(spectrum, K)
        
        wavelength  = np.linspace(int(lambdaLow), int(lambdaHigh), int((lambdaHigh-lambdaLow)/lambdaDisc))
        spectralGDC = gdc(wavelength)
        
        self.GDC    = np.mean(spectralGDC)
        
# ------------------------------------------------------------------------ #
# Light source's number of photons
# ------------------------------------------------------------------------ #          
    def set_number_of_photons(self, num):
        self.numOfPhotons = num

# ------------------------------------------------------------------------ #
# Light source's direction
# ------------------------------------------------------------------------ #          
    def set_direction(self, wingMesh, sunAzimuth, sunElevation, sunRange):
        # Onera M6-wing's quarter-chord line
        cr4                 = np.array([0.201475, 0,      0])
        ct4                 = np.array([0.803905, 1.1963, 0])
        
        # Rotation matrix
        rotZ                = np.array([[np.cos(sunAzimuth*np.pi/180.0), -np.sin(sunAzimuth*np.pi/180.0),0], [np.sin(sunAzimuth*np.pi/180.0), np.cos(sunAzimuth*np.pi/180.0),0], [0,0,1]])
        rotX                = np.array([[1,0,0], [0,np.cos(sunElevation*np.pi/180.0), -np.sin(sunElevation*np.pi/180.0)],[0,np.sin(sunElevation*np.pi/180.0), np.cos(sunElevation*np.pi/180.0)]])

        # Rotate quarter-chord line
        aziRot              = np.matmul(rotZ, ct4-cr4)+cr4
        elevRot             = np.matmul(rotX, aziRot-cr4)+cr4

        # Illumination direction
        self.direction      = (cr4-elevRot)/np.linalg.norm(cr4-elevRot)
        
        # Wing's projection surface in the illumination direction
        self.dirOfProj      = -self.direction
        self.orgOfProj      = wingMesh.grid.center + sunRange*self.dirOfProj
        self.surfOfProj     = pv.PolyData(wingMesh.grid.points).project_points_to_plane(origin=self.orgOfProj, normal=self.dirOfProj).delaunay_2d()
        self.areaOfProj     = self.surfOfProj.area
      
# ------------------------------------------------------------------------ #
# Light source's origin(s)
# ------------------------------------------------------------------------ #          
    def set_origin(self, wingMesh):
        edge = vtk.vtkFeatureEdges()
        edge.SetInputData(self.surfOfProj)
        edge.FeatureEdgesOff()
        edge.NonManifoldEdgesOff()
        edge.Update()
        
        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(wingMesh.grid)
        geomFilter.Update()
        
        # Create a plane discretised with the required number of points = no. of photons at the center of the wing
        res     = int(np.floor(np.sqrt(self.numOfPhotons)))
        plane   = pv.Plane(center=wingMesh.grid.center, direction=[0,0,1], i_size=1.2, j_size=1.2, i_resolution=res, j_resolution=res)
        
        # Slice wing with the plane
        plane.compute_implicit_distance(geomFilter.GetOutput(), inplace=True)
        inner   = pv.PolyData(plane.points[np.argwhere(plane['implicit_distance'] < 0)[:,0],:])
        
        # Project intersection of plane with wign to the origin and direction of illumination
        inProj  = inner.project_points_to_plane(origin=self.orgOfProj, normal=self.dirOfProj)
        
        # Photons' origins and direction
        self.photonsOrg = inProj.points
        self.photonsDir = np.ones((inProj.points.shape[0], inProj.points.shape[1]))*self.direction
        
        # Visualisation

        
        # mesh  = pv.StructuredGrid(self.photonsOrg[:,0],self.photonsOrg[:,1],self.photonsOrg[:,2])
        # mesh["direction"] = self.photonsDir
        # mesh["magnitude"] = np.ones((inProj.points.shape[0]))*0.1
        # glyphs = mesh.glyph(orient="direction", scale="magnitude", geom=pv.Arrow(shaft_radius=0.01, tip_radius=0.03, tip_length=0.2, tip_resolution=100))
       
        # plotter = pv.Plotter()
        # plotter.add_mesh(glyphs, color="yellow")
        # plotter.add_mesh(wingMesh.grid, color='green', opacity=0.3)
        # plotter.add_mesh(pv.PolyData(edge.GetOutput()), color='magenta', show_edges=True)
        # plotter.add_mesh(pv.PolyData(self.photonsOrg), color='blue', show_edges=True)
        # plotter.set_background('white')
        # plotter.show()
        # pdb.set_trace()
# ------------------------------------------------------------------------ #
# Single light ray origin and direction
# ------------------------------------------------------------------------ #       
    def set_primary_ray(self, origin, direction):
        self.set_number_of_photons(1)
        self.photonsOrg = origin
        self.photonsDir = direction

# ------------------------------------------------------------------------ #
# Add traced light ray to photon map
# ------------------------------------------------------------------------ #       
    def add_photon_to_map(self, pos, dir, pow, normal):
        buff                = pv.PolyData(pos)
        buff['Direction']   = dir.reshape((1,3))
        buff['Power']       = pow.reshape((1,3))
        buff['Normal']      = normal.reshape((1,3))
        
        apd = vtk.vtkAppendPolyData()
        apd.AddInputData(self.photonMap)
        apd.AddInputData(buff)
        apd.Update()
        
