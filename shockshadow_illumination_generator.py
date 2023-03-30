# Import external libraries
import  vtk
import  os
import  pdb
import  pyvista             as pv
import  numpy               as np
import  time
import  math
from    vtk.tk.vtkTkRenderWindowInteractor import vtkTkRenderWindowInteractor
from    vtk.util.numpy_support import vtk_to_numpy
import  scipy.integrate     as      integrate
from    scipy.interpolate   import  Rbf
from    matplotlib.colors   import  ListedColormap
import  matplotlib.pyplot   as      plt
from    scipy.integrate     import  odeint
from    scipy.integrate     import  solve_ivp

# Import classes
import Mesh
import Light
import DataManager


# ------------------------------------------------------------------------ #
# Intersection detection
# ------------------------------------------------------------------------ #
def wing_hit(t, state):
    
    x, y, z, dxdt, dydt, dzdt = state
        
    if (wingMesh.locate_point_in_out_grid(np.array([x,y,z])) == 0):
        return 1
    else:
        return 0

wing_hit.terminal   = True
wing_hit.direction  = 0        
    
# ------------------------------------------------------------------------ #
# (System of) ODE(s) definition 
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
    
    return derCurrentState
    
# ------------------------------------------------------------------------ #
# Intersection computation
# ------------------------------------------------------------------------ #
def get_light_intersection(photonMapDict, sol):
    
    # ***************************************************************** #
    # Get interpolated normal at linear hit point - Gouraud/Phong Shading
    # ***************************************************************** #
    
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
   
    # ************************************************************** #
    # Get nonlinear hit point - Bisection on the 7th-order interpolant
    # ************************************************************** #
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

# ------------------------------------------------------------------------ #
# (System of) ODE(s) solution 
# ------------------------------------------------------------------------ #
def nonlinear_ray_trace(photonMapDict, initialState, plotter):
    # Solve ODE
    sol = solve_ivp(fun=ray_ode, t_span=[0,10], y0=initialState, method='DOP853', dense_output=True, first_step=0.1, max_step=0.1, rtol=1*(10**-6), atol=1*(10**-6), events=wing_hit)
    
    # Point inside the wing (hit)
    if (sol.status == 1):
        get_light_intersection(photonMapDict, sol)

    light.add_photon_to_map(photonMapDict['pos'], photonMapDict['dir'], photonMapDict['pow'], photonMapDict['normal'])
    
# ------------------------------------------------------------------------ #
# CFD Mesh and Data
# ------------------------------------------------------------------------ #
cwd = os.getcwd()
os.chdir(cwd + "\onera_solution")

# CFD volume solution and mesh loading
fileName = 'flow.vtu'
fluidMesh = Mesh.Mesh()
fluidMesh.grid = pv.read(fileName)
    # Correcting from non-dimensional density
fluidMesh.grid['Density'] = (101325/(287*288.15))*fluidMesh.grid['Density']

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
# Mesh auxiliaries
# ------------------------------------------------------------------------ #
# Build cell locator for interpolation
fluidMesh.build_cell_locator()
# Build grid checker for intersection detection
wingMesh.build_grid_in_out_checker(1*(10**-20))
# Build obb tree for linear intersection computation
wingMesh.build_obb_tree()

# ------------------------------------------------------------------------ #
# Light Source Configuration
# ------------------------------------------------------------------------ #
light = Light.Light()

# Position and Direction
lsType          = 'sun'
lsazimuth         = 30
lsElevation       = 90
lsRange           = 1
light.set_direction(wingMesh, lsAzimuth, lsElevation, lsRange)
light.set_origin(wingMesh)

# Number of photons
lsNumOfPhotons    = 100
light.set_number_of_photons(lsNumOfPhotons)

# Spectrum
lsSpectrumBin     = [380, 780]
lsSpectrumDisc    = 1
light.set_total_rgb_power(type=lsType, lambdaLow=lsSpectrumBin[0], lambdaHigh=lsSectrumBin[1], lsSpectrumDisc=1)

# ------------------------------------------------------------------------ #
# Fluid-Light Relationship
# ------------------------------------------------------------------------ #
light.set_gladstone_dale_constant(lambdaLow=lsSpectrumBin[0], lambdaHigh=lsSectrumBin[1], lsSpectrumDisc=1)
fluidMesh.compute_index_of_refraction(light.GDC)

# ------------------------------------------------------------------------ #
# Cell gradient computation
# ------------------------------------------------------------------------ #
gradYN = 1
fluidMesh.compute_cell_gradient()

# ------------------------------------------------------------------------ #
# Surface mesh normal computation
# ------------------------------------------------------------------------ #
wingMesh.compute_surface_normals()

# ------------------------------------------------------------------------ #
# Ray tracing
# ------------------------------------------------------------------------ #
timer = vtk.vtkTimerLog()
timer.StartTimer()
for i in range(light.photonsOrg.shape[0]):
    print(i)
    
    photonMapDict = {}
    
    # Linear Ray Trace - Visibility Check
    source          = light.photonsOrg[i,:]
    t               = 10**10
    target          = source + t*light.photonsDir[i,:]
        
    tol             = 1*(10**-20)
    t               = vtk.mutable(0)
    pcoords         = [0.0,0.0,0.0]
    subId           = vtk.mutable(0)
    cellId          = vtk.mutable(0)
    
    linearHitPoint  = [0.0,0.0,0.0]
    hitTri          = vtk.vtkGenericCell()
    tmp             = wingMesh.obbtree.IntersectWithLine(source, target, tol, t, linearHitPoint, pcoords, subId, cellId, hitTri)  
    
    # Surface visible to light ray!
    if tmp != 0:
        # Get interpolated normal at linear hit point - Gouraud/Phong Shading
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
        
        # Store linear hit point, direction and normal
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
        
        
        # Nonlinear Ray Trace
        initialState    = (light.photonsOrg[i,0], light.photonsOrg[i,1], light.photonsOrg[i,2], light.photonsDir[i,0], light.photonsDir[i,1], light.photonsDir[i,2])
        nonlinear_ray_trace(photonMapDict, initialState, plotter)

# ------------------------------------------------------------------------ #
# Save illumination photon map
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





