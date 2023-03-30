# Import external libraries
import pyvista as pv
import numpy as np
import pdb
import vtk
import os

class Mesh():
    self = None
# ------------------------------------------------------------------------ #
# Class constructor
# ------------------------------------------------------------------------ #    
    def __init__(self):
        self.boundBox = pv.Box()
        pass
# ------------------------------------------------------------------------ #
# Tree data structures for point location
# ------------------------------------------------------------------------ #        
    def build_obb_tree(self):
        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(self.grid)
        geomFilter.Update()
        self.obbtree    = vtk.vtkOBBTree()
        self.obbtree.SetDataSet(geomFilter.GetOutput())
        self.obbtree.BuildLocator()
        
    def build_kd_tree(self):
        self.kdtree     = vtk.vtkKdTreePointLocator()
        self.kdtree.SetDataSet(self.grid)
        self.kdtree.BuildLocator() 
    def build_point_locator(self):
        self.pointLocator = vtk.vtkKdTreePointLocator()
        self.pointLocator.SetDataSet(self.grid)
        self.pointLocator.BuildLocator()       
    def build_octree(self):
    
        self.octree     = vtk.vtkOctreePointLocator()
        self.octree.SetDataSet(self.grid)
        self.octree.BuildLocator()
# ------------------------------------------------------------------------ #
# Tree data structure for cell location
# ------------------------------------------------------------------------ #
    def build_cell_locator(self):
        # self.cellLocator = vtk.vtkCellLocator()
        self.cellLocator = vtk.vtkCellTreeLocator()
        self.cellLocator.SetDataSet(self.grid)
        self.cellLocator.BuildLocator()
# ------------------------------------------------------------------------ #
# Filter for point in/out location
# ------------------------------------------------------------------------ #
    def build_grid_in_out_checker(self, tol):
        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(self.grid)
        geomFilter.Update()
        gridBoundary = geomFilter.GetOutput()
        self.inOutChecker = vtk.vtkSelectEnclosedPoints()
        self.inOutChecker.SetSurfaceData(gridBoundary)
        self.inOutChecker.SetTolerance(tol)      
        
    def build_bbox_in_out_checker(self, tol):
        self.boundBox   = pv.Box(self.grid.bounds)
        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(self.boundBox)
        geomFilter.Update()
        bBoxBoundary = geomFilter.GetOutput()
        self.boundBox.inOutChecker = vtk.vtkSelectEnclosedPoints()
        self.boundBox.inOutChecker.SetSurfaceData(bBoxBoundary)
        self.boundBox.inOutChecker.SetTolerance(tol)    
# ------------------------------------------------------------------------ #
# Filter and chosen kernel for point interpolation
# ------------------------------------------------------------------------ #
    def set_gaussian_interpolator(self, sharpness, radius):
        gaussianKernel = vtk.vtkGaussianKernel()
        gaussianKernel.SetSharpness(sharpness)
        gaussianKernel.SetRadius(radius)
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetSourceData(self.grid)
        self.interpolator.SetKernel(gaussianKernel)
        self.interpolator.SetLocator(self.octree)
        self.interpolator.SetNullValue(0.0)
        self.interpolator.SetNullPointsStrategyToNullValue()

    def set_voronoi_interpolator(self):
        voronoiKernel = vtk.vtkVoronoiKernel()
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetSourceData(self.grid)
        self.interpolator.SetKernel(voronoiKernel)
        self.interpolator.SetLocator(self.octree)
        self.interpolator.SetNullValue(0.0)
        self.interpolator.SetNullPointsStrategyToNullValue()

    def set_shepard_interpolator(self, power, radius):
        shepardKernel = vtk.vtkShepardKernel()
        shepardKernel.SetPowerParameter(power)
        shepardKernel.SetRadius(radius)
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetSourceData(self.grid)
        self.interpolator.SetKernel(shepardKernel)
        self.interpolator.SetLocator(self.octree)
        self.interpolator.SetNullValue(0.0)
        self.interpolator.SetNullPointsStrategyToNullValue()

    def set_linear_interpolator(self, radius):
        linearKernel = vtk.vtkLinearKernel()
        linearKernel.SetRadius(radius)
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetSourceData(self.grid)
        self.interpolator.SetKernel(linearKernel)
        self.interpolator.SetLocator(self.octree)
        self.interpolator.SetNullValue(0.0)
        self.interpolator.SetNullPointsStrategyToNullValue()
        
    def set_ellipsoidal_gaussian_interpolator(self, sharpness, radius):
        gaussianKernel = vtk.vtkEllipsoidalGaussianKernel()
        gaussianKernel.UseScalarsOn()
        gaussianKernel.UseNormalsOn()
        gaussianKernel.SetSharpness(sharpness)
        gaussianKernel.SetRadius(radius)
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetSourceData(self.grid)
        self.interpolator.SetKernel(gaussianKernel)
        self.interpolator.SetLocator(self.octree)
        self.interpolator.SetNullValue(0.0)        
        self.interpolator.SetNullPointsStrategyToNullValue()        
  
# ------------------------------------------------------------------------ #
# Surface normal computation
# ------------------------------------------------------------------------ #
    def compute_surface_normals(self):
        geomFilter = vtk.vtkGeometryFilter()
        geomFilter.SetInputData(self.grid)
        geomFilter.Update()
        
        normalGenerator = vtk.vtkTriangleMeshPointNormals()
        normalGenerator.SetInputData(geomFilter.GetOutput())
        normalGenerator.Update()
        self.grid['Normals'] = -np.array(normalGenerator.GetOutput().GetPointData().GetArray('Normals'))
        
# ------------------------------------------------------------------------ #
# Index-of-refraction computation
# ------------------------------------------------------------------------ #        
    def compute_index_of_refraction(self, GDC):
        self.grid['IOR']                = (1 + GDC*self.grid["Density"])
        self.grid["IORSqrdVertex"]      = 0.5*(self.grid['IOR']**2)
        
        tmp = vtk.vtkPointDataToCellData()
        tmp.ProcessAllArraysOff()
        tmp.AddPointDataArray('IORSqrdVertex')
        tmp.SetInputData(self.grid)
        tmp.Update()
        tmp = pv.UnstructuredGrid(tmp.GetOutput())
        
        self.grid.cell_arrays['IORSqrdCell'] = tmp['IORSqrdVertex']
        
# ------------------------------------------------------------------------ #
# Gradient vector computation
# ------------------------------------------------------------------------ #        
    def compute_gradient(self):
        self.grid = self.grid.compute_derivative(scalars="IORSqrdVertex", gradient="Gradient", preference='point')

    def compute_vertex_gradient(self):
        self.grid = self.grid.compute_derivative(scalars="IORSqrdVertex", gradient="GradientVertex", preference='point')
            
    def compute_cell_gradient(self):
        self.grid = self.grid.compute_derivative(scalars="IORSqrdCell", gradient="GradientCell", faster=True, preference='cell')        

# ------------------------------------------------------------------------ #
# Point in/out location computation
# ------------------------------------------------------------------------ #
    def locate_point_in_out_grid(self, p):
        self.inOutChecker.SetInputData(pv.PolyData(p))
        self.inOutChecker.Update()
        check = self.inOutChecker.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0]
        return check

    def locate_point_in_out_bbox(self, p):
        self.boundBox.inOutChecker.SetInputData(pv.PolyData(p))
        self.boundBox.inOutChecker.Update()
        check = self.boundBox.inOutChecker.GetOutput().GetPointData().GetArray('SelectedPoints').GetTuple(0)[0]
        return check
        
# ------------------------------------------------------------------------ #
# Scalar field interpolation via Hyperplane Fitting method
# ------------------------------------------------------------------------ #
    def interpolate_density_field(self, interpPt):
        closestPoint  = [0,0,0]
        genCell       = vtk.vtkGenericCell()
        cellId        = vtk.mutable(0)
        subId         = vtk.mutable(0)
        distance      = vtk.mutable(0)
        
        
        self.cellLocator.FindClosestPoint(interpPt, closestPoint, genCell, cellId, subId, distance)
        
        v = 0
        closestVertexId = genCell.GetPointIds().GetId(v)
        closestVertex   = self.grid.points[closestVertexId,:]
        dist = np.linalg.norm(interpPt-closestVertex)
        for v in range(1,4,1):
            vertex  = self.grid.points[genCell.GetPointIds().GetId(v),:]
            if (np.linalg.norm(interpPt-vertex) < dist):
                closestVertexId = genCell.GetPointIds().GetId(v)
                closestVertex   = vertex
            
        adjCells = vtk.vtkIdList()
        self.grid.GetPointCells(closestVertexId, adjCells)
        
        X = np.zeros((adjCells.GetNumberOfIds()*4, 4))
        F = np.zeros((adjCells.GetNumberOfIds()*4))
        
        for ac in range(0, adjCells.GetNumberOfIds(), 1):
            cellVertices = vtk.vtkIdList()
            self.grid.GetCellPoints(adjCells.GetId(ac), cellVertices)
            
            for av in range(0,4,1):
                adjVertex = self.grid.points[cellVertices.GetId(av),:]
                
                X[av + 4*ac, 0] = adjVertex[0]
                X[av + 4*ac, 1] = adjVertex[1]
                X[av + 4*ac, 2] = adjVertex[2]
                X[av + 4*ac, 3] = 1
                
                F[av + 4*ac]    = self.grid['Density'][cellVertices.GetId(av)]
        
        A = np.linalg.solve(np.matmul(np.transpose(X),X), np.matmul(np.transpose(X),F))
        
        interpVal = (A[0]*interpPt[0])+(A[1]*interpPt[1])+(A[2]*interpPt[2])+A[3]
        
        return(interpVal)
        
        
        
        
        
        
        
        
        
        
        




