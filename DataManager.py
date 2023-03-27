import pdb
import numpy as np
import vtk
import pyvista as pv
# Last edited on the computer on 28/04/21
class DataManager():
    def __init__(self, fileName):
        self.fileName       = fileName
        self.numOfNodes     = 0
        self.numOfElements  = 0
        self.points         = []
        self.attributes     = []
        self.cells          = []
        self.cellTypes      = []
        
        
    def open_and_read_mixed_ele_dat(self):
    
        fileReader = open(self.fileName, "rt")
    
        line                = fileReader.readline()
        gotNumOfNodes       = False
        gotNumOfElements    = False

        while line:

            if line.find("Nodes=") != -1:
                global numOfNodes
                numOfNodes = ""
                i = line.find("Nodes=")+6
                while (line[i] != ","):
                    numOfNodes += line[i]
                    i += 1
                self.numOfNodes = int(numOfNodes)
                gotNumOfNodes = True
                
            if line.find("Elements=") != -1:
                global numOfElements
                numOfElements = ""
                i = line.find("Elements=")+9
                while (line[i] != ","):
                    numOfElements += line[i]
                    i += 1
                self.numOfElements = int(numOfElements)
                gotNumOfElements = True
                line        = fileReader.readline()
                line        = fileReader.readline()
            
            if gotNumOfElements == True and gotNumOfNodes == True:
                
                countNodes = 0
                while (countNodes < self.numOfNodes):
                    line        = fileReader.readline()
                    data = line.split()
                    x   = float(data[0])
                    y   = float(data[1])
                    z   = float(data[2])
                    rho = float(data[3])
                    
                    self.points.append([x,y,z])
                    self.attributes.append([rho])
                    countNodes += 1
                

                countElements = 0
                while (countElements < self.numOfElements):
                    line = fileReader.readline()
                    data = line.split()
                                                           
                    if (len(set(data)) == 4):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                        v3   = int(data[3]) - 1
                        self.cells.append([4, v0, v1, v2, v3])
                        self.cellTypes.append([vtk.VTK_TETRA])
                    elif (len(set(data)) == 5):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                        v3   = int(data[3]) - 1
                        v4   = int(data[4]) - 1
                        self.cells.append([5, v0, v1, v2, v3, v4])                        
                        self.cellTypes.append([vtk.VTK_PYRAMID])
                    elif (len(set(data)) == 6):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                        v3   = int(data[3]) - 1
                        v4   = int(data[4]) - 1
                        v5   = int(data[5]) - 1
                        self.cells.append([6, v0, v1, v2, v3, v4, v5])                        
                        self.cellTypes.append([vtk.VTK_WEDGE])
                    elif (len(set(data)) == 8):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                        v3   = int(data[3]) - 1
                        v4   = int(data[4]) - 1
                        v5   = int(data[5]) - 1
                        v6   = int(data[6]) - 1
                        v7   = int(data[7]) - 1
                        self.cells.append([8, v0, v1, v2, v3, v4, v5, v6, v7])    
                        self.cellTypes.append([vtk.VTK_HEXAHEDRON])
                    countElements += 1

                self.numOfElements =  len(self.cells)

            line = fileReader.readline()


        fileReader.close()    
        
        
        
    def open_and_read_dat(self):
    
        fileReader = open(self.fileName, "rt")
    
        line                = fileReader.readline()
        gotNumOfNodes       = False
        gotNumOfElements    = False

        while line:

            if line.find("Nodes=") != -1:
                global numOfNodes
                numOfNodes = ""
                i = line.find("Nodes=")+6
                while (line[i] != ","):
                    numOfNodes += line[i]
                    i += 1
                self.numOfNodes = int(numOfNodes)
                gotNumOfNodes = True
                
            if line.find("Elements=") != -1:
                global numOfElements
                numOfElements = ""
                i = line.find("Elements=")+9
                while (line[i] != ","):
                    numOfElements += line[i]
                    i += 1
                self.numOfElements = int(numOfElements)
                gotNumOfElements = True
                line        = fileReader.readline()
                line        = fileReader.readline()
            
            if gotNumOfElements == True and gotNumOfNodes == True:
                
                countNodes = 0
                while (countNodes < self.numOfNodes):
                    line        = fileReader.readline()
                    data = line.split()
                    x   = float(data[0])
                    y   = float(data[1])
                    z   = float(data[2])
                    rho = float(data[3])
                    
                    self.points.append([x,y,z])
                    self.attributes.append([rho])
                    countNodes += 1
                
                countElements = 0
                while (countElements < self.numOfElements):
                    line        = fileReader.readline()
                    data = line.split()
                    
                    if (len(set(data)) == 3):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                                                
                        self.cells.append([3, v0, v1, v2])
                        self.cellTypes.append([vtk.VTK_TRIANGLE])
                        
                    elif (len(set(data)) == 4):
                        v0   = int(data[0]) - 1
                        v1   = int(data[1]) - 1
                        v2   = int(data[2]) - 1
                        v3   = int(data[3]) - 1
                        
                        self.cells.append([4, v0, v1, v2, v3])
                        self.cellTypes.append([vtk.VTK_TETRA])
                    countElements += 1

                self.numOfElements =  len(self.cells)
            line = fileReader.readline()


        fileReader.close()


    def open_and_read_obj(self):
    
        with open(self.fileName, "r") as file:
            line = file.readlines()
            
            
            
            for idx in range(0, len(line), 1):
                items = line[idx].split()
                
                if (items[0] == "v"):
                    self.points.append([float(items[1]), float(items[2]), float(items[3])])
                    self.attributes.append([0.0])
                    self.numOfNodes += 1
                elif (items[0] == "f"):
                    self.cells.append([3, int(items[1].split("/")[0])-1, int(items[2].split("/")[0])-1, int(items[3].split("/")[0])-1])
                    self.numOfElements += 1
                    
    def open_and_read_node(self):

        with open(self.fileName, "r") as file:
        
            line = file.readlines()
                       
            for idx in range(0, len(line),1):
                items = line[idx].split()
                self.points.append([float(items[1]), float(items[2]), float(items[3])])
                self.attributes.append([float(items[4])])
                self.numOfNodes += 1
                    
    def open_and_read_ele(self):

        with open(self.fileName, "r") as file:
            
            line = file.readlines()
            
            for idx in range(0, len(line),1):
                items           = line[idx].split()
                self.cells.append([4, int(int(items[1])-1),int(int(items[2])-1),int(int(items[3])-1),int(int(items[4])-1)])
                self.numOfElements +=1                   
                    
    def write_vtp(self, polyData):
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(self.fileName)
        writer.SetInputData(polyData)
        writer.Write()
                    
    def read_vtp(self):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(self.fileName)
        reader.Update()
        
        return (reader.GetOutput())
                    
                    
                    
                    
                    
                    
                    