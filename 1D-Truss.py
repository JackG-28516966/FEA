# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
Finite element analysis of 1D Truss Structure
version 0.002
--------------------------------------------------------------------------
Restraints entered as rigid '*' or with stiffness (kN/m)
Units kN, m

Reference text: Fundamentals of Finite Element Analysis by Ioannis 
Koutromanos

--------------------------------------------------------------------------
Symbols
--------------------------------------------------------------------------
E       Young's Modulus (kN/m^2)
A       Cross sectional area (m^2)
L       Element length (m)
x1      Node 1 global x coordinate (m)
x2      Node 2 global x coordinate (m)
P       Point load (kN)
R       Restraint stiffness (kN/m)

--------------------------------------------------------------------------
Degrees of freedom
--------------------------------------------------------------------------
Each node:
x
Each element:
x1,x2

--------------------------------------------------------------------------
Notes
--------------------------------------------------------------------------
Model can be constructed using scatter-gather boolean arrays
OR more efficient connectivity (LM) and/or ID array methods
The latter method is used here
"""

import numpy as np
from scipy import linalg as sp_linalg
# import matplotlib.pyplot as plt

"""
--------------------------------------------------------------------------
MODEL BUILDER
--------------------------------------------------------------------------
"""

class ModelBuilder:
    """Builds global model from elements"""
    def __init__(self,ModelID=1):
        """Collects model objects and properties for each instance"""
        self.ModelID=ModelID # Allows for multiple model instances
        self.ElemList=[] # List of element objects within model
        self.NodeList={} # Takes {NodeID:x} entries
        self.RestraintList={} # Takes {NodeID:restraint} entries
        self.NodeLoadList={} # Takes {NodeID:P} entries
        self.NodeDisplList={} # Takes {NodeID:presrcibed displacement} entries
    
    def AddElement(self,ElemID,E,A,x1,x2):
        """
        Adds element into model
        self = model object, ElemID = element ID, E = stiffness, A = cross-
        section area, L = length, x1 = x coordinate 1, x2 = x coordinate 2
        """
        ElemObj=TrussElement(ElemID,E,A,x1,x2) # Creates element
        self.ElemList.append(ElemObj) # Adds element to model list

    def AddNode(self,NodeID,x):
        """
        Defines model global coordinates.
        self = model object, NodeID = node ID, x = x coordinate
        """
        self.NodeList[NodeID]=x # Adds node and coord as dict entry
        self.NodeLoadList[NodeID]=0 # Initialises node loads
        self.RestraintList[NodeID]=0 # Initialises node restraints
        self.NodeDisplList[NodeID]=0 # Initialises node displacements
        
    def AddRestraint(self,NodeID,x_res):
        """
        Adds restraints to nodes.
        NodeID = Node number, 
        x_res = '*' for fixed, else give stiffness (kN/m)
        """
        if x_res =='*':
            self.RestraintList[NodeID]=x_res
        else:
            self.RestraintList[NodeID]+=x_res
            # Adds node and restraint to dict
        
    def AddNodeLoad(self,NodeID,P):
        """
        Adds nodal load.
        self = model object, NodeID = applied node, P = point load.
        """
        self.NodeLoadList[NodeID]+=P
            # Adds load to node in dict
    
    def AddNodeDispl(self,NodeID,u):
        """
        Adds prescribed nodal displacements
        self = model object, NodeID = node, u = prescribed displacement
        """
        self.NodeDisplList[NodeID]+=u # Adds prescribed displ to node in dict
        
"""
--------------------------------------------------------------------------
ELEMENT BUILDER
--------------------------------------------------------------------------
"""

class TrussElement:
    """"Defines 1D truss element within model"""
    def __init__(self,ElemID,E,A,x1,x2):
        """Associates args with element instance"""
        self.E=E # Input variables necessary to define element
        self.A=A
        self.L=abs(x2-x1)
        self.ElemID=ElemID # Associates element with ID number
        self.x1=x1 # Associates element with global coords
        self.x2=x2
        self.EK=self.KElem() # Creates element stiffness matrix

    def KElem(self):
        """Defines element stiffness matrix"""
        const1=self.E*self.A/self.L
        k1=np.array([[1, -1],[-1, 1]])
        EK=const1*k1
        return EK

"""
--------------------------------------------------------------------------
SOLVER AND ANALYSIS FUNCTIONS
--------------------------------------------------------------------------
"""

def AsmblGK(Model):
    """Assembles global stiffness matrix GK"""
    NodeDof=1 # Number of dof per node
    NumENodes=2 # Number of element nodes   
    NumEDof=NodeDof*NumENodes # Number of element dof
    NumGNodes=len(Model.NodeList) # Number of global nodes
    NumGDof=NumGNodes*NodeDof # Number of global dof
    GK=np.zeros((NumGDof,NumGDof))
    for e in Model.ElemList: # iterates through elements
        EK=np.array(e.EK) # extracts element stiffness matrix
        for jc in range(1,NumEDof+1): # iterates columns
            LMj=LM(Model,e,jc)
            for ir in range(1,NumEDof+1): # iterates rows
                LMi=LM(Model,e,ir)
                GK[LMi-1,LMj-1]+=EK[ir-1,jc-1]
    RestraintList=Model.RestraintList # Extact restraint dict
        # adds spring support stiffnesses into matrix
    for key in RestraintList:
        x_res=RestraintList[key]
        if x_res!='*':
            GK[key-1,key-1]+=x_res
    return GK

def SubAsmblGK(Model):
    """Uses GK to obtain GKff, GKfs, GKss, ff, Us"""
        # Import GK and model dicts
    GK=AsmblGK(Model) # Calculates GK
    NodeList=Model.NodeList # Extracts node dict
    RestraintList=Model.RestraintList # Extracts restraint dict
    NodeDisplList=Model.NodeDisplList # Extracts prescribed displ
    NodeLoadList=Model.NodeLoadList # Extracts nodal loads
        # Calculate submatrix shapes
    Uf=(Uf_Gen(Model)) # Returns free node dict with zeros
    Ufvals=np.array(list(Uf.values())) # Free node index values
    Uff=Ufvals[Ufvals!=0] # Remove zero values
    GKffShape=len(Uff) # Calculate shape of Kff (unknowns matrix)    
    GKfsShape=len(GK)-GKffShape
        # Initialise GK submatrices
    GKff=np.zeros((GKffShape,GKffShape))
    GKfs=np.zeros((GKffShape,GKfsShape))
    GKss=np.zeros((GKfsShape,GKfsShape))
        # Calculate associated knowns matrices Us and ff
    UsNodes=[]  
    Us=[] # known displacements
    UfNodes=[]
    ff=[] # free node forces
    for NodeID in NodeList: # iterate global node ID keys
        if RestraintList[NodeID]!='*': # if unrestrained
            UfNodes.append(NodeID)
            ff.append(NodeLoadList[NodeID])
        else: # if restrained
            UsNodes.append(NodeID)
            Us.append(NodeDisplList[NodeID])
        # Calculate GK submatrix GKss
    for col in range(0,GKfsShape):
        for row in range(0,GKfsShape):
            rowVal=UsNodes[row]-1
            colVal=UsNodes[col]-1
            GKss[row,col]=GK[rowVal,colVal]            
        # Calculate GK submatrix GKfs
    for col in range(0,GKfsShape):
        for row in range(0,GKffShape):
            rowVal=UfNodes[row]-1
            colVal=UsNodes[col]-1
            GKfs[row,col]=GK[rowVal,colVal]
        # Calculate GK submatrix GKsf
    GKsf=np.transpose(GKfs)
        # Calculate GK submatrix GKff
    NodeDof=1 # Number of dof per node
    NumENodes=2 # Number of element nodes   
    NumEDof=NodeDof*NumENodes # Number of element dof
    for e in Model.ElemList: # iterates through elements
        EK=np.array(e.EK) # extracts element stiffness matrix
        for jc in range(1,NumEDof+1): # iterates columns
            IDj=ID(Model,LM(Model,e,jc))
            if IDj>0:
                for ir in range(1,NumEDof+1): # iterates rows
                    IDi=ID(Model,LM(Model,e,ir))
                    if IDi>0:
                        IDj=ID(Model,LM(Model,e,jc))
                        GKff[IDi-1,IDj-1]+=EK[ir-1,jc-1]    
    #   Convert result lists in numpy arrays
    UsNodes=np.array(UsNodes)
    Us=np.array(Us)
    UfNodes=np.array(UfNodes)
    ff=np.array(ff)
    return [GKff,GKfs,GKsf,GKss,ff,Us,UfNodes,UsNodes]

def ID(Model,node):
    """Calculates the ID matrix"""
    Uf=Uf_Gen(Model)
    return Uf[node]
    
def Uf_Gen(Model):
    """Calculates unrestrained node matrix Uf"""
    NodeList=Model.NodeList # Extracts node dict
    RestraintList=Model.RestraintList # Extracts restraint dict
    FreeNodeList={} # Takes {global node: free node}
    FreeNodeIndex=1 # Initialise free node count
    for i in range(1,len(NodeList)+1):
        xR=RestraintList[i] # Extract restraint condition
        if xR!='*': # If unrestrained
            FreeNodeList[i]=FreeNodeIndex # Add to free node dict
            FreeNodeIndex+=1 # Update free node count    
        else:
            FreeNodeList[i]=0
    return FreeNodeList

def LM(Model,ElemE,dof):
    """
    Returns the LM connectivity matrix.
    Model = model object, ElemE = element object, dof = elem dof
    x1: dof = 1, x2: dof = 2 
    """
    x1=ElemE.x1 # Extracts global coords from element E
    x2=ElemE.x2
    Ulist=[x1,x2] # Collate coords
    U=Ulist[dof-1] # Select coord of interest using dof input
    for key in Model.NodeList: # Iterate global nodes
        if Model.NodeList[key]==U: # If node coord = elem dof
            LMval=key
    return LMval
       

def Solver(Model):
    """Solves linear equations using SciPy linalg"""
    MatrixList=SubAsmblGK(Model) 
    # Outputs[GKff,GKfs,GKsf,GKss,ff,Us,UfNodes,UsNodes]
        # Extract matrices for solving
    GlobalNodeList=Model.NodeList
    GKff=MatrixList[0]
    GKfs=MatrixList[1]
    GKsf=MatrixList[2]
    GKss=MatrixList[3]
    ff=MatrixList[4]
    Us=MatrixList[5]
    UfNodes=MatrixList[6]
    UsNodes=MatrixList[7]
        # Calculate Pf (free node forces including prescribed displ)
    mult1=np.dot(GKfs,Us)
    Pf=ff-mult1
        # Use SciPy linalg to obtain free node displacements
    Uf=sp_linalg.solve(GKff,Pf)
        # Calculate reactions
    Uf=np.reshape(Uf,(-1,1))
    fs=np.dot(GKsf,Uf)+np.dot(GKss,Us)
        # Create nodal dictionaries to contain results
    NodeReaction={}
    AllNodeF={}
    NodeDispl={}
    AllNodeDispl={}
    for k in GlobalNodeList: # initialise result dicts with keys and zero vals
        NodeReaction[k]=0
        AllNodeF[k]=0
        NodeDispl[k]=0
        AllNodeDispl[k]=0
    for i in range(0,len(UfNodes)): # iterate free result nodes
        NodeID=UfNodes[i]
        uf=Uf[i][0]
        NodeDispl[NodeID]+=uf
        AllNodeDispl[NodeID]+=uf
        f_x=ff[i]
        AllNodeF[NodeID]+=f_x
    for j in range(0,len(UsNodes)): # iterate support reaction results    
        NodeID=UsNodes[j]
        us=Us[j]
        AllNodeDispl[NodeID]+=us
        r_x=fs[j][0]
        NodeReaction[NodeID]+=r_x
        AllNodeF[NodeID]+=r_x
    return [AllNodeDispl,AllNodeF]

"""
--------------------------------------------------------------------------
INPUT CODE
--------------------------------------------------------------------------
"""


"Initialise Model"
Model1=ModelBuilder()
"Define Nodes"
Node1=ModelBuilder.AddNode(Model1,1,0)
Node2=ModelBuilder.AddNode(Model1,2,5)
Node3=ModelBuilder.AddNode(Model1,3,9)
Node4=ModelBuilder.AddNode(Model1,4,14)
"Define Elements"
AddElem1=ModelBuilder.AddElement(Model1,1,2e8,0.005,0,5)
AddElem2=ModelBuilder.AddElement(Model1,2,2e8,0.01,5,9)
AddElem3=ModelBuilder.AddElement(Model1,1,2e8,0.0025,9,14)
Elem1=TrussElement(1,2e8,0.005,0,5)
Elem2=TrussElement(2,2e8,0.01,5,9)
Elem3=TrussElement(3,2e8,0.0025,9,14)
"Define Restraints"
Res1=ModelBuilder.AddRestraint(Model1,3,'*')
Res2=ModelBuilder.AddRestraint(Model1,2,10)
"Define Prescribed Displacements"
Displ1=ModelBuilder.AddNodeDispl(Model1,3,0.00002)
"Define Loading"
P1=ModelBuilder.AddNodeLoad(Model1,4,10)

#print(AsmblGK(Model1))
#print(SubAsmblGK(Model1))
print(Solver(Model1))
