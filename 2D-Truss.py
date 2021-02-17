# -*- coding: utf-8 -*-
"""
--------------------------------------------------------------------------
Finite element analysis of 2D Truss Structure
version 0.001

done: Change element definition to 2D
    
to do: AddElement, AddNode, AddRestraint
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
x,y
Each element:
x1,y1,x2,y2 = u1,u2,u3,u4

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
        self.NodeList={} # Takes {NodeID:[x,y] coords} 
        self.RestraintList={} # Takes {NodeID:[x,y]restraint} 
        self.NodeLoadList={} # Takes {NodeID:[Px,Py]force} 
        self.NodeDisplList={} # Takes {NodeID:[x,y]prescribed displacement} 
        self.DOFList={} # Takes {NodeID: [dof1,dof2]}

    def AddElement(self,ElemID,E,A,Node1,Node2):
        """
        Adds element into model
        self = model object, ElemID = element ID, E = stiffness, A = cross-
        section area, L = length, x1 = x coordinate 1, y1 = y coordinate 1
        x2 = x coordinate 2, y2 = y coordinate 2
        """
        x1=self.NodeList[Node1][0]
        y1=self.NodeList[Node1][1]
        x2=self.NodeList[Node2][0]
        y2=self.NodeList[Node2][1]
        ElemObj=TrussElement(ElemID,E,A,x1,y1,x2,y2) # Creates element
        self.ElemList.append(ElemObj) # Adds element to model list

    def AddNode(self,NodeID,x,y):
        """
        Defines model global coordinates.
        self = model object, NodeID = node ID, x = x coordinate, 
        y = y coordinate
        """
        self.NodeList[NodeID]=[x,y] # Adds node and coord as dict entry
        self.NodeLoadList[NodeID]=[0,0] # Initialises node loads
        self.RestraintList[NodeID]=[0,0] # Initialises node restraints
        self.NodeDisplList[NodeID]=[0,0] # Initialises node displacements
        self.DOFList[NodeID]=[2*NodeID-1,2*NodeID] # Initialises DOFs
        
    def AddRestraint(self,NodeID,x_res,y_res):
        """
        Adds restraints to nodes.
        NodeID = Node number, 
        x_res = '*' for fixed, else give stiffness (kN/m)
        y_res = '*' for fixed, else give stiffness (kN/m)
        """
        if x_res =='*':
            self.RestraintList[NodeID][0]=x_res
        else:
            self.RestraintList[NodeID][0]+=x_res
            # Adds node and restraint to dict
        if y_res =='*':
            self.RestraintList[NodeID][1]=y_res
        else:
            self.RestraintList[NodeID][1]+=y_res
            # Adds node and restraint to dict
        
    def AddNodeLoad(self,NodeID,Px,Py):
        """
        Adds nodal load.
        self = model object, NodeID = applied node, P = point load
        """
        self.NodeLoadList[NodeID][0]+=Px
        self.NodeLoadList[NodeID][1]+=Py
            # Adds load to node in dict
    
    def AddNodeDispl(self,NodeID,ux,uy):
        """
        Adds prescribed nodal displacements
        self = model object, NodeID = node, u = prescribed displacement
        """
        self.NodeDisplList[NodeID][0]+=ux # Adds prescribed displ to node in dict
        self.NodeDisplList[NodeID][1]+=uy
        
"""
--------------------------------------------------------------------------
ELEMENT BUILDER
--------------------------------------------------------------------------
"""

class TrussElement:
    """"Defines 1D truss element within model"""
    def __init__(self,ElemID,E,A,x1,y1,x2,y2):
        """Associates args with element instance"""
        self.E=E # Input variables necessary to define element
        self.A=A
        self.L=((x2-x1)**2+(y2-y1)**2)**0.5
        self.ElemID=ElemID # Associates element with ID number
        self.x1=x1 # Associates element with global coords
        self.y1=y1
        self.x2=x2
        self.y2=y2
        self.cosphi=(x2-x1)/self.L # Associates element angle
        self.sinphi=(y2-y1)/self.L
        self.EK=self.KElem() # Creates element stiffness matrix

    def KElem(self):
        """Defines element stiffness matrix in global coords"""
        const1=self.E*self.A/self.L
        c=self.cosphi
        s=self.sinphi
        k1=np.array([[c**2,c*s,-c**2,-c*s],[c*s,s**2,-c*s,-s**2],
                     [-c**2,-c*s,c**2,c*s],[-c*s,-s**2,c*s,s**2]])
        EK=const1*k1
        return EK

"""
--------------------------------------------------------------------------
SOLVER AND ANALYSIS FUNCTIONS
--------------------------------------------------------------------------
"""

def AsmblGK(Model):
    """Assembles global stiffness matrix GK"""
    NodeDof=2 # Number of dof per node
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
        x_res=RestraintList[key][0]
        y_res=RestraintList[key][1]
        dofx=Model.DOFList[key][0]
        dofy=Model.DOFList[key][1]
        if x_res!='*':
            GK[dofx-1,dofx-1]+=x_res
        if y_res!='*':
            GK[dofy-1,dofy-1]+=y_res
    return GK

def SubAsmblGK(Model):
    """Uses GK to obtain GKff, GKfs, GKss, ff, Us"""
        # Import GK and model dicts
    GK=AsmblGK(Model) # Calculates GK
    NodeList=Model.NodeList # Extracts node dict
    RestraintList=Model.RestraintList # Extracts restraint dict
    NodeDisplList=Model.NodeDisplList # Extracts prescribed displ
    NodeLoadList=Model.NodeLoadList # Extracts nodal loads
    DOFList=Model.DOFList
        # Calculate submatrix shapes
    Uf=(Uf_Gen(Model)) # Returns free DOF dict with zeros
    Ufvals=np.array(list(Uf.values())) # Free DOF index values
    Uff=Ufvals[Ufvals!=0] # Remove zero values
    GKffShape=len(Uff) # Calculate shape of Kff (unknowns matrix)    
    GKfsShape=len(GK)-GKffShape
        # Initialise GK submatrices
    GKff=np.zeros((GKffShape,GKffShape))
    GKfs=np.zeros((GKffShape,GKfsShape))
    GKss=np.zeros((GKfsShape,GKfsShape))
        # Calculate associated knowns matrices Us and ff
    UsDOFs=[]  
    Us=[] # known displacements
    UfDOFs=[]
    ff=[] # free node forces
    for NodeID in NodeList: # iterate global node ID keys
        dofx=DOFList[NodeID][0]
        dofy=DOFList[NodeID][1]
        x_res=RestraintList[NodeID][0]
        y_res=RestraintList[NodeID][1]
        Px=NodeLoadList[NodeID][0]
        Py=NodeLoadList[NodeID][1]
        u_x=NodeDisplList[NodeID][0]
        u_y=NodeDisplList[NodeID][1]
        if x_res!='*' and y_res!='*': # if unrestrained
            UfDOFs.append(dofx)
            UfDOFs.append(dofy)
            ff.append(Px)
            ff.append(Py)
        elif x_res!='*':
            UfDOFs.append(dofx)
            UsDOFs.append(dofy)
            ff.append(Px)
            Us.append(u_y)
        elif y_res!='*':
            UfDOFs.append(dofy)
            UsDOFs.append(dofx)
            ff.append(Py)
            Us.append(u_x)
        else: # if restrained
            UsDOFs.append(dofx)
            UsDOFs.append(dofy)
            Us.append(u_x)
            Us.append(u_y)
        # Calculate GK submatrix GKss
    for col in range(0,GKfsShape):
        for row in range(0,GKfsShape):
            rowVal=UsDOFs[row]-1
            colVal=UsDOFs[col]-1
            GKss[row,col]=GK[rowVal,colVal]            
        # Calculate GK submatrix GKfs
    for col in range(0,GKfsShape):
        for row in range(0,GKffShape):
            rowVal=UfDOFs[row]-1
            colVal=UsDOFs[col]-1
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
    UsDOFs=np.array(UsDOFs)
    Us=np.array(Us)
    UfDOFs=np.array(UfDOFs)
    ff=np.array(ff)
    return [GKff,GKfs,GKsf,GKss,ff,Us,UfDOFs,UsDOFs]

def ID(Model,ElemDOF):
    """Calculates the ID matrix"""
    Uf=Uf_Gen(Model)
    return Uf[ElemDOF]
    
def Uf_Gen(Model):
    """Calculates unrestrained node matrix Uf"""
    DOFList=Model.DOFList # Extracts DOF dict
    RestraintList=Model.RestraintList # Extracts restraint dict
    FreeDOFList={} # Takes {global DOF: free DOF}
    FreeDOFIndex=1 # Initialise free node count
    for Node in DOFList: # Iterates global DOF
        dofx=DOFList[Node][0]
        dofy=DOFList[Node][1]
        x_res=RestraintList[Node][0] # Extract restraint condition
        y_res=RestraintList[Node][1]
        if x_res!='*' and y_res!='*':
            FreeDOFList[dofx]=FreeDOFIndex
            FreeDOFIndex+=1
            FreeDOFList[dofy]=FreeDOFIndex
            FreeDOFIndex+=1
        elif x_res!='*':
            FreeDOFList[dofx]=FreeDOFIndex
            FreeDOFIndex+=1
            FreeDOFList[dofy]=0
        elif y_res!='*':
            FreeDOFList[dofx]=0
            FreeDOFList[dofy]=FreeDOFIndex
            FreeDOFIndex+=1
        else:
            FreeDOFList[dofx]=0
            FreeDOFList[dofy]=0
    return FreeDOFList

def LM(Model,ElemE,dof):
    """
    Returns the LM connectivity matrix.
    Model = model object, ElemE = element object, dof = elem dof
    x1: dof = 1, y1: dof = 2, x2: dof = 3, y2: dof = 4
    """
    x1=ElemE.x1 # Extracts global coords from element E
    y1=ElemE.y1
    x2=ElemE.x2
    y2=ElemE.y2
    if dof < 3:
        xcoord=x1
        ycoord=y1
        dofmod=dof
    else:
        xcoord=x2
        ycoord=y2
        dofmod=dof-2
    for key in Model.NodeList: # Iterate global nodes
        xNode=Model.NodeList[key][0]
        yNode=Model.NodeList[key][1]
        if xNode==xcoord and yNode==ycoord:
            LMval=Model.DOFList[key][dofmod-1]
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
Node1=ModelBuilder.AddNode(Model1,1,0,0)
Node2=ModelBuilder.AddNode(Model1,2,120,0)
Node3=ModelBuilder.AddNode(Model1,3,0,80)
"Define Elements"
AddElem1=ModelBuilder.AddElement(Model1,1,3e4,1,1,2)
AddElem2=ModelBuilder.AddElement(Model1,2,3e4,2,2,3)
#AddElem3=ModelBuilder.AddElement(Model1,1,2e8,0.0025,1,3)
Elem1=TrussElement(1,3e4,1,0,0,120,0)
Elem2=TrussElement(2,3e4,2,120,0,0,80)
#Elem3=TrussElement(3,2e8,0.0025,0,0,5,5)
"Define Restraints"
Res1=ModelBuilder.AddRestraint(Model1,1,'*','*')
Res2=ModelBuilder.AddRestraint(Model1,3,'*','*')
"Define Prescribed Displacements"
#Displ1=ModelBuilder.AddNodeDispl(Model1,3,0.00002)
"Define Loading"
P1=ModelBuilder.AddNodeLoad(Model1,2,0,-10)

#print(AsmblGK(Model1))
#print(SubAsmblGK(Model1))
#print(Solver(Model1))

print(Model1.DOFList)
#print(LM(Model1,Elem2,3))
#print(AsmblGK(Model1))
print(Uf_Gen(Model1))
print(Model1.RestraintList)

print('----------------------------------------------')
Uf=(Uf_Gen(Model1)) # Returns free node dict with zeros
print(np.array(list(Uf.values())))