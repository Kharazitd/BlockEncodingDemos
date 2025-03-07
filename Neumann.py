from scipy.sparse import identity
from scipy.sparse.linalg import spsolve
from helpers import *


class NeumBoundaryData:
    def __init__(self, bndry_points, interior, exterior, bndry_func = None, method = None):
        #Points should be passed as a list or array of tuples, where the number of elements in each tuple is the spatial dimension d
        #We will assume the entries are integers that correspond to grid points in $d$ dimensions 
        #bndry_func is a function which takes in a grid point (i,j,...) and returns whether it is: 0 (interior), 1 (boundary), or 2 (exterior)
        if bndry_func != None:
            self.bndry_func = bndry_func
            #need to figure out how to do this part at some point
        else:
            self.bndry_points = bndry_points
            self.interior = interior
            self.exterior = exterior


class SimpleNeumannBCs:
    def __init__(self, deriv):
        self.deriv = deriv
        self.N = deriv.N
    
    def constructMat(self):
        mat = -2*identity(self.N)
        mat += 1/2*(-reflectOp(self.N, [0, self.N-1]) + identity(self.N))
        mat += 1/2*(reflectOp(self.N,[0]) + identity(self.N) )@(self.deriv.shiftOps(-1))
        mat += 1/2*(reflectOp(self.N,[self.N-1]) + identity(self.N) )@(self.deriv.shiftOps(1))
        return mat.tocsr()

class nonSimpleNeumannBC1d:
    def __init__(self,deriv, bndry_data):
        self.deriv = deriv
        self.N = deriv.N
        self.bndry_data = bndry_data
        self.interior = bndry_data.interior
        self.bndry = bndry_data.bndry_points
        self.exterior = bndry_data.exterior
    
    def constructMat(self):
        bndry_points = self.bndry
        l_pt = bndry_points[0]
        r_pt = bndry_points[1]
        N = self.N
        proj_ext = projOp(N, self.exterior)
        proj_bndry = projOp(N, bndry_points)
        
        mat = self.deriv.fdMat()
        mat = -2*proj_ext+1/2*((identity(N) - proj_ext)@mat + mat@(identity(N) - proj_ext)) 
        mat += 1/2*projOp(N,[l_pt,r_pt-1])@shiftOps(N,1) + 1/2*(projOp(N,[l_pt+1,r_pt]))@shiftOps(N,-1)
        mat += -shiftOps(N,1)@projOp(N,[l_pt+1,r_pt]) - shiftOps(N,-1)@(projOp(N,[l_pt,r_pt-1]))
        mat += projOp(N, self.exterior)
        return mat.tocsr()
