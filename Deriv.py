import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
from Dirichlet import *
from Neumann import *
from findiff import coefficients
import matplotlib.pyplot as plt
from itertools import product
#helper functions
from helpers import *


class Derivatives:
    """
        Can be used to generate matrix approximation to differential operator with periodic BC
    """
    def __init__(self, ord, N, p, dim = 1):
        assert 2*p >= ord
        self.ord = ord
        self.N = N
        self.p = p
        self.dim = dim

    def offsets(self):
        offsets = list(range(-self.p,self.p+1,1))
        parity = self.ord%2
        if parity:
            del offsets[p]
        return offsets

    def shiftOps(self,j):
        rows = np.arange(self.N)
        cols = (rows + j) % self.N
        data = np.ones(self.N)
        return coo_matrix((data, (rows, cols)), shape=(self.N, self.N)).tocsr() 

    def fdCoeffs(self):
        """Assuming a central difference scheme for everything here, with evenly spaced grid points
            The number of points used will be 2*p + (ord+1)%2
        """
        coeffs = coefficients(self.ord, offsets = self.offsets())['coefficients']
        return coeffs

    def fdMat(self):
        coeffs =self.fdCoeffs()
        shifts = [self.shiftOps(j) for j in self.offsets()]
        scaled_shifts = [c*S for c, S in zip(coeffs,shifts)]
        dim = self.dim
        if dim == 1:
            return sum(scaled_shifts)
        if dim > 1:
            vs_dim = self.N**dim 
            if vs_dim >= 1e6:
                user_input = input("Are you sure you want to construct a {vs_dim}-dimensional matrix? Press anything to continue or 'crtl+c' to exit the program.")
                mat = sum(scaled_shifts)
                dmat = [mat]*dim
                return directSum(dmat)
            else:
                mat = sum(scaled_shifts)
                dmat = [mat]*dim
                return directSum(dmat)    
    def LCU(self):
        coeffs = self.fdCoeffs()
        p = self.p
        dim = self.dim
        if dim == 1:
            return [(coeffs[j],f"S^({j}-p)") for j in range(len(coeffs)) ]
        else:
            return [(coeffs[j]/np.sqrt(dim),f"S_{d}^({j}-p)") for j in range(len(coeffs)) for d in range(dim)]



