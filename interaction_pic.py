from Deriv import *
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
from helpers import *

##1D EXAMPLE

N=32
periodic_1d = Derivatives(2,N,1,dim=1).fdMat()
projector = projOp(N,[0,N-1])
lam = 100
"""
mat = periodic_1d + 1j*lam*projector
b = 1/N*np.ones(N)
x = np.real(spsolve(mat, b))
plt.plot(x)
plt.show()
"""

##2D EXAMPLE
periodic_2d = Derivatives(2,N,1,dim=2).fdMat()
projector2d = kron(projector,identity(N)) + kron(identity(N),projector) 
proj_corners = projOp(N**2, [0, N-1, N**2 - N , N**2 -1])
projector2d -= proj_corners
#just checking to make sure I'm marking the right bndry points
"""
diag = projector2d.diagonal()
bndry = np.reshape(diag,(N,N))
plt.imshow(bndry)
plt.show()
"""
lam = 100
mat = periodic_2d + 1j*lam*projector2d
#uniform rhs
#b = 1/(N**2)*np.ones(N**2)
#point source
b = np.zeros(N**2)
b_grid = np.reshape(b, (N,N))
b_grid[N//2, N//2] = 1
b = np.reshape(b_grid, N**2)
x = np.real(spsolve(mat, b))
x_grid = np.reshape(x, (N,N))
plt.imshow(x_grid)
plt.show()

#You can verify by making lam bigger or smaller that the value at the boundary gets smaller