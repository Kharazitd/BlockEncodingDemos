from Deriv import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
import numpy as np
from helpers import *
from scipy.integrate import RK45

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
"""
periodic_2d = Derivatives(2,N,1,dim=2).fdMat()
projector2d = kron(projector,identity(N)) + kron(identity(N),projector) 
proj_corners = projOp(N**2, [0, N-1, N**2 - N , N**2 -1])
projector2d -= proj_corners
#just checking to make sure I'm marking the right bndry points

diag = projector2d.diagonal()
bndry = np.reshape(diag,(N,N))
plt.imshow(bndry)
plt.show()

lam = 100
mat = periodic_2d + 1j*lam*projector2d
#uniform rhs
#b = 1/(N**2)*np.ones(N**2)
#point source
b = np.zeros(N**2)
b_grid = np.reshape(b, (N,N))
b_grid[N//2, N//2] = 50
b = np.reshape(b_grid, N**2)
x = np.real(spsolve(mat, b))
x_grid = np.reshape(x, (N,N))
plt.imshow(x_grid)
plt.show()
"""
#You can verify by making lam bigger or smaller that the value at the boundary gets smaller

###TIME STEPPING 2d
N = 32
lam = 1000
#laplacian (scale-free)
periodic_2d = Derivatives(2,N,1,dim=2).fdMat()
#project onto boundary
projector2d = kron(projector,identity(N)) + kron(identity(N),projector) 
proj_corners = projOp(N**2, [0, N-1, N**2 - N , N**2 -1])
projector2d -= proj_corners

diag = projector2d.diagonal()
kappa = 1
L = kappa*periodic_2d
#rotation matrix
U = lambda t: diags(np.exp(1.j*lam*t*diag),0)
#interaction pic hamiltonian
A = lambda t: U(t)@L@U(-t)
#ode function
f = lambda t, x : A(t)@x

#construct initial state
width = 30
gaussian = lambda x, y: np.exp(-(x-N//2)**2/width)*np.exp(-(y-N//2)**2/width)
x_grid = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        x_grid[i,j] = gaussian(i,j)
x0 = np.reshape(x_grid, (N**2)) + 0.j * np.ones(N**2)

#step size, dt, simulation time T, solve with RK45
dt = .01
T = 3
sol = RK45(f, 0, x0, T, dt)

n_steps = int(T*(1/dt))
t_array = []
val_array = []
soln_bndry = []
for i in range(n_steps):
    sol.step()
    t_array.append(sol.t)
    #convert back from interaction picture 
    bndry_points = projector2d.diagonal().nonzero()
    #vals = U(-sol.t)@sol.y
    #seems to work better if you don't do this though?
    vals = sol.y
    #collect boundary values
    bndry_vals = np.zeros(N**2)
    bndry_vals[bndry_points] =  vals[bndry_points]
    soln_bndry.append(np.reshape(bndry_vals, (N,N)))
    #collect full solution
    val_grid = np.reshape(vals, (N,N))
    val_array.append(val_grid)
"""
#PLOT ANIMATED SOLUTION OVER TIME
fig, ax = plt.subplots()
cax = ax.imshow(np.real(val_array[0]), cmap='viridis')
fig.colorbar(cax,)

def update(frame):
    cax.set_array(np.real(val_array[frame]))
    ax.set_title(f'time_step {frame*dt:.2f}')
    return [cax]

anim = FuncAnimation(fig, update, frames=len(val_array), interval=5, blit=False)
plt.show()"""

#PLOT ANIMATED BOUNDARY VALUES OVER TIME
fig, ax = plt.subplots()
cax = ax.imshow(np.real(soln_bndry[0]), cmap='viridis')
fig.colorbar(cax,)

def update(frame):
    cax.set_array(np.real(soln_bndry[frame]))
    ax.set_title(f'lambda = {lam}, t = {frame*dt:.2f}')
    return [cax]

anim = FuncAnimation(fig, update, frames=len(soln_bndry), interval=5, blit=False)
plt.show()
