#%%
import numpy as np
from Deriv import *
from Dirichlet import *


def naiveOdd(f, x,end_points=[0,1]):
    x_l = end_points[0]
    x_r = end_points[1]
    if  -x_r <= x < x_l:
        return -f(-x)
    else:
        return f(x)

def naiveEven(f, x, end_points = [0,1]):
    x_l = end_points[0]
    x_r = end_points[1]
    if  -x_r <= x < x_l:
        return f(-x)
    else:
        return f(x)



normalizer = 2.2522836210436354

def _mollifier(t: float) -> float:
    """reference bump function"""
    return normalizer*np.exp(1.0/(t*t - 1.0)) if abs(t) < 1.0 else 0.0

def _mollify(signal, k):
    """Discrete mollification of the given signal with window size `k`."""
    n = signal.size
    Mu = np.copy(signal)
    for j in range(k, n - k):
        Mu[j] = 1/k*sum(_mollifier(i/k)*signal[j - i] for i in range(1 - k, k))
    return Mu

def mollify(u, k: int):
    assert(type(k) == int)

    n = u.size
    padded = np.zeros(n + 2*k)
    #pad with same value of function at end point so as to avoid boundary effects 
    padded[:k] = u[0]
    padded[k:k + n] = u
    padded[k + n:] = u[-1]
    mollified = _mollify(padded, k)
    return mollified[k:n + k]



def oddExtendSmooth(func,N, end_points =[0,1], k=5):
    x_l = end_points[0]
    x_r = end_points[1]
    x_lp = x_l - x_r
    x_rp = x_r
    oddFunc = lambda x: naiveOdd(func, x, end_points=end_points)

    sharpVals = np.array([oddFunc(x) for x in np.linspace(x_lp,x_rp,2*N)])
    new = mollify(sharpVals, k)
    return new

def evenExtendSmooth(func,N, end_points =[0,1], k=5):
    x_l = end_points[0]
    x_r = end_points[1]
    x_lp = x_l - x_r
    x_rp = x_r
    evenFunc = lambda x: naiveEven(func, x, end_points=end_points)

    sharpVals = np.array([evenFunc(x) for x in np.linspace(x_lp,x_rp,2*N, endpoints=False)])
    new = mollify(sharpVals, k)
    return new



"""
#%%
###ODD EXTENSION EXAMPLE
N = 128
f = lambda x: np.exp(-x)
oddExtension = lambda x: naiveOdd(f, x, end_points=[0,2])
grid = np.linspace(-2,2,2*N)
oddVals = [oddExtension(x) for x in grid]

oddSmooth = oddExtendSmooth(f, N, end_points=[0,2], k=50)

fdMat = Derivatives(2, 2*N, p =6).fdMat()

u = spsolve(fdMat, oddVals)
norm = np.linalg.norm(u)
u /= norm


u_smooth = spsolve(fdMat, oddSmooth)
norm = np.linalg.norm(u_smooth)
u_smooth /= norm

plt.plot(grid, u)
plt.plot(grid, u_smooth)
plt.show()
# %%


#EVEN EXTENSION EXAMPLE
N = 128
f = lambda x: np.exp(-x)
evenExtension = lambda x: naiveEven(f, x, end_points=[0,4])
grid = np.linspace(-4,4,2*N)
evenVals = [evenExtension(x) for x in grid]

evenSmooth = evenExtendSmooth(f, N, end_points=[0,4], k=20)

fdMat = Derivatives(2, 2*N, p =6).fdMat()

u = spsolve(fdMat, evenVals)
norm = np.linalg.norm(u)
u /= norm


u_smooth = spsolve(fdMat, evenSmooth)
norm = np.linalg.norm(u_smooth)
u_smooth /= norm

plt.plot(grid, u)
plt.plot(grid, u_smooth)
plt.show()
#%%
"""