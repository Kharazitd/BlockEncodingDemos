import numpy as np
from scipy.sparse import identity, kron, diags, coo_matrix

def directSum(matList):
    dims = [mat.shape[0] for mat in matList]
    leftDims = lambda i : int(np.prod(dims[0:i]))
    rightDims = lambda i : int(np.prod(dims[i+1:]))
    i = 0
    dilated_mats = []
    for mat in matList:
        if i == 0:
            dilated_mats.append(kron(mat, identity(rightDims(0))))
        elif i == len(dims):
            dilated_mats.append(kron(identity(leftDims(-1)), mat))
        else:
            dilated_mats.append(kron(kron(identity(leftDims(i)),mat),identity(rightDims(i))))
        i+=1
    return sum(dilated_mats)

def projOp(N, idxs):
    assert max(idxs) < N
    diagVals = np.zeros(N)
    diagVals[idxs] = 1
    proj = diags(diagVals, offsets = 0, format='csr')
    return proj

def projComp(N, idxs):
    return identity(N) - projOp(N,idxs)
def reflectOp(N, idxs):
    return identity(N) - 2*projOp(N,idxs)


def pltDomain(inside, bndry, outside):
    # Unpack into x and y coordinates
    x_values, y_values = zip(*bndry)
    x_int, y_int = zip(*inside)
    x_out, y_out = zip(*outside)


    # Plot
    plt.scatter(x_values, y_values, color="blue", marker="o", label="bndry")
    plt.scatter(x_int, y_int, color="red", marker="o", label="interior")
    plt.scatter(x_out, y_out, color="green", marker="o", label="outside")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Ordered Pairs")
    plt.legend()
    plt.grid(True)

    plt.show()


def shiftOps(N,j):
    rows = np.arange(N)
    cols = (rows + j) % N
    data = np.ones(N)
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr() 


def vec2grid(vec, dim):
    k = np.log2(len(vec))
    print(k)
    N = int(2**(k/dim))
    print(N)
    print(len(vec))
    #make tuple of d dimension
    grid_soln = vec.reshape((N,)*dim)
    return grid_soln