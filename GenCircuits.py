import numpy as np
import pennylane as qml
from pennylane.transforms import decompose
from pennylane import MultiControlledX
from helpers import *
from Deriv import *
from functools import partial
import matplotlib.pyplot as plt
from Incrementer import _lc_pow2, closest_power_of_2, _incpow2_, _decpow2_, _inc_by_j_








def periodicLCU(N, p=1, display_circ = False, show_BE = False):
    nq = int(np.log2(N))
    ops = [ qml.prod(_inc_by_j_)(nq,k) for k in range(-p, p+1, 1)]
    deriv = Derivatives(2, N, p)
    coeffs = deriv.fdCoeffs()
    normalization = np.sum(np.abs(coeffs))
    lcu = qml.dot(coeffs, ops)
    anc = int(np.ceil(np.log2(2*p+1)))
    dev = qml.device("default.qubit")
    register = qml.registers({"system": nq, "ancilla": anc})
    
    @qml.qnode(dev)
    def circuit(lcu):
        qml.PrepSelPrep(lcu, register["ancilla"])
        return qml.state()
    if display_circ:
        print(qml.draw(circuit,level = 2,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))
    if show_BE:
        output_matrix = normalization*np.real(qml.matrix(circuit,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))[0:2**nq, 0:2**nq]
        plt.imshow(output_matrix, cmap='magma')
        plt.show()


#Generates LCU for Dirichlet BCs, but only in 1d and with fixed endpoints
#Can readily be extended by using the construction provided in Dirichlet.py

def dirichletLCU(N, display_circ = False, show_BE = False):
    nq = int(np.log2(N))
    ops = [ qml.prod(_inc_by_j_)(nq,k) for k in range(-1, 2, 1)]

    anc = 3
    dev = qml.device("default.qubit")
    register = qml.registers({"system": nq, "ancilla": anc})
    
    #add terms for enforcing BCs, switched ordering from different indexing 
    reflectBCL = reflectOp(N, [N-1]).diagonal()
    reflectBCR = reflectOp(N,[0]).diagonal()
    opL = qml.DiagonalQubitUnitary(reflectBCL, wires = [x for x in range(nq)] )
    opR = qml.DiagonalQubitUnitary(reflectBCR, wires =[x for x in range(nq)] )
    shiftL = qml.prod(opL, qml.prod(_inc_by_j_)(nq,-1))
    shiftR = qml.prod(opR, qml.prod(_inc_by_j_)(nq,1))
    ops.append(shiftL)
    ops.append(shiftR)

    coeffs = [1/2, -2, 1/2, 1/2, 1/2]
    normalization = np.sum(np.abs(coeffs))

    lcu = qml.dot(coeffs, ops)
    @qml.qnode(dev)
    def circuit(lcu):
        qml.PrepSelPrep(lcu, register["ancilla"])
        return qml.state()
    if display_circ:
        print(qml.draw(circuit,level = 2,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))
    if show_BE:
        output_matrix = np.real(qml.matrix(circuit,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))[0:2**nq, 0:2**nq]
        plt.imshow(normalization*output_matrix, cmap='magma')
        plt.show()


def neumannLCU(N, display_circ=False, show_BE = False):
    nq = int(np.log2(N))
    ops = [ qml.prod(_inc_by_j_)(nq,k) for k in range(-1, 2, 1)]
    coeffs = [1/2, -3/2, 1/2]
    reflectBCs = reflectOp(N, [0,N-1]).diagonal()
    refBndryOp = qml.DiagonalQubitUnitary(reflectBCs, wires = [x for x in range(nq)] )
    ops.append(refBndryOp)
    coeffs.append(-1/2)

    #add terms for enforcing BCs, switched ordering from different indexing 
    reflectBCL = reflectOp(N, [N-1]).diagonal()
    reflectBCR = reflectOp(N,[0]).diagonal()
    opL = qml.DiagonalQubitUnitary(reflectBCL, wires = [x for x in range(nq)] )
    opR = qml.DiagonalQubitUnitary(reflectBCR, wires =[x for x in range(nq)] )
    shiftL = qml.prod(opL, qml.prod(_inc_by_j_)(nq,-1))
    shiftR = qml.prod(opR, qml.prod(_inc_by_j_)(nq,1))
    ops.append(shiftL)
    ops.append(shiftR)

    coeffs.append(1/2)
    coeffs.append(1/2) 
    normalization = np.sum(np.abs(coeffs))

    anc = int(np.ceil(np.log2(len(coeffs))))
    dev = qml.device("default.qubit")
    register = qml.registers({"system": nq, "ancilla": anc})
    
    lcu = qml.dot(coeffs, ops)
    @qml.qnode(dev)
    def circuit(lcu):
        qml.PrepSelPrep(lcu, register["ancilla"])
        return qml.state()
    if display_circ:
        print(qml.draw(circuit,level = 2,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))
    if show_BE:
        output_matrix = np.real(qml.matrix(circuit,wire_order=[x for x in range(nq,nq+anc+1)][::-1] + [x for x in range(nq)][::-1])(lcu))[0:2**nq, 0:2**nq]
        plt.imshow(normalization*output_matrix, cmap='magma')
        plt.show()

class PosOpLCU:
    def __init__(self, N):
        self

