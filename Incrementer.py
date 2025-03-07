import numpy as np
import pennylane as qml
from pennylane import MultiControlledX

dev = qml.device("default.qubit", shots = 1)
bitArray2Int = lambda arr: sum(arr[j]*2**j for j in range(len(arr)))
int2bitArray = lambda intg, nq: [(intg >> i) & 1 for i in range(nq - 1, -1, -1)]


#All the above addition and subtraction circuits are operating modulo 2**n, where n is the number of qubits in the register
#Calculate closest power of 2 to the integer, can be greater or less than
def closest_power_of_2(num):
    l_pow = int(np.floor(np.log2(num)))
    r_pow = int(np.ceil(np.log2(num)))
    if l_pow == r_pow:
        return l_pow
    elif abs(2**l_pow - num) > abs(2**r_pow - num):
        return r_pow
    elif abs(2**l_pow - num) == abs(2**r_pow - num):
        return r_pow
    else:
        return l_pow
    
#Represnt integer j as signed summation of powers of 2
def _lc_pow2(nq, j):
    sign = 1
    if j < 0:
        sign = -1
        j = -j
    shifts = []
    k = closest_power_of_2(j)
    shifts.append((1*sign,k))
    j -= 2**k
    while j != 0:
        if j < 0:
            kp = closest_power_of_2(-j)
            shifts.append((-1*sign,kp))
            j += 2**kp
        else:
            kp = closest_power_of_2(j)
            shifts.append((1*sign,kp))
            j -= 2**kp
    return shifts

#increment by any power of 2
def _incpow2_(nq, pow):
    for j in [x for x in range(nq-pow-1)]:
        MultiControlledX(control_wires = [k for k in range(pow,nq-j-1)], wires = [nq-j-1])
    qml.X([pow])
#decrement by any power of 2
def _decpow2_(nq, pow):
    """
    for x in range(pow, nq):
        qml.X([x])
    _incpow2_(nq,pow)
    for x in range(pow, nq):
        qml.X([x])
    """
    for j in [x for x in range(nq-pow-1)]:
        MultiControlledX(control_wires = [k for k in range(pow,nq-j-1)], wires = [nq-j-1],control_values=[0]*len(range(pow,nq-j-1)))
    qml.X([pow])
    
#increment by 1
def _inc_(nq):
    _incpow2_(nq, 0)

#decrement by 1
def _dec_(nq):
    for x in range(nq):
        qml.X([x])
    _incpow2_(nq,0)
    for x in range(nq):
        qml.X([x])

def _inc_by_j_(nq, j):
    if j == 0:
        return
    shifts = _lc_pow2(nq, j)
    for sign, pow in shifts:
        if sign > 0:
            _incpow2_(nq, pow)
        else:
            _decpow2_(nq, pow)



def non_clif_counter(nq, j):
    shifts = _lc_pow2(nq, j)
    count = 0
    for _,pow in shifts:
        if pow < nq - 1:
            count += nq - pow - 1
    return count

def _inc_by_j_mod_L(nq, j, L):
    assert (1 < L <= 2**nq) and L == int(L)



@qml.qnode(dev)
def increment(nq, input = 0):
    bitstring = int2bitArray(input,nq)[::-1]

    for idx in range(nq):
        if bitstring[idx]:
            qml.X(wires = [idx])

    _inc_(nq)
    return qml.sample(wires = [x for x in range(nq)])

@qml.qnode(dev)
def decrement(nq, input = 0):
    bitstring = int2bitArray(input,nq)[::-1]

    for idx in range(nq):
        if bitstring[idx]:
            qml.X(wires = [idx])
    _dec_(nq)
    return qml.sample(wires = [x for x in range(nq)])

@qml.qnode(dev)
def increment_pow_2(nq, pow, input = 0):
    bitstring = int2bitArray(input,nq)[::-1]
    
    for idx in range(nq):
        if bitstring[idx]:
            qml.X(wires = [idx])

    _incpow2_(nq, pow)

    return qml.sample(wires = [x for x in range(nq)])

@qml.qnode(dev)
def decrement_pow_2(nq, pow, input = 0):
    bitstring = int2bitArray(input,nq)[::-1]

    for idx in range(nq):
        if bitstring[idx]:
            qml.X(wires = [idx])

    _decpow2_(nq, pow)

    return qml.sample(wires = [x for x in range(nq)])

@qml.qnode(dev)
def increment_by_const(nq, const, input = 0):
    bitstring = int2bitArray(input,nq)[::-1]

    for idx in range(nq):
        if bitstring[idx]:
            qml.X(wires = [idx])
    
    _inc_by_j_(nq, const)
    return qml.sample(wires=[x for x in range(nq)])


nq= 5
###TESTING
"""
for j in range(2**nq):
    for k in range(2**nq):
        arr = increment_by_const(nq, const = k, input = j)
        corr = (k+j)%2**nq
        out = bitArray2Int(arr)
        if corr - out != 0:
            print("ERROR on input: ", j, "shifted by ", k )

print(qml.draw(increment_by_const)(8,3))
"""