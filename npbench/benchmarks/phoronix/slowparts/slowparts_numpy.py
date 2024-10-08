#from: https://groups.google.com/forum/#!topic/parakeet-python/p-flp2kdE4U

#pythran export slowparts(int, int, float [][][], float [][][], float [][], float [][], float [][][], float [][][], int)
from numpy import zeros, power, tanh
def slowparts(d, re, preDz, preWz, SRW, RSW, yxV, xyU, resid):
    """ computes the linear algebra intensive part of the gradients of the grae
    """
    fprime = lambda x: 1 - power(tanh(x), 2)

    partialDU = zeros((d+1, re, 2*d, d))
    for k in range(2*d):
        for i in range(d):
            partialDU[:,:,k,i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i,k] + RSW[i,k]) * yxV[:,:,i]
    
    return partialDU
