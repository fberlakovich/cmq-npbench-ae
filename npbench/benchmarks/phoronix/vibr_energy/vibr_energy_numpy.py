#from: http://stackoverflow.com/questions/17112550/python-and-numba-for-vectorized-functions

#pythran export vibr_energy(float64[], float64[], float64[])
import numpy
def vibr_energy(harmonic, anharmonic, i):
    return numpy.exp(-harmonic * i - anharmonic * (i ** 2))
