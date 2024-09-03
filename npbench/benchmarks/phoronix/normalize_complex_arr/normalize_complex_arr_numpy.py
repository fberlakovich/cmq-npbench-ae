import numpy as np
#from: https://stackoverflow.com/questions/41576536/normalizing-complex-values-in-numpy-python

#pythran export normalize_complex_arr(complex[])

def normalize_complex_arr(a):
    a_oo = a - a.real.min() - 1j*a.imag.min() # origin offsetted
    return a_oo/np.abs(a_oo).max()
