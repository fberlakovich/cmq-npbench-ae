#from http://stackoverflow.com/questions/77999777799977/numpy-vs-cython-speed
#pythran export multiple_sum(float[][])
import numpy as np
def multiple_sum(array):

    rows = array.shape[0]
    cols = array.shape[1]

    out = np.zeros((rows, cols))

    for row in range(0, rows):
        out[row, :] = np.sum(array - array[row, :], 0)

    return out
