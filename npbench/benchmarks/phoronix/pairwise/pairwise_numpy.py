#from: http://people.duke.edu/~ccc14/sta-663-2016/03A_Numbers.html#Example:-Calculating-pairwise-distance-matrix-using-broadcasting-and-vectorization

#pythran export pairwise(float [][])

import numpy as np
def pairwise(pts):
    return np.sum((pts[None,:] - pts[:, None])**2, -1)**0.5
