#from: http://stackoverflow.com/questions/13815719/creating-grid-with-numpy-performance
#pythran export create_grid(float [])
import numpy as np
def create_grid(x):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:,:,0] = x.reshape(-1,1)
    z[:,:,1] = x
    fast_grid = z.reshape(N*N, 3)
    return fast_grid
