#from: High Performance Python by Micha Gorelick and Ian Ozsvald, http://shop.oreilly.com/product/0636920028963.do

#pythran export evolve(float64[][], float)
import numpy as np
def laplacian(grid):
    return np.roll(grid, +1, 0) + np.roll(grid, -1, 0) + np.roll(grid, +1, 1) + np.roll(grid, -1, 1) - 4 * grid

def evolve(grid, dt, D=1):
    return grid + dt * D * laplacian(grid)
