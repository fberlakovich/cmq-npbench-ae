def initialize():
  import numpy as np
  grid_shape = (512, 512)
  grid = np.zeros(grid_shape)
  block_low = int(grid_shape[0] * .4)
  block_high = int(grid_shape[0] * .5)
  grid[block_low:block_high, block_low:block_high] = 0.005
  return grid_shape, grid, block_low, block_high, grid[block_low:block_high, block_low:block_high]