def initialize():
  import numpy as np
  N = 20
  x = y = z = np.arange(0., N, 0.1)
  L = 4
  periodic = True
  return N, x, L, periodic