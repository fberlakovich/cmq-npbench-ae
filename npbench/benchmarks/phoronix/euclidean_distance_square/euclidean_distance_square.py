def initialize():
  import numpy as np
  np.random.seed(0)
  x1 = np.random.random((1, 512))
  x2 = np.random.random((10000, 512))
  return x1, x2