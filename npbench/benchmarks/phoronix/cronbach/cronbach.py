def initialize():
  import numpy as np
  np.random.seed(0)
  N = 600
  items = np.random.rand(N,N)
  return N, items