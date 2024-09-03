def initialize():
  import numpy as np
  N = 70
  A = np.random.rand(N,N)
  B =  np.random.rand(N,N)
  W = np.random.rand(N,N)
  return N, A, B, W