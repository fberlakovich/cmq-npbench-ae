def initialize():
  import numpy as np
  np.random.seed(0)
  N = 10000
  x = np.random.random(N) + 1j *  np.random.random(N)
  return N, x