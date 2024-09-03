def initialize():
  N = 10000
  import numpy as np
  np.random.seed(0)
  t0, p0, t1, p1 = np.random.randn(N), np.random.randn(N), np.random.randn(N), np.random.randn(N)
  return N, t0, p0, t1, p1