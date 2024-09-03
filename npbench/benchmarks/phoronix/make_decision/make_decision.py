def initialize():
  import numpy as np, random
  np.random.seed(0)
  s=np.random.randn(2**16)+np.random.randn(2**16)*1.j
  sc=np.random.choice(s, 32)
  return s, sc