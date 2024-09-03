def initialize():
  import numpy as np
  N = 500000
  np.random.seed(0)
  values = np.array(np.random.randint(0,3298,size=N),dtype='u4')
  values.sort()
  return N, values