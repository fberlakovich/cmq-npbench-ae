def initialize():
  import numpy as np
  shape = (5,4,3,2)
  x = np.arange(120, dtype=np.float64).reshape(*shape)
  return shape, x