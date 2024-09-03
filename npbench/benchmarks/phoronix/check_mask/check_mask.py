def initialize():
  n=1000
  import numpy as np
  np.random.seed(0)
  db = np.array(np.random.randint(2, size=(n, 4)), dtype=bool)
  return n, db