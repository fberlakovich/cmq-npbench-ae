def initialize():
  import numpy as np
  d = 10
  re = 5
  params = (d, re, np.ones((2*d, d+1, re)), np.ones((d, d+1, re)),  np.ones((d, 2*d)), np.ones((d, 2*d)), np.ones((d+1, re, d)), np.ones((d+1, re, d)), 1)
  return d, re, params