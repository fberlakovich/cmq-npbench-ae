def initialize():
  import numpy as np
  lx,ly=(2**7,2**7)
  u=np.zeros([lx,ly],dtype=np.double)
  u[lx//2,ly//2]=1000.0
  tempU=np.zeros([lx,ly],dtype=np.double)
  return lx, ly, u, u[lx//2, ly//2], tempU