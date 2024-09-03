#from: http://stackoverflow.com/questions/16541618/perform-a-reverse-cumulative-sum-on-a-numpy-array
#pythran export reverse_cumsum(float[])
import numpy as np
def reverse_cumsum(x):
    return np.cumsum(x[::-1])[::-1]
