import pitensor as pt
import numpy as np
from pitensor.nn.layers import Linear

pi_tensor = pt.Tensor([
    [1, 2, 3], 
    [4, 5, 6],
])

np_array = np.array([
    [1, 2, 3], 
    [4, 5, 6],
])

if __name__ == '__main__':
    prod = pi_tensor @ pi_tensor.T
    print(prod, type(prod), type(prod[0]))
    print(type(Linear(11,11)))
    print(np_array)