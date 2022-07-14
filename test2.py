import torch
import numpy as np
import random
def test_1():
    ctr = dict()
    ctr[0] = [-1,-1]
    ctr[0][1] = 8
    ctr[0][0] = 16
    print('ctr: ', ctr)

def axis0_And_axis1_test():
    array_1 = torch.arange(12).reshape(-1, 3)
    print('array_1: ')
    print(array_1)
    argmax_axis0 = array_1.argmax(axis=0)
    argmax_axis1 = array_1.argmax(axis=1)
    print('argmax_axis0: ', array_1.argmax(axis=0))
    print('argmax_axis1: ', array_1.argmax(axis=1))

    x = array_1[argmax_axis0, np.arange(array_1.shape[1])]

    print('x:')
    print(x)

def array_equal_test():
    array1 = np.arange(16).reshape(-1,2)
    random.shuffle(array1)
    print(array1)
if __name__ == '__main__':
    # test_1()
    # axis0_And_axis1_test()
    array_equal_test()