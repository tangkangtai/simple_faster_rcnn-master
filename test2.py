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
    array1 = np.arange(16).reshape(-1,2) # (8,2)

    print('打乱前：', array1)
    # random.shuffle(array1)
    np.random.shuffle(array1)
    print('打乱后: ', array1)
    array1[3][1] = 15
    argmax_index = np.argmax(array1, axis=0)
    print('array1在axis=0列操作中argmax最大值index: ', argmax_index)
    argmax = array1[argmax_index,np.arange(array1.shape[1])]
    print('array每列的最大值为：', argmax)

    array1_argmax = np.where(array1 == argmax)
    # array1_argmax = np.where(array1 == argmax)[0]
    print('值相等时坐标index: ', array1_argmax)

def vstack_test():
    dy = [0, 10, 100, 1000]
    dx = [1, 11, 111, 1111]
    dh = [2, 22, 222, 2222]
    dw = [3, 33, 333, 3333]
    anchor = np.vstack((dy,dx,dh,dw))
    anchor_transpose = np.vstack((dy,dx,dh,dw)).transpose()
    print('anchor:')
    print(anchor)
    print('anchor_transpose:')
    print(anchor_transpose)
def add_test():
    anchors = np.arange(20).reshape(5,-1)
    print('anchors.shape: ', anchors.shape)
    print('len(anchors): ', len(anchors))
    print('type(len(anchors)): ', type(len(anchors)))

    print('anchors.shape[1:]: ', anchors.shape[1:])
    print('type(anchoes.shape[1:]): ', type(anchors.shape[1:])) # tuple
    shape_1 = (len(anchors),) + anchors.shape[1:]
    print(shape_1)

def test_tuple():
    tuple_1 = 5 + (1,)
    print(tuple_1)

def torch_permute_test():                               #h  w
    tensor_1 = torch.arange(2 * 4 * 5 * 5).reshape(1, 8, 5, 5) # 8个5*5的tensor
    # print('tensor_1: ')
    # print(tensor_1)
    # # h w
    tensor_1_permute = tensor_1.permute(0, 2, 3, 1) # (1,5,5,8)
    print('tensor_1_permute: ')
    print(tensor_1_permute)

    # view = tensor_1_permute.contiguous().view(1, 5, 5, 2, 4)
    # print(view)
    # tensor_1_permute_view = tensor_1.permute(0, 2, 3, 1).contiguous().view(1,-1,4) # (1,5,5,8)
    # print('tensor_1_permute_view:')
    # print(tensor_1_permute_view)


def slice_test():
    reshape = torch.arange(1*2*3*4).reshape(1,4,3,2)
    print(reshape)
    print(reshape[:,:,:,1])


if __name__ == '__main__':
    # test_1()
    # axis0_And_axis1_test()
    # array_equal_test()
    # vstack_test()
    # add_test()
    # test_tuple()
    # torch_permute_test()
    slice_test()