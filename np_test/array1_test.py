import numpy as np
import random

def test1():
    anchors = 50
    label = np.random.choice(np.arange(100), 10)
    print('label = ', label)
    valid_anchor_index = np.random.choice(np.arange(20), 10)
    print(valid_anchor_index)
    anchor_conf = np.empty(anchors, dtype=label.dtype)
    anchor_conf.fill(-1)
    print(anchor_conf)
    anchor_conf[valid_anchor_index] = label
    print('anchor_conf', anchor_conf)

def test2():
    number1 = random.sample(range(20),10)
    print(type(range(20))) # range为不可变序列，不可以修改
    print(type(np.arange(20)))
    number2 = np.random.random_sample(10)
    print(number1)
    print(number2)
    # pass

def test3():
    anchors = np.arange(44).reshape(11,-1)
    # print('anchors.shape = ', anchors.shape)
    # print('anchors.shape[1:] = ', anchors.shape[1:])
    # print('len(anchors,1) = ', (len(anchors),1))
    # print('(len(anchors),) + anchors.shape[1:] = ', (len(anchors),) + anchors.shape[1:])
    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=np.int)
    # print(anchor_locations.shape)
    anchor_locations.fill(0)

    anchor_locs = np.arange(30,54).reshape(6,-1)
    print(anchor_locs)

    print(anchor_locations)
    valid_anchor_index = np.random.choice(np.arange(11), 6, replace=False)

    print(valid_anchor_index)
    # print(valid_anchor_index.shape)
    # print(type(valid_anchor_index))
    anchor_locations[valid_anchor_index, :] = anchor_locs
    print(anchor_locations)
    # print(anchor_locs[3,:])

if __name__ == '__main__':

    # test1()
    # test2()
    test3()