import numpy as np
import random

def test1():
    """
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    # replace:True表示可以取相同数字，False表示不可以取相同数字
    # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。

    :return:
    """
    # ---------------------------------------------

    randint1 = np.random.randint(0, 6) # 从[0, 6)中随机输出一个随机数
    print(randint1)
    # 等价于 np.random.choice(6)
    choice1 = np.random.choice(6)
    print(choice1)
    randint2 = np.random.randint(0, 5, 3) #  #在[0, 5)内输出三个数字并组成一维数组（ndarray）
    print(randint2)
    choice2 = np.random.choice(5, 3 ,replace= False)
    print(choice2)
    # 相当于np.random.choice(5, 3)

    # ---------------------------------------------
    # anchors = 50
    # label = np.random.choice(np.arange(100), 10, replace=False)
    # print('label = ', label)
    #
    # valid_anchor_index = np.random.choice(np.arange(20), 10, replace=False) # replace 是否取相同数字，默认为True
    # print(valid_anchor_index)
    #
    # anchor_conf = np.empty(anchors, dtype=label.dtype)
    # anchor_conf.fill(-1)
    # print(anchor_conf)
    # anchor_conf[valid_anchor_index] = label
    # print('anchor_conf', anchor_conf)
    #

def test2():
    number1 = random.sample(range(20), 10)
    print(type(range(20))) # range为不可变序列，不可以修改
    print(type(np.arange(20)))
    number2 = np.random.random_sample(10) # 产生10个[0, 1)的均匀分布数值
    print(number1)
    print(number2)

    # 产生 [3,2] 数组的随机数值在[-5, 0)
    p = 5 * np.random.random_sample((3, 2)) - 5
    print(p)
    # pass

def test3():
    anchors = np.arange(44).reshape(11,-1)
    # print('anchors.shape = ', anchors.shape)
    # print('anchors.shape[1:] = ', anchors.shape[1:])
    # print('len(anchors,1) = ', (len(anchors),1)) # tuple
    # print('(len(anchors),) + anchors.shape[1:] = ', (len(anchors),) + anchors.shape[1:]) # tuple的加法

    anchor_locations = np.empty( (len(anchors),) + anchors.shape[1:] , dtype=np.uint32)
    print(anchor_locations.shape)
    anchor_locations.fill(0)

    anchor_locs = np.arange(30,54).reshape(6,-1)
    print(anchor_locs)
    #
    # print(anchor_locations) # 全0
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