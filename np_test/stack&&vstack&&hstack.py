import numpy as np


def satck_test():
    a = [[1, 2, 3],
         [4, 5, 6]]
    # print(a)
    #==================================================
    # a_ = np.stack(a, axis=0)
    # print(a_)
    # print(a_.shape)
    # a_ = np.stack(a, axis=1)
    # print(a_)
    # print(a_.shape)

#============================================================
    # arrays = [np.random.randn(3, 4) for _ in range(10)] # _起到循环标志作用, 即生成 10个(3,4)数组
    # print(arrays)
    # print(arrays[0].shape)
#--------------------------------------------------------
    b = [[1, 2, 3],
         [4, 5, 6]]
    c = [[1, 2, 3],
         [4, 5, 6]]
    print("a=", a)
    print("b=", b)
    print("c=", c)
    d = np.stack((a, b, c), axis=0)
    print('axis=0后shape = ', d.shape)
    print('"axis=0":\n', d)
    d = np.stack((a, b, c), axis=1)
    print('axis=1后shape = ', d.shape)
    print('"axis=1":\n', d)
    d = np.stack((a, b, c), axis=2)
    print('axis=2后shape = ', d.shape)
    print('"axis=2":\n', d)

def hstack1():
    """
    Horizontal 水平 (column wise)
    tup：ndarrays数组序列，除了一维数组的堆叠可以是不同长度外，
    其它数组堆叠时，除了第二个轴的长度可以不同外，其它轴的长度必须相同
    原因在于一维数组进行堆叠是按照第一个轴进行堆叠的，其他数组堆叠都是按照第二个轴堆叠的
    """
    # 一维数组
    a = np.array((1, 2, 3))
    b = np.array((2, 3, 4))
    print(np.hstack((a, b)))

def hstack2():
    a = [[1, 2, 3],
         [4, 5, 6]]
    b = [[1, 2, 3],
         [4, 5, 6]]
    c = [[1, 2, 3],
         [4, 5, 6]]
    print(np.hstack((a, b, c)))


def vstack1():
    """
    Vertical 垂直 (row wise)
    沿着第一个轴堆叠数组
    tup：ndarrays数组序列，如果是一维数组进行堆叠，则数组长度必须相同；
    除此之外，其它数组堆叠时，除数组第一个轴的长度可以不同，其它轴长度必须一样
    """
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print(np.vstack((a,b)).shape) #(2,3)
    print(np.vstack((a, b)))
    print('转置后: ')
    print(np.vstack((a, b)).transpose())

def vstack2():
    a = np.array([[1], [2], [3]])
    b = np.array([[2], [3], [4]])
    print(np.vstack((a, b)))

def vstack3():
    import numpy as np
    a = [[1, 2, 3],
         [4, 5, 6]]
    b = [[1, 2, 3],
         [4, 5, 6]]
    c = [[1, 2, 3],
         [4, 5, 6]]
    print(np.vstack((a, b, c)))


if __name__ == '__main__':
    # satck_test()
    # vstack1()
    # vstack2()
    vstack3()
    # hstack2()

    # hstack1()
    # pass