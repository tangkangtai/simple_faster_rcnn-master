import numpy as np
import torch

def torch_permute_test():                               #h  w
    tensor_1 = torch.arange(2 * 4 * 5 * 5).reshape(1, 8, 5, 5) # 8个5*5的tensor
    tensor_1_permute = tensor_1.permute(0, 2, 3, 1) # (1,5,5,8)
    print('tensor_1_permute: ')
    print(tensor_1_permute)


def torch_cat_test():
    tensor_1 = torch.arange(3)
    tensor_2 = torch.arange(12).reshape(4,3)
    print('tensor_1.shape: ', tensor_1.shape)
    print('tensor_1: ')
    print(tensor_1)
    print('tensor_2: ')
    print(tensor_2)
    # tensor_1_1 = tensor_1[:, None]
    # print('tensor_1_1.shape: ',tensor_1_1.shape)
    print('tensor_1_1: ')
    # print(tensor_1_1)
    cat = torch.cat([tensor_1[None, :], tensor_2], dim=0)
    print("cat: ")
    print(cat)

def array_test_axis_1():
    tensor_1 = torch.arange(20).reshape(-1, 5)
    print('tensor_1: ')
    print(tensor_1)
    axis_tensor_1 = tensor_1[:, [0, 2, 1, 4, 3]]
    axis_tensor_2 = tensor_1[[0, 2, 1, 3], :]
    print('axis_tensor_1: ')
    print(axis_tensor_1)
    print('axis_tensor_2: ')
    print(axis_tensor_2)

def np_ones_test():
    import numpy as np
    # reshape和resize 都可以改变数组的形状，但是reshape不改变原有数组的数据，resize可以改变原数组的数据
    sample_roi = np.arange(128*4).reshape(-1,4)
    rois = torch.from_numpy(sample_roi).float()
    roi_indices = 0 * np.ones((len(rois),), dtype=np.int32) # 128个0的数组
    print(roi_indices)
    roi_indices = torch.from_numpy(roi_indices).float()
    print(roi_indices)

def stack_test():
    pass

def array_test_3():
    import numpy as np
    reshape_1 = torch.arange(12).reshape(-1, 4)
    # label = np.array([0,0,1])
    # label = np.arange(3)
    # labels = torch.from_numpy(label).long()
    # mask = labels.unsqueeze(1).expand_as(reshape_1)
    print(reshape_1)
    # print(mask)
    # reshape_1_2 = reshape_1[mask]
    # print(reshape_1_2)
    # print(reshape_1_2.shape)
#---------------------------------------------------
    # label = np.array([[0, 2, 1], [2, 0, 1]])
    # labels = torch.from_numpy(label).long()
    # print(labels)
    # reshape_1_2 = reshape_1[labels]
    # print(reshape_1_2)
#     -------------------------------------------
#     label = np.array([False, True, False])
#     labels = torch.from_numpy(label)
#
#     mask = labels.unsqueeze(1).expand_as(reshape_1)
#     print(mask)
#     reshape_1_2 = reshape_1[mask]
#     print(reshape_1_2)

def array_test_4():

    tensor1 = torch.arange(2 * 3 * 4).reshape(2, 3, 4) # 2个3*4的tensor
    print('tensor1: ')
    print(tensor1)
    n_sample = tensor1.shape[0]
    tensor2 = torch.arange(2)
    print('torch.arange(0,n_sample).long(): ', torch.arange(0,n_sample).long())
    print('tensor2: ', tensor2)
    #   即 取tensor1的  第i个tensor里面的， 第j个数据
    tensor1_tensor2 = tensor1[torch.arange(0,n_sample).long(), tensor2]
    print('tensor1_tensor2: ', tensor1_tensor2)


def squeeze_test():
    """
    从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    numpy.squeeze(a,axis = None)
     1）a表示输入的数组；
    2）axis用于指定需要删除的维度，但是指定的维度必须为单维度，否则将会报错；
    3）axis的取值可为None 或 int 或 tuple of ints, 可选。若axis为空，则删除所有单维度的条目；
    4）返回值：数组
    5) 不会修改原数组；
    """
    e = np.arange(10).reshape(1, 1, 10)
    print('e.shape: ', e.shape)
    # array([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]])
    a = np.squeeze(e)
    print('a.shape: ', a.shape)
    # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == '__main__':

    # torch_permute_test()
    # torch_cat_test()
    # array_test_axis_1()
    # np_ones_test()
    array_test_3()
    # squeeze_test()
    # array_test_4()