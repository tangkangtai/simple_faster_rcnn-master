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



if __name__ == '__main__':

    # torch_permute_test()
    # torch_cat_test()
    array_test_axis_1()