import torch

def torch_permute_test():                               #h  w
    tensor_1 = torch.arange(2 * 4 * 5 * 5).reshape(1, 8, 5, 5) # 8个5*5的tensor
    tensor_1_permute = tensor_1.permute(0, 2, 3, 1) # (1,5,5,8)
    print('tensor_1_permute: ')
    print(tensor_1_permute)


if __name__ == '__main__':

    torch_permute_test()