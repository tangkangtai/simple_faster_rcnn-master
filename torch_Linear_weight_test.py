import torch
from torch.nn.parameter import Parameter

"""
torch.nn.Linear(in_features, out_features, bias=True) 对输入数据做线性变换：y=Ax+b
in_features: 每个输入样本的大小
out_features: 每个输出样本的大小
bias 若设置为False 这层不会学习偏置。默认为True
输入: (N,in_features)
输出： (N,out_features)

y=x * A(转置) + b, x为输入层(所谓的w 权重)

默认: weight参数采用了He-uniform初始化策略，而bias采用了简单的均匀分布初始化策略（均匀分布参数根据特征数计算）
"""
def Linear_test():
    import numpy as np
    output_ = torch.arange(8, dtype=torch.float32).reshape(4, 2) # 定义一个2*2张量
    print(output_)
    linear = torch.nn.Linear(2, 3)

    weight_ = torch.arange(6, dtype=torch.float32).reshape(3, 2)


    linear.weight = Parameter(weight_)
    linear.bias.data.zero_()
    # linear.weight.data = torch.zeros(2,3)
    print('linear.weight: ', linear.weight) # 默认初始化为
    print('linear.bias: ', linear.bias)
    full_linear = linear(output_)
    print(full_linear)


if __name__ == '__main__':
    Linear_test()