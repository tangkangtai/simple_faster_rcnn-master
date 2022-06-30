import torch
import torch.nn as nn
import torch.nn.functional as F

# max_sent_len=35, batch_size=50, embedding_dim=300
conv2 = nn.Conv2d(in_channels=1, out_channels=100, kernel_size=(3, 300))

"""
torch.randn(*size,out=None)——>返回张量
返回一个张量，包含了从标准正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数，形状由可变参数sizes定义
"""
# batch_size x 1 × max_sent_len x embedding_dim
input = torch.randn(50, 1, 35, 300)

print("input:", input.size())
output = conv2(input)
# batch_size × kernel_num × H × 1，其中H=max_sent_len-kernel_size+1
print("output:", output.size())

# 最大池化
# pool = nn.MaxPool1d(kernel_size=35-3+1)
output = output.squeeze(3)
pool1d_value = F.max_pool1d(output, output.size(2))
print("最大池化输出：", pool1d_value.size())

# 全连接
fc = nn.Linear(in_features=100, out_features=2)
fc_inp = pool1d_value.view(-1, pool1d_value.size(1))
print("全连接输入：", fc_inp.size())
fc_outp = fc(fc_inp)
print("全连接输出：", fc_outp.size())
# softmax
out = F.softmax(fc_outp, dim=1)
print("输出结果值：", out)
