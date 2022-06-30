"""
F.Cross_entropy(input, target)函数中包含了s o f t m a x softmaxsoftmax和l o g的操作，即网络计算送入的input参数不需要进行这两个操作。
例如在分类问题中，input表示为一个torch.Size([N, C])的矩阵，其中，N为样本的个数，C是类别的个数，input[i][j]可以理解为第i个样本的类别为j的Scores，
Scores值越大，类别为j的可能性越高，


torch.nn.functional.cross_entropy(input, target, weight=None, size_average=True)

input表示为一个torch.Size([N, C])的矩阵，其中，N为样本的个数，C是类别的个数
target : N

"""