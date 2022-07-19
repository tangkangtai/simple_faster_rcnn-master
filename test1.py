import torch
import numpy as np

def utils_get_pos_neg_sample_test():
    #ious1 = torch.arange(8).reshape(-1,2)
    ious2 = torch.randn(4,2)
    print(ious2)
    gt_argmax_ious = ious2.argmax(axis=0)
    print(gt_argmax_ious)
    print("iou2.shape[1] = ", ious2.shape[1])
    #print("np.arange(iou2.shape[1])",np.arange(ious2.shape[1]))
    gt_max_ious2 = ious2[gt_argmax_ious, np.arange(ious2.shape[1])]
    print(gt_max_ious2)
    gt_argmax_ious = np.where(ious2 == gt_max_ious2)
    gt_argmax_ious1 = np.where(ious2 == gt_max_ious2)[0]
    print(gt_argmax_ious)
    print(gt_argmax_ious1)
    print(gt_argmax_ious1.shape)

  # print(ious2.argmax(axis=0))
  # print(ious2.argmax(axis=1))
    #print(ious1)
    #print(ious1.argmax(axis=0))

def where_test():
    """
    当数组是二维数组时，满足条件的数组值返回的是值的位置索引，
    因此会有两组索引数组来表示值的位置，
    返回的第一个array表示行坐标，第二个array表示纵坐标，两者一一对应
    """
    b = np.arange(4 * 5).reshape(4, 5)

    print(b)

    print(np.where(b > 14))


def random_choice_test():
    valid_anchor_len = 8940
    x = np.random.randn(8940)
    # print(x.shape)
    label = np.empty((valid_anchor_len,),dtype=np.int32)
    label.fill(-1)
    label[x < 0] = 0
    label[x > 0] = 1

    pos = np.where(label == 1)
    pos1 = np.where(label == 1)[0]
    print(pos)
    print(pos1)

def random_choice_test1():
    x = np.random.randn(10)
    y = np.random.choice(x,5)
    print("x = ", x)
    print("y = ", y)

def anchor_test():
    """
    anchor: (x,4)
    :return:
    """
    #anchor = np.random.randn(6,4)
    anchor = torch.randn(24).reshape(-1,4)
    print(anchor)
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]

    base_ctr_y = anchor[:, 0] + 0.5 * height
    base_ctr_x = anchor[:, 1] + 0.5 * width

    # print(anchor[:,2])
    print("height = ", height)
    print("width = ", width)
    print("base_ctr_x = ", base_ctr_x)
    print("base_ctr_y = ", base_ctr_y)
    gt_roi_locs = np.vstack((base_ctr_y, base_ctr_x, height, width)).transpose()
    print(gt_roi_locs)

def test3():
    randn = torch.randn(size=(1, 2, 3, 2, 3))
    print(randn)
    randn1 = randn[:,:,:,:, 1]
    print(randn1)

def test4():
    pred_anchor_locs_numpy = torch.arange(24).reshape(-1, 4)
    print(pred_anchor_locs_numpy)
    # x = torch.arange(10)
    # print(x) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # print(x[0::4])  # tensor([0, 4, 8])
    dy = pred_anchor_locs_numpy[:, 0::4]
    dx = pred_anchor_locs_numpy[:, 1::4]
    dh = pred_anchor_locs_numpy[:, 2::4]
    dw = pred_anchor_locs_numpy[:, 3::4]
    print(dy)
    print(dx)
    print(dh)
    print(dw)


def test_np_newaxis():

    array = torch.randn(4,4)
    print(array.shape)
    print(array)

    array_add_aix = array[:, np.newaxis]
    print(array_add_aix.shape)
    print(array_add_aix)

def test6():
    anchor = torch.randn(24).reshape(-1, 4)
    print(anchor)
    height = anchor[:, 2] - anchor[:, 0]
    print(height)
    print(height.shape)
    anchor_add = height[:, np.newaxis]
    print(anchor_add)
    print(anchor_add.shape)

def slice_test():
    reshape = np.arange(100).reshape(-1, 4)
    print(reshape)
    # clip = np.clip(reshape, 4, 10)
    # print(clip)\
    # s = slice(0, 5, 2)
    # print(reshape[s])
    '''
    # slice(start, stop, step)
    # np.clip(a,a_min,a_max) 将将数组中的元素限制在a_min, a_max之间，
    # 大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min.
    '''
    slice_1 = reshape[:,slice(0, 4, 2)]
    slice_2 = reshape[:,slice(1, 4, 2)]
    print(slice_1)
    print(slice_2)

def test7():
    arange1 = np.arange(10)
    arange2 = np.arange(3, 10)
    arange3 = np.arange(3, 10, 2)
    arange4 = np.arange(3, 10, 0.5)
    print(arange1)
    print(arange2)
    print(arange3)
    print(arange4)
def dict_init_test():
    """
    字典初始化 几种方式
    :return:
    """
    x = dict({1: [8, 8], 2: [9, 9]})
    y = {1: [8, 8], 2: [9, 9]}
    z = dict([(1, [2, 2]), (2, [3, 3])])
    j = dict()
    j[1] = [2,2]
    j[0] = [1,1]
    for c in j:
        print('key: ', c)
        d = j[c]
        d1,d2 = d
        print('d1 = ', d1, ",d2 = ", d2)
    print(j)
    # print(z)
    # print(type(z))
    # print(type(y))
    # print(type(x))

def test8():
    # sqrt = np.sqrt(512)
    # print(sqrt)
    # print(round(sqrt))
    ratios = [0.5, 1, 2]
    for i in range(len(ratios)):
        print(np.sqrt(ratios[i]))
        print(np.sqrt(1.0 / ratios[i]))

def test9():
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]
    sub_sample = 16
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            # anchor_scales 是针对特征图的，所以需要乘以下采样"sub_sample"
            # sub_sample=16 # ratios = [0.5, 1, 2]
            # anchor_scales = [8, 16, 32]  # 该尺寸是针对特征图的

            # size_ratios = (w * h)/ratio ——> (16*16)/[0.5,1,2] = [512,256,128] 自己备注
            # ws = np.round(np.sqrt(size_ratios)) # ws: [23,16,11]   自己备注
            # hs = np.round(ws * ratios)          # hs: [12 16 22]   自己备注
            h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])  # h = w * ratios
            w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratios[i])
            print('h = ', h, ', w = ', w)

def test10():
    reshape = np.arange(12).reshape(-1, 2)
    print(reshape)
    argmax_axis_0 = np.argmax(reshape, axis=0)
    argmax_axis_1 = np.argmax(reshape, axis=1)
    print('argmax_axis_0 = ', argmax_axis_0)
    print('argmax_axis_1 = ', argmax_axis_1)

    np.random.shuffle(reshape)
    # print(reshape.shape)
    # print(len(reshape))
    print(reshape)
    print('---------------------------------')
    argmax_axis_0 = np.argmax(reshape, axis=0)
    argmax_axis_1 = np.argmax(reshape, axis=1)
    print('argmax_axis_0 = ', argmax_axis_0)
    print('argmax_axis_1 = ', argmax_axis_1)

def test11():
    # x = np.arange(2)
    # print(type((3,))) # 元组
    # x = (3,) + (4,)
    # print(x) # (3, 4)
    anchors1 = np.arange(24).reshape(-1,2,3)
    anchors2 = np.arange(24).reshape(-1,4)
    print(anchors1.shape[1:]) # (2,3)
    print(type(anchors2.shape[1:])) # (4,,)

def test12():
    """
    模拟 100张 4*4的张量，resize测试
    """
    tensor1 = torch.arange(18*4*4).reshape(1,18,4,4) # 10个4*4
    # print(tensor1.shape)
    # print(tensor1)
    # tensor2 = tensor1.permute(0, 2, 3, 1).contiguous() # 4个4*10
    tensor3 = tensor1.permute(0, 2, 3, 1).contiguous().view(1, 4, 4, 9, 2)
    tensor4 = tensor1.permute(0, 2, 3, 1).contiguous().view(1, 4, 4, 9, 2)[:, :, :, :, 1]
    print(tensor3.shape)
    print(tensor3)
    print(tensor4)

def test13():
    """
    列表和数组的区别
    :return:
    """
    x = np.array([6, 8])
    p = [0, 1, 1, 0, 1, 0, 1, 1]
    y = x[p]
    print(y)

def torch_narrow_test():
    data = torch.arange(12).reshape(-1,6)
    # torch.from_numpy() # 用来将数组array转换为张量
    # data_to_numpy = data.numpy() # 张量转成数组
    # data_to_list = data.tolist() # 张量转成列表 && 或者 list(data)
    # data.item() # 张量转成数值
    print('data: ', data)
    narrow1 = data.narrow(1, 3, 3)
    print("narrow_data: ", narrow1)

def slice_test_roi():
    reshape = torch.arange(5 * 20 * 20).reshape(-1, 20, 20) # tensor (5, 4, 4)
    print('reshape: ', reshape)
    im = reshape[..., 3:8, 4:6]
    print('im: ', im)
    print('im.shape: ', im.shape)

def max_pooling_test():
    maxpooling = torch.nn.AdaptiveMaxPool2d((2,2))

    input = torch.autograd.Variable(torch.randn(1, 20, 3, 4))
    print('input: ', input)
    print('maxpooling(input).shape', maxpooling(input).shape)
    print('maxpooling(input)[0]', maxpooling(input)[0])
    print('maxpooling(input)[0].data', maxpooling(input)[0].data)
    # print('maxpooling(input)', maxpooling(input))

def cat_test():
    list1 = []
    reshape = torch.arange(1 * 10 * 4 * 4).reshape(1, -1, 4, 4)
    for i in range(5):
        list1.append(reshape)

    print('reshape.shape: ', reshape.shape)
    print('len(append1): ', len(list1))

    # print(list1)
    list1 = torch.cat(list1, 0)
    print('list1.shape: ', list1.shape)

def test14():
    n_sample = 10
    reshape1 = torch.arange(10 * 5 * 4).reshape(10, 5, 4) # (10, 5, 4)
    reshape2 = torch.empty(10, dtype=np.int)
    print('reshape1: ', reshape1)
    reshape2.fill_(0)
    # print(reshape2)
    padding = torch.arange(6)
    reshape2[padding] = 3
    print('reshape2: ', reshape2)
    # # print(reshape)
    # print(torch.arange(6))
    # print(torch.arange(0, 6))
    reshape1 = reshape1[torch.arange(0, n_sample).long(), reshape2]
    print('reshape1.shape: ', reshape1.shape)
    print('reshape1: ', reshape1)

def test15():
    t1 = torch.arange(3 * 4 * 5).reshape(3, 4, 5) # 3个 4 * 5 tensor
    print(t1)
    print(t1[0,1]) # 第0个, 里面的 第 1(下标) 个 tensor

def test16():
    t_expand = torch.arange(40).reshape(-1,4)
    t1 = torch.arange(10)
    print('t1.shape: ', t1.shape)
    print('t1: ', t1)
    t1_unsqueeze = t1.unsqueeze(1)
    print('t1_squeeze.shape: ', t1_unsqueeze.shape)
    print('t1_squeeze: ', t1_unsqueeze)
    t1_unsqueeze_expand_as = t1_unsqueeze.expand_as(t_expand)
    print('t1_unsqueeze_expand_as.shape: ', t1_unsqueeze_expand_as.shape)
    print('t1_unsqueeze_expand_as: ', t1_unsqueeze_expand_as)

def test17():
    resize = torch.arange(20).reshape(1, 4, 5)
    print('resize.shape: ', resize.shape)
    print(resize[0])
    print(resize[0].data)

def test5():
    pred_anchor_locs_numpy = torch.arange(24).reshape(-1, 4)
    print(pred_anchor_locs_numpy)

    anchors = torch.arange(24).reshape(-1,4)
    anc_height = anchors[:, 2] - anchors[:, 0]
    print('anc_height.shape: ', anc_height.shape)
    print('anc_height: ',anc_height)
    print('=========================================')
    print('anc_height[:, np.newaxis]:')
    print(anc_height[:,np.newaxis])
    print(anc_height[:,np.newaxis].shape)
    dy = pred_anchor_locs_numpy[:, 0::4]
    # dx = pred_anchor_locs_numpy[:, 1::4]
    # dh = pred_anchor_locs_numpy[:, 2::4]
    # dw = pred_anchor_locs_numpy[:, 3::4]
    print('dy.shape: ',dy.shape)
    print('dy: ',dy)
    ctr_y = dy * anc_height[:, np.newaxis]
    print('ctr_y: ', ctr_y)

def tensor_mutiple_test():
    a = torch.tensor([[1, 2], [2, 3], [3, 4]])
    b = torch.tensor([[1, 2], [2, 3], [3, 4]])
    print('a: ')
    print(a)
    print('b: ')
    print(b)
    print('a * b = ')
    print(a * b)

    # 输出：
    # tensor([[1, 4],
    #         [4, 9],
    #         [9, 16]])
def new_axis_test():
    # anchors = torch.arange(24).reshape(-1, 4)
    # anc_height = anchors[:, 2] - anchors[:, 0]
    # print('anc_height.shape: ', anc_height.shape)
    # print('anc_height: ',anc_height)
    # print('=========================================')
    # print('anc_height[:, np.newaxis]:')
    # print(anc_height[:,np.newaxis])
    # print(anc_height[:,np.newaxis].shape)
##########################################################
    anchors = torch.arange(24).reshape(4, 6)
    print(anchors.shape)
    print(anchors)
    print('==================================')
    print(anchors[:, np.newaxis].shape)
    print(anchors[:, np.newaxis])
    print('==================================')
    print(anchors[:, :, np.newaxis].shape)
    print(anchors[:, :, np.newaxis])

def test_list_index():
    k = [1,2,3,4,5,6,7,8,9,10]
    k = k[:100]
    print(k)

if __name__ == '__main__':
    # utils_get_pos_neg_sample_test()
    # where_test()
    # random_choice_test()
    # random_choice_test1()
    # anchor_test()
    # test3()
    # test4()
    # test_np_newaxis()
    # test6()
    # slice_test()
    # test7()
    # dict_init_test()
    # test8()
    # test9()
    # test10()
    # test11()
    # test12()
    # test13()
    # torch_narrow_test()
    # slice_test_roi()
    # max_pooling_test()
    # cat_test()
    # test14()
    # test15()
    # test16()
    # test17()
    # test5()
    # tensor_mutiple_test()
    # new_axis_test()
    test_list_index()