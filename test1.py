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
    anchor = torch.arange(24).reshape(-1,4)
    print(anchor)
    height = anchor[:, 2] - anchor[:, 0]
    width = anchor[:, 3] - anchor[:, 1]
    print(anchor[:,2])
    print("height = ",height)
    print("width = ", width)

if __name__ == '__main__':
    #utils_get_pos_neg_sample_test()
    #where_test()
    #random_choice_test()
    #random_choice_test1()
    anchor_test()