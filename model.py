#coding:utf8

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw
import numpy as np


# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

featur_cfg = ''


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        # VGG-16
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._make_layers(cfg)
        self._rpn_model()

        size = (7, 7)
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                """    
                torch.nn.MaxPool2d(kernel_size, stride=None, padding=0...)
                对于输入信号的输入通道，提供2维最大池化（max pooling）操作
                kernel_size(int or tuple) - max pooling的窗口大小
                stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
                padding(int or tuple, optional) - 输入的每一条边补充0的层数
                """
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                """
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation..)      
                in_channels(int) – 输入信号的通道
                out_channels(int) – 卷积产生的通道
                kerner_size(int or tuple) - 卷积核的尺寸
                stride(int or tuple, optional) - 卷积步长
                padding(int or tuple, optional) - 输入的每一条边补充0的层数
                dilation: 用于控制内核点之间的距离
                ----------------------------------------------------------------------------
                torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
                num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x height x width'
                eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
                momentum： 动态均值和动态方差所使用的动量。默认为0.1。
                ---------------------------------------------------------------------------
                torch.nn.ReLU(inplace=False)
                对输入运用修正线性单元函数${ReLU}(x)= max(0, x)$
                 inplace-选择是否进行覆盖运算
                """
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)]

        # torch.nn.Sequential(* args)
        # 一个时序容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict。
        return nn.Sequential(*layers)
        # return layers

    def _rpn_model(self, mid_channels=512, in_channels=512, n_anchor=9):
        """
        RPN网络首先经过3x3卷积，再分别生成positive anchors和对应bounding box regression偏移量，
        然后计算出proposals；
        """
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        # I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        # ------------------
        #   权重和偏置初始化
        # -----------------
        # conv sliding layer
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()

        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, data):

        out_map = self.features(data)
        # for layer in self.features:
        #     # print layer
        #     data = layer(data)
        #     # print data.data.shape
        #
        # # out = data.view(data.size(0), -1)
        # self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1) #前面函数的,自己加的
        x = self.rpn_conv(out_map)

        # self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0) 前面函数里面的，自己加的
        pred_anchor_locs = self.reg_layer(x)  # 回归层，计算有效anchor转为目标框的四个系数
        pred_cls_scores = self.cls_layer(x)  # 分类层，判断该anchor是否可以捕获目标

        return out_map, pred_anchor_locs, pred_cls_scores

    def roi_classifier(self, class_num=20):  # 假设为VOC数据集，共20分类

        # torch.nn.Linear(in_features,out_features) 用于设置网络中的全连接层的
        # 全连接层的输入与输出一般都设置为二维张量 [batch_size, size]
        # in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
        # out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数。
        # 从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量。
        # 分类层                                      #  (7*7)[proposal feature map]*512
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096),
                                                   nn.ReLU(),
                                                   nn.Linear(4096, 4096),
                                                   nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num+1) * 4)  # (VOC 20 classes + 1 background. Each will have 4 co-ordinates)

        # self.cls_loc 坐标分类模型
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()

        # self.score 类别分类模型
        self.score = nn.Linear(4096, class_num+1)  # (VOC 20 classes + 1 background)

    def rpn_loss(self, rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label, weight=10.0):
        # 对与classification我们使用Cross Entropy损失
        gt_rpn_label = torch.autograd.Variable(gt_rpn_label.long()) # torch.autograd.Variable用来包裹张量并记录应用的操作

        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        # print(rpn_cls_loss)  # Variable containing: 0.6931

        # 对于 Regression 我们使用smooth L1 损失
        pos = gt_rpn_label.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        # print(mask.shape)  # (22500L, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = rpn_loc[mask].view(-1, 4) # .view 类似tf.resize函数
        mask_target_loc = gt_rpn_loc[mask].view(-1, 4)
        # print(mask_pred_loc.shape, mask_target_loc.shape)  # ((18L, 4L), (18L, 4L))

        # regression损失应用如下
        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        # print x.shape  # (18, 4)
        # print (x < 1)
        rpn_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))
        # print rpn_loc_loss.shape  # (18, 4)
        rpn_loc_loss = rpn_loc_loss.sum()  # 1.1628926242031001
        # print rpn_loc_loss
        # print rpn_loc_loss.shape
        # rpn_loc_loss = np.squeeze(rpn_loc_loss)
        # print rpn_loc_loss

        N_reg = (gt_rpn_label > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())

        # print"N_reg: {}, {}".format(N_reg, N_reg.shape)
        rpn_loc_loss = rpn_loc_loss / N_reg
        rpn_loc_loss = np.float32(rpn_loc_loss)
        # rpn_loc_loss = torch.autograd.Variable(torch.from_numpy(rpn_loc_loss))

        rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
        # print "rpn_cls_loss: {}".format(rpn_cls_loss)  # 0.693146109581
        # print 'rpn_loc_loss: {}'.format(rpn_loc_loss)  # 0.0646051466465
        rpn_loss = rpn_cls_loss + (weight * rpn_loc_loss)
        # print("rpn_loss: {}".format(rpn_loss))  # 1.33919757605
        return rpn_loss

    def roi_loss(self, pre_loc, pre_conf, target_loc, target_conf, weight=10.0):
        # 分类损失
        target_conf = torch.autograd.Variable(target_conf.long())
        pred_conf_loss = torch.nn.functional.cross_entropy(pre_conf, target_conf, ignore_index=-1)
        # print(pred_conf_loss)  # Variable containing:  3.0515

        #  对于 Regression 我们使用smooth L1 损失
        # 用计算RPN网络回归损失的方法计算回归损失
        # pre_loc_loss = REGLoss(pre_loc, target_loc)
        pos = target_conf.data > 0  # Regression 损失也被应用在有正标签的边界区域中
        mask = pos.unsqueeze(1).expand_as(pre_loc)  # (128, 4L)

        # 现在取有正数标签的边界区域
        mask_pred_loc = pre_loc[mask].view(-1, 4)
        mask_target_loc = target_loc[mask].view(-1, 4)
        # print(mask_pred_loc.shape, mask_target_loc.shape)  # ((19L, 4L), (19L, 4L))

        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        # print x.shape  # (19, 4)

        pre_loc_loss = ((x < 1) * 0.5 * x ** 2) + ((x >= 1) * (x - 0.5))
        # print(pre_loc_loss.sum())  # 1.4645805211187053

        N_reg = (target_conf > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())
        pre_loc_loss = pre_loc_loss.sum() / N_reg
        pre_loc_loss = np.float32(pre_loc_loss)
        # print pre_loc_loss  # 0.077294916
        # pre_loc_loss = torch.autograd.Variable(torch.from_numpy(pre_loc_loss))
        # 损失总和
        pred_conf_loss = np.squeeze(pred_conf_loss.data.numpy())
        total_loss = pred_conf_loss + (weight * pre_loc_loss)

        return total_loss


if __name__ == '__main__':
    vgg = VGG()
    print(vgg)
    # torch.randn(*size....)返回一个符合均值为0，方差为1的正态分布
    data = torch.randn((1, 3, 800, 800))
    print(data.shape)
    data = torch.autograd.Variable(data)
    out = vgg.forward(data)
    print(out.data.shape)


