# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:49:51 2020

@author: 86186
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, utils

import numpy as np
import spectral as spy
import matplotlib.pyplot as plt


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28),#
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(), # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(), # ToTensor()能够把灰度范围从0-255变换到0-1之间
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x1, x2):
        # print("lengh", len(x))    # 2 / 10
        # convert to list
        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]

        # 处理多尺度输入，将多个尺度的输入逐个经过相同的特征提取器(backbone)，
        # 然后将提取的特征进行合并。
        # torch.cumsum() 沿着最后一个维度，计算累计和。
        # torch.unique_consecutive 仅仅消除连续的重复值。
        # 只是为了产生一个 2 和 10
        idx_crops = torch.cumsum(
                    torch.unique_consecutive(
                        torch.tensor([inp.shape[-1] for inp in x1]),
                        return_counts=True,)[1], 0)

        start_idx, output = 0, torch.empty(0).to(x1[0].device)
        x_put, y_put = torch.empty(0).to(x1[0].device), torch.empty(0).to(x1[0].device)
        # print("idx_crops", idx_crops, "output", output.shape)   # tensor([2]) torch.Size([0])

        # idx_crops 是累计和，为什么不是每一个，而采用累计和
        for end_idx in idx_crops:
            # print("end_idx", end_idx)             # 2             / 2          , 4（为什么这里一下次输进去这么多）
            # idx_crops 分别是 【2】， 【2，10】
            input1 = torch.cat(x1[start_idx: end_idx]) 
            input2 = torch.cat(x2[start_idx: end_idx]) 
            # print("input2", input1.shape, input2.shape)  
            _out, _x, _y = self.backbone(input1, input2)

            # ([256, 512, 1, 1]) / ([256, 512, 2, 2]) / ([256, 512, 2, 2])
            # print("_out", _out.shape, _x.shape, _y.shape)

            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]

            # accumulate outputs, empty 的输出cat之后直接没有
            output = torch.cat((output, _out))
            x_put = torch.cat((x_put, _x))
            y_put = torch.cat((y_put, _y))
            start_idx = end_idx

        # Run the head forward on the concatenated features.

        # ([256, 512, 1, 1]) / ([256, 512, 2, 2]) / ([256, 512, 2, 2])
        # print("output", output.shape, x_put.shape, y_put.shape)            # [128*4, 512]
        
        # out = self.head(output)
        # # print("out", out.shape)                  # [128*4, 128]
        # # print("*"*10)

        return output, x_put, y_put


class MultiCropWrapper_MS2(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper_MS2, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x1, x2):
        # print("lengh", len(x))    # 2 / 10
        # convert to list
        if not isinstance(x1, list):
            x1 = [x1]
            x2 = [x2]

        # 处理多尺度输入，将多个尺度的输入逐个经过相同的特征提取器(backbone)，
        # 然后将提取的特征进行合并。
        # torch.cumsum() 沿着最后一个维度，计算累计和。
        # torch.unique_consecutive 仅仅消除连续的重复值。
        # 只是为了产生一个 2 和 10
        idx_crops = torch.cumsum(
                    torch.unique_consecutive(
                        torch.tensor([inp.shape[-1] for inp in x1]),
                        return_counts=True,)[1], 0)

        start_idx, output = 0, torch.empty(0).to(x1[0].device)
        x_put, y_put = torch.empty(0).to(x1[0].device), torch.empty(0).to(x1[0].device)
        # print("idx_crops", idx_crops, "output", output.shape)   # tensor([2]) torch.Size([0])

        # idx_crops 是累计和，为什么不是每一个，而采用累计和
        for end_idx in idx_crops:
            # print("end_idx", end_idx)             # 2             / 2          , 4（为什么这里一下次输进去这么多）
            # idx_crops 分别是 【2】， 【2，10】
            input1 = torch.cat(x1[start_idx: end_idx]) 
            input2 = torch.cat(x2[start_idx: end_idx]) 
            # print("input2", input1.shape, input2.shape)  
            _x, _y = self.backbone(input1, input2)

            # ([256, 512, 1, 1]) / ([256, 512, 2, 2]) / ([256, 512, 2, 2])
            # print("_out", _out.shape, _x.shape, _y.shape)

            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            # if isinstance(_out, tuple):
            #     _out = _out[0]

            # accumulate outputs, empty 的输出cat之后直接没有
            # output = torch.cat((output, _out))
            x_put = torch.cat((x_put, _x))
            y_put = torch.cat((y_put, _y))
            start_idx = end_idx

        # Run the head forward on the concatenated features.

        # ([256, 512, 1, 1]) / ([256, 512, 2, 2]) / ([256, 512, 2, 2])
        # print("output", output.shape, x_put.shape, y_put.shape)            # [128*4, 512]
        
        # out = self.head(output)
        # # print("out", out.shape)                  # [128*4, 128]
        # # print("*"*10)

        return x_put, y_put
        # return output, x_put, y_put


def save_weights(train_loss, test_loss, best_loss, best_acc, \
                 test_accuracy, epoch, model, super_head, contra_head, optimizer, args):


    if test_loss < best_loss:
        best_loss = test_loss
        torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "super_head": super_head.state_dict(),
                "contra_head": contra_head.state_dict(),
                "optimizer": optimizer.state_dict()}, 
                os.path.join(args.result_dir, "test_loss.pth"))


            
    return best_loss, best_acc


def draw(label, name, scale: float = 4.0, dpi: int = 400, save_img=True):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)