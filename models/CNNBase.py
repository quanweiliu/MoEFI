'''
first-conv: 3* 3x3 conv
layer:1-3
cross: identity
skip_connection: +
center: +
'''

import torch
from torch import nn
from torchvision import models
from functools import partial
import torch.nn.functional as F
# from modules import DropBlock2D, DropSpectral, AttentionPooling2D
# from modules import ModalitySpecificMoE, ModalityAwareMoE, Gated_Fusion, CrossAttentionFusion, BiModalCrossAttentionFusion

from .modules import DropBlock2D, DropSpectral, AttentionPooling2D
from .modules import ModalitySpecificMoE, ModalityAwareMoE, Gated_Fusion, CrossAttentionFusion, BiModalCrossAttentionFusion

nonlinearity = partial(F.relu, inplace=True)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

class Cross_identity(nn.Module):
    def __init__(self):
        super(Cross_identity, self).__init__()

    def forward(self,x, y):
        return x, y


class Model_base(nn.Module):
    def __init__(self, channell=3, channel2=1, is_pretrained="ResNet18_Weights.DEFAULT"):
    # def __init__(self, channell=3, channel2=1, is_pretrained="ResNet50_Weights.DEFAULT"):
        super(Model_base, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet18
        rgb_resnet = models.resnet18(weights=is_pretrained)
        lidar_resnet = models.resnet18(weights=is_pretrained)
        # rgb_resnet = models.resnet34(weights=is_pretrained)
        # lidar_resnet = models.resnet34(weights=is_pretrained)
        # print(rgb_resnet)

        # filters = [64, 256, 512, 2048]
        # rgb_resnet = models.resnet50(weights=is_pretrained)
        # lidar_resnet = models.resnet50(weights=is_pretrained)   
        # print(rgb_resnet)

        # rgb-decoder
        self.rgb_first = nn.Sequential(
            ConvBNReLU(channell, filters[0], ks=3, stride=1, padding=1),
        )

        self.rgb_encoder1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            rgb_resnet.layer1
        )

        self.rgb_encoder2 = rgb_resnet.layer2
        self.rgb_encoder3 = rgb_resnet.layer3
        self.rgb_encoder4 = rgb_resnet.layer4

        # lidar-decoder
        self.lidar_first = nn.Sequential(
            ConvBNReLU(channel2, filters[0], ks=3, stride=1, padding=1),
        )

        self.lidar_encoder1 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            lidar_resnet.layer1)

        self.lidar_encoder2 = lidar_resnet.layer2
        self.lidar_encoder3 = lidar_resnet.layer3
        self.lidar_encoder4 = lidar_resnet.layer4

        # cross_block
        self.cross_block0 = Cross_identity()
        self.cross_block1 = Cross_identity()
        self.cross_block2 = Cross_identity()
        self.cross_block3 = Cross_identity()

        self.dSpe = DropSpectral(0.1, mode="bernoulli")
        self.dSpa = DropBlock2D(0.1, block_size=3)

        # self.gated_fusion = Gated_Fusion(filters[3])
        self.SpMoE = ModalitySpecificMoE(filters[3])
        # self.crossAtten = CrossAttentionFusion(dim=filters[3], num_heads=8, dropout=0.1)
        # self.biModalCrossAtten = BiModalCrossAttentionFusion(dim=filters[3], num_heads=8, dropout=0.1, fuse_out_dim=filters[3])
        # self.AwMoE = ModalityAwareMoE(filters[3])

        # self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.APooling2D1 = AttentionPooling2D(filters[3])  # [B, C, H, W] -> [B, C]
        self.APooling2D2 = AttentionPooling2D(filters[3])  # [B, C, H, W] -> [B, C]
        self.APooling2D3 = AttentionPooling2D(filters[3])  # [B, C, H, W] -> [B, C]


    def get_visulization(self, x, y):
        x_first, y_first = self.cross_block0(self.rgb_first(x), self.lidar_first(y))
        # print("x_first", x_first.shape, "y_first", y_first.shape) # [64, 512, 11, 11]
        xe1, ye1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first))
        # print("xe1", xe1.shape, "ye1", ye1.shape) # [64, 512, 11, 11]
        xe2, ye2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1))
        # print("xe2", xe2.shape, "ye2", ye2.shape) # [64, 512, 6, 6]
        # xe2 = self.dSpe(xe2)
        # ye2 = self.dSpa(ye2)

        xe3, ye3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2))
        # print("xe3", xe3.shape, "ye3", ye3.shape) # [64, 512, 3, 3]

        xe4 = self.rgb_encoder4(xe3)
        ye4 = self.lidar_encoder4(ye3)
        # print("xe4", xe4.shape, "ye4", ye4.shape) # [64, 512, 2, 2]

        ## center
        # center = xe4 + ye4
        # center = self.gated_fusion(xe4, ye4)
        # center = self.AwMoE(xe4, ye4)
        center = self.SpMoE(xe4, ye4)
        # center = self.crossAtten(xe4, ye4)  # ye4 attends to xe4
        # center = self.biModalCrossAtten(xe4, ye4)
        # print(center.shape)

        return xe4, ye4, center
    

    def forward(self, x, y):
        x_first, y_first = self.cross_block0(self.rgb_first(x), self.lidar_first(y))
        # print("x_first", x_first.shape, "y_first", y_first.shape) # [64, 512, 11, 11]
        xe1, ye1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first))
        # print("xe1", xe1.shape, "ye1", ye1.shape) # [64, 512, 11, 11]
        xe2, ye2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1))
        # print("xe2", xe2.shape, "ye2", ye2.shape) # [64, 512, 6, 6]
        xe2 = self.dSpe(xe2)
        ye2 = self.dSpa(ye2)

        xe3, ye3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2))
        # print("xe3", xe3.shape, "ye3", ye3.shape) # [64, 512, 3, 3]

        xe4 = self.rgb_encoder4(xe3)
        ye4 = self.lidar_encoder4(ye3)
        # print("xe4", xe4.shape, "ye4", ye4.shape) # [64, 512, 2, 2]

        ## center
        # center = xe4 + ye4
        # center = self.gated_fusion(xe4, ye4)
        # print("center1", center.shape)   # 64, 512, 2, 2
        # center = self.AwMoE(xe4, ye4)
        center = self.SpMoE(xe4, ye4)
        # center = self.crossAtten(xe4, ye4)  # ye4 attends to xe4
        # print("center2", center.shape)   # 64, 512, 2 ,2
        # center = self.biModalCrossAtten(xe4, ye4)
        # print(center.shape)

        # xoutput = self.avgpool2(xe4).flatten(1)  # [batch, 512]
        # youtput = self.avgpool3(ye4).flatten(1)
        # output = self.avgpool1(center).flatten(1) 

        xoutput = self.APooling2D1(xe4)  # [batch, 512]
        youtput = self.APooling2D2(ye4)
        output = self.APooling2D3(center)

        return xoutput, youtput, output


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(4, 15, 11, 11, device=device)
    y = torch.randn(4, 1, 11, 11, device=device)

    model = Model_base(channell=15, channel2=1).to(device)
    # print(model)
    # output, x_put, y_put = model(x, y)
    xoutput, youtput, output = model(x, y)
    print("output", xoutput.shape, youtput.shape, output.shape)

