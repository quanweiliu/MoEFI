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
# from torchsummary import summary
# from torchsummaryX import summary
from functools import partial
import torch.nn.functional as F

from .heads import AttentionPooling
from .modules import DropBlock2D, DropSpectral
from .modules import ModalitySpecificMoE, Gated_Fusion, CrossAttentionFusion, BiModalCrossAttentionFusion

# from heads import AttentionPooling
# from modules import DropBlock2D, DropSpectral
# from modules import ModalitySpecificMoE, Gated_Fusion, CrossAttentionFusion, BiModalCrossAttentionFusion

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


class ResNet(nn.Module):
    def __init__(self, channell=3, channel2=1, is_pretrained="ResNet18_Weights.DEFAULT"):
        super(ResNet, self).__init__()
        filters = [64, 128, 256, 512]  # ResNet18
        rgb_resnet = models.resnet18(weights=is_pretrained)
        lidar_resnet = models.resnet18(weights=is_pretrained)

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
        
        self.msmoe = ModalitySpecificMoE(filters[3])
        self.gated_fusion = Gated_Fusion(filters[3])
        self.cross_attention_fusion = CrossAttentionFusion()
        self.bimodal_cross_attention_fusion = BiModalCrossAttentionFusion()

        self.dSpe = DropBlock2D(0.5)
        self.dSpa = DropSpectral(0.5)

        self.Apooling2D1 = AttentionPooling(filters[3])
        self.Apooling2D2 = AttentionPooling(filters[3])
        self.Apooling2D3 = AttentionPooling(filters[3])
        
        # self.avgpool1 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.avgpool2 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.avgpool3 = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x, y):
        x_first, y_first = self.cross_block0(self.rgb_first(x), self.lidar_first(y))
        xe1, ye1 = self.cross_block1(self.rgb_encoder1(x_first), self.lidar_encoder1(y_first))
        xe2, ye2 = self.cross_block2(self.rgb_encoder2(xe1), self.lidar_encoder2(ye1))
        xe3, ye3 = self.cross_block3(self.rgb_encoder3(xe2), self.lidar_encoder3(ye2))

        # xe2 = self.dSpe(xe2)
        # ye2 = self.dSpa(ye2)
        
        xe4 = self.rgb_encoder4(xe3)
        ye4 = self.lidar_encoder4(ye3)

        ## center
        center = xe4 + ye4
        # center = self.gated_fusion(xe4, ye4)
        # center = self.msmoe(xe4, ye4)
        # center = self.cross_attention_fusion(xe4, ye4)
        # center = self.bimodal_cross_attention_fusion(xe4, ye4)    
        # print(center.shape)    # [1024, 512, 2, 2]

        # output = self.avgpool1(center)
        # xe4 = self.avgpool2(center)
        # ye4 = self.avgpool3(center)

        xoutput = self.Apooling2D1(xe4)  # [batch, 512]
        youtput = self.Apooling2D2(ye4)
        output = self.Apooling2D3(center)

        return xoutput, youtput, output


if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(4, 144, 28, 28, device=device)
    y = torch.randn(4, 3, 28, 28, device=device)

    model = ResNet(channell=144, channel2=3).to(device)
    # print(model)
    output, x_put, y_put = model(x, y)
    print("output", output.shape, x_put.shape, y_put.shape)

