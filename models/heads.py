import torch
from torch import nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.g1 = nn.Sequential(
                               nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
        self.g2 = nn.Sequential(
                               nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
        self.g3 = nn.Sequential(
                               nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
    def forward(self, x, y, z=None):
        x = self.g1(x)
        y = self.g2(y)
        if z is not None:
            z = self.g3(z)
        return x, y, z


class DINOHead2(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()

        self.g1 = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(start_dim=1),
                                nn.Linear(in_dim, 256, bias=False), 
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
        self.g2 = nn.Sequential(
                                nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(start_dim=1),
                                nn.Linear(in_dim, 256, bias=False), 
                                nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
    def forward(self, x, y):
        x = self.g1(x)
        y = self.g2(y)
        return x, y


class MLP_head(nn.Module):
    def __init__(self, in_dim=512, class_num=16):
        super(MLP_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                               nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num, bias=True))   #2048
    def forward(self, x):
        x = self.c(x)
        return x


class FDGC_head(nn.Module):
    def __init__(self, in_dim=128, class_num=16):
        super(FDGC_head, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 1024),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(1024),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(1024, 256),
                               nn.BatchNorm1d(256),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num))   #2048
    def forward(self, x):
        x = self.c(x)
        return x


class linearHead(nn.Module):
    def __init__(self, in_dim=128, class_num=16):
        super(linearHead, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 256),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(256),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num))
    def forward(self, x):
        x = self.c(x)
        return x
    


class Conv_head(nn.Module):
    def __init__(self, in_dim=128, class_num=16):
        super(Conv_head, self).__init__()
        self.c = nn.Sequential(
            nn.Conv2d(in_dim, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Conv2d(64, class_num, 3, padding=1),
            nn.Flatten()
        )
    def forward(self, x):
        x = self.c(x)
        return x


class MS2_head(nn.Module):
    def __init__(self, in_dim=256, class_num=16):
        super(MS2_head, self).__init__()

        self.out1 = nn.Linear(in_dim, class_num)
        self.out2 = nn.Linear(in_dim, class_num)
        self.out3 = nn.Linear(in_dim, class_num)

    def forward(self, x1, x2):
        out1 = self.out1(x1)
        out2 = self.out2(x2)
        x = x1 + x2
        out3 = self.out3(x)
        
        return out1, out2, out3
    

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, in_channels))  # [1, C]
        self.scale = in_channels ** -0.5

    def forward(self, x):
        # x: [B, C, H, W] → reshape → [B, C, N]
        B, C, H, W = x.size()
        N = H * W
        x_flat = x.view(B, C, N)                      # [B, C, N]
        x_flat_t = x_flat.permute(0, 2, 1)            # [B, N, C]

        q = self.query.expand(B, -1)                  # [B, C]
        attn = torch.bmm(x_flat_t, q.unsqueeze(2)).squeeze(-1)  # [B, N]
        attn = F.softmax(attn * self.scale, dim=-1)             # [B, N]
        pooled = torch.bmm(x_flat, attn.unsqueeze(-1)).squeeze(-1)  # [B, C]
        return pooled


class gate_head(nn.Module):
    def __init__(self, in_dim=256, class_num=16):
        super(gate_head, self).__init__()

        self.pool = AttentionPooling(in_dim*2)
        self.classifier = nn.Linear(in_dim*2, class_num)

    def forward(self, x):
        pooled = self.pool(x)                     # [B, D]
        logits = self.classifier(pooled)          # [B, num_classes]
        return logits
