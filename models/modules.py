
import torch
from torch import nn
import torch.nn.functional as F


# 快速 FGSM 攻击风格的扰动
class AdversarialPerturbation(nn.Module):
    def __init__(self, epsilon=0.01):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x, grad=None):
        if grad is None or not self.training:
            return x
        perturbation = self.epsilon * grad.sign()
        return torch.clamp(x + perturbation, 0.0, 1.0)  # 确保数值有效
    

class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=3):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x) # 表示每个像素有多大概率是一个 block 的中心
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float() # 选出所有符合要求的点作为block 中心点
            mask = mask.to(x.device)
            block_mask = self._compute_block_mask(mask)   # 扩展这些“中心点”为 block 大小的 mask
            out = x * block_mask[:, None, :, :]
            out = out * block_mask.numel() / block_mask.sum() # 被遮掉部分变成 0，那剩下的值就整体乘以一个系数，使期望值保持一致。
            return out


    def _compute_block_mask(self, mask):
        # 很巧妙：所有被中心点覆盖的 block_size×block_size 区域都被标为 1
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)



        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)   # 1 - 变成遮掉的部分为 1，保留的部分为 0
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


# 这个应用到特征级别的dropout，类似于DropBlock，但在光谱维度上进行。
class DropSpectral(nn.Module):
    def __init__(self, dropout_rate=0.1, mode="bernoulli"):  # or "fixed"
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mode = mode

    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            return x
        B, C, H, W = x.shape
        if self.mode == "bernoulli":
            # 按样本、按通道随机
            drop_mask = torch.rand(B, C, 1, 1, device=x.device) > self.dropout_rate


        elif self.mode == "fixed":  # fixed mode
            # 全 batch 丢相同通道
            drop_mask = torch.ones(C, device=x.device)
            num_drop = int(C * self.dropout_rate)
            drop_indices = torch.randperm(C)[:num_drop]
            drop_mask[drop_indices] = 0
            drop_mask = drop_mask.view(1, C, 1, 1).expand(B, -1, -1, -1)
        return x * drop_mask


class Gated_Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        gate = self.gate(out)

        PG = x * gate
        FG = y * (1 - gate)

        # return torch.cat([FG, PG], dim=1)
        return PG + FG  # 或者直接相加，取决于具体任务和实验结果
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)  # 合并 key 和 value
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, a, b):
        # a, b: [B, C, 2, 2]
        B, C, H, W = a.shape
        N = H * W

        a_flat = a.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        b_flat = b.view(B, C, N).permute(0, 2, 1)  # [B, N, C]

        q = self.q_proj(a_flat)                   # [B, N, C]
        k, v = self.kv_proj(b_flat).chunk(2, dim=-1)  # [B, N, C], [B, N, C]

        attn_out, _ = self.attn(q, k, v)          # [B, N, C]
        out = self.out_proj(attn_out)
        out = self.norm(out + a_flat)             # residual + norm

        return out.permute(0, 2, 1).view(B, C, H, W)



class BiModalCrossAttentionFusion(nn.Module):
    def __init__(self, dim=512, num_heads=8, dropout=0.1, fuse_out_dim=512):
        super().__init__()
        # HSI attends to Lidar/SAR
        self.attn_hsi = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.q_hsi = nn.Linear(dim, dim)
        self.kv_lidar = nn.Linear(dim, dim * 2)

        # Lidar/SAR attends to HSI
        self.attn_lidar = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.q_lidar = nn.Linear(dim, dim)
        self.kv_hsi = nn.Linear(dim, dim * 2)

        # 融合：concat 后压缩
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, fuse_out_dim, kernel_size=1),
            nn.BatchNorm2d(fuse_out_dim),
            nn.ReLU(inplace=True)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, hsi_feat, lidar_feat):
        # [B, C, 2, 2] -> [B, 4, C]
        B, C, H, W = hsi_feat.shape
        N = H * W

        hsi = hsi_feat.view(B, C, N).permute(0, 2, 1)   # [B, 4, 512]
        lidar = lidar_feat.view(B, C, N).permute(0, 2, 1)

        # HSI → attend Lidar
        q_h = self.q_hsi(hsi)
        kv_l = self.kv_lidar(lidar).chunk(2, dim=-1)
        hsi_attn_out, _ = self.attn_hsi(q_h, *kv_l)
        hsi_out = self.norm(hsi + hsi_attn_out)  # residual

        # Lidar → attend HSI
        q_l = self.q_lidar(lidar)
        kv_h = self.kv_hsi(hsi).chunk(2, dim=-1)
        lidar_attn_out, _ = self.attn_lidar(q_l, *kv_h)
        lidar_out = self.norm(lidar + lidar_attn_out)

        # concat and reshape
        hsi_out = hsi_out.permute(0, 2, 1).view(B, C, H, W)
        lidar_out = lidar_out.permute(0, 2, 1).view(B, C, H, W)

        fused = torch.cat([hsi_out, lidar_out], dim=1)  # [B, 1024, 2, 2]
        out = self.fuse(fused)  # [B, 512, 2, 2]
        return out


class ModalitySpecificMoE(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, rgb_experts=2, hsi_experts=2, shared_experts=2, top_k=1):
        super(ModalitySpecificMoE, self).__init__()
        self.rgb_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, in_channels, 1)
            ) for _ in range(rgb_experts)
        ])

        self.hsi_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, in_channels, 1)
            ) for _ in range(hsi_experts)
        ])


        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, in_channels, 1)
            ) for _ in range(shared_experts)
        ])

        self.rgb_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, rgb_experts)
        )

        self.hsi_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hsi_experts)
        )

        self.shared_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, shared_experts)
        )

        self.top_k = top_k


    def forward(self, x, y):
        B, C, H, W = x.size()
        fusion_input = torch.cat([x, y], dim=1)

        rgb_scores = self.rgb_gate(x)  # [B, rgb_experts]
        hsi_scores = self.hsi_gate(y)  # [B, hsi_experts]
        shared_scores = self.shared_gate(fusion_input)  # [B, shared_experts]

        fused = torch.zeros_like(x)
        # Top-k routing for each expert group

        rgb_topk_val, rgb_topk_idx = torch.topk(rgb_scores, self.top_k, dim=1)
        hsi_topk_val, hsi_topk_idx = torch.topk(hsi_scores, self.top_k, dim=1)
        shared_topk_val, shared_topk_idx = torch.topk(shared_scores, self.top_k, dim=1)

        for i in range(self.top_k):
            for b in range(B):
                rgb_weight = F.softmax(rgb_topk_val, dim=1)[b, i]
                hsi_weight = F.softmax(hsi_topk_val, dim=1)[b, i]
                shared_weight = F.softmax(shared_topk_val, dim=1)[b, i]

                rgb_out = self.rgb_experts[rgb_topk_idx[b, i]](x[b].unsqueeze(0))
                hsi_out = self.hsi_experts[hsi_topk_idx[b, i]](y[b].unsqueeze(0))
                shared_out = self.shared_experts[shared_topk_idx[b, i]](fusion_input[b].unsqueeze(0))
                fused[b] += rgb_weight * rgb_out.squeeze(0) + \
                            hsi_weight * hsi_out.squeeze(0) + \
                            shared_weight * shared_out.squeeze(0)

        return fused


class ModalityAwareMoE_sparse(nn.Module):
    def __init__(self, in_channels, num_experts=4, hidden_dim=256, k=2):
        super(ModalityAwareMoE_sparse, self).__init__()
        self.num_experts = num_experts
        self.k = k  # top-k sparse routing

        # Gating networks for each modality
        self.gate_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,C,1,1]
            nn.Flatten(),             # [B,C]
            nn.Linear(in_channels, num_experts)
        )
        self.gate_y = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_experts)
        )

        # Experts shared across modalities
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels * 2, hidden_dim, 1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, in_channels, 1)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, y):
        B, C, H, W = x.shape

        # Get gate scores for each modality
        gate_x_out = self.gate_x(x)  # [B, num_experts]
        gate_y_out = self.gate_y(y)  # [B, num_experts]

        # Combined gating
        gate_scores = gate_x_out + gate_y_out  # [B, num_experts]
        topk_val, topk_idx = torch.topk(gate_scores, self.k, dim=1)  # [B, k]
        weights = F.softmax(topk_val, dim=1)  # [B, k]

        # Prepare input for experts
        fusion_input = torch.cat([x, y], dim=1)  # [B, 2C, H, W]

        # Compute fused output using only top-k experts
        fused = torch.zeros_like(x)
        for i in range(self.k):
            idx = topk_idx[:, i]  # [B]
            weight = weights[:, i].view(B, 1, 1, 1)  # [B,1,1,1]
            for b in range(B):
                expert_out = self.experts[idx[b]](fusion_input[b].unsqueeze(0))  # [1,C,H,W]
                fused[b] += weight[b] * expert_out.squeeze(0)

        return fused


    

# if __name__=="__main__":
#     # Example usage


#     # import numpy as np
#     # model=SEBlock(128)
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#     x = torch.randn(4, 144, 28, 28, device=device)
#     y = torch.randint(low=0, high=10, size=(4, 28, 28), dtype=torch.long, device=device)

#     model = LabelSmoothSoftmaxCEV1().to(device)
#     # print(model)
#     # output, x_put, y_put = model(x, y)
#     loss = model(x, y)
#     print("loss", loss)


# if __name__ == "__main__":
#     # Example usage
#     x = torch.randn(2, 128, 8, 8)  # Example input tensor
#     y = torch.randn(2, 128, 8, 8)  # Another input tensor

#     gf = Gated_Fusion(in_channels=128)
#     cross = CrossAttentionFusion(dim=128)
#     bimodal_cross = BiModalCrossAttentionFusion(dim=128)
#     moe = ModalitySpecificMoE(in_channels=128)


#     output_moe = moe(x, y)
#     output_gt = gf(x, y)
#     output_cross = cross(x, y)
#     output_bimodal_cross = bimodal_cross(x, y)


#     print(output_moe.shape, output_gt.shape, output_cross.shape, output_bimodal_cross.shape)  # Should match the input shape