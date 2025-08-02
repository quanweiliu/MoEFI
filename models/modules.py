import torch
from torch import nn
import torch.nn.functional as F


# Heads
class DINOHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.g1 = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
        self.g2 = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
        
        self.g3 = nn.Sequential(nn.Linear(in_dim, 256, bias=False), 
                               nn.BatchNorm1d(256),
                                nn.ReLU(inplace=True), 
                                nn.Linear(256, out_dim, bias=True))
    def forward(self, x, y, output=None):
        x = self.g1(x)
        y = self.g2(y)
        if output is not None:
            output = self.g3(output)
        return x, y, output
    

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


class FDGCHead(nn.Module):
    def __init__(self, in_dim=128, class_num=16):
        super(FDGCHead, self).__init__()
        self.c = nn.Sequential(nn.Linear(in_dim, 512),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(512),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(512, 256),
                               nn.Dropout(0.5),
                               nn.BatchNorm1d(256),
                            #    nn.ReLU(inplace=True), 
                               nn.Linear(256, class_num))   #2048
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
    

class AttentionPooling1D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, embed_dim))  # [1, D]
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        # x: [B, D, N]
        B, D, N = x.shape
        q = self.query.expand(B, -1)                          # [B, D]
        k = x.permute(0, 2, 1)                                # [B, N, D]
        attn = torch.bmm(k, q.unsqueeze(2)).squeeze(-1)       # [B, N]
        attn = F.softmax(attn * self.scale, dim=-1)           # [B, N]
        pooled = torch.bmm(x, attn.unsqueeze(-1)).squeeze(-1) # [B, D]
        return pooled


class AttentionPooling2D(nn.Module):
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


# perturbation
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


# fusion modules
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



# class Gated_Fusion(nn.Module):

#     def __init__(self, in_channels):
#         super().__init__()

#         self.gate = nn.Sequential(
#             nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
#             nn.Sigmoid(),
#         )

#     def forward(self, x, y):
#         out = torch.cat([x, y], dim=1)
#         G = self.gate(out)

#         PG = x * G
#         FG = y * (1 - G)

#         return torch.cat([FG, PG], dim=1)


class Gated_Fusion(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        G = self.gate(out)

        PG = x * G
        FG = y * (1 - G)

        fusion = torch.cat([FG, PG], dim=1)
        fusion = self.conv(fusion)

        return fusion

        
class ModalityAwareMoE(nn.Module):
    def __init__(self, in_channels, num_experts=4, hidden_dim=256, k=2):
        super(ModalityAwareMoE, self).__init__()
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


# class ModalitySpecificMoE(nn.Module):
#     def __init__(self, in_channels, hidden_dim=256, rgb_experts=2, hsi_experts=2, shared_experts=2, top_k=1):
#         super(ModalitySpecificMoE, self).__init__()
#         self.rgb_experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_dim, 1),
#                 nn.ReLU(),
#                 nn.Conv2d(hidden_dim, in_channels, 1)
#             ) for _ in range(rgb_experts)
#         ])

#         self.hsi_experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_dim, 1),
#                 nn.ReLU(),
#                 nn.Conv2d(hidden_dim, in_channels, 1)
#             ) for _ in range(hsi_experts)
#         ])

#         self.shared_experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels * 2, hidden_dim, 1),
#                 nn.ReLU(),
#                 nn.Conv2d(hidden_dim, in_channels, 1)
#             ) for _ in range(shared_experts)
#         ])

#         self.rgb_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, rgb_experts)
#         )

#         self.hsi_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels, hsi_experts)
#         )

#         self.shared_gate = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(in_channels * 2, shared_experts)
#         )

#         self.top_k = top_k

#     def forward(self, x, y):
#         B, C, H, W = x.size()
#         fusion_input = torch.cat([x, y], dim=1)

#         rgb_scores = self.rgb_gate(x)  # [B, rgb_experts]
#         hsi_scores = self.hsi_gate(y)  # [B, hsi_experts]
#         shared_scores = self.shared_gate(fusion_input)  # [B, shared_experts]

#         fused = torch.zeros_like(x)

#         # Top-k routing for each expert group
#         rgb_topk_val, rgb_topk_idx = torch.topk(rgb_scores, self.top_k, dim=1)
#         hsi_topk_val, hsi_topk_idx = torch.topk(hsi_scores, self.top_k, dim=1)
#         shared_topk_val, shared_topk_idx = torch.topk(shared_scores, self.top_k, dim=1)

#         for i in range(self.top_k):
#             for b in range(B):
#                 rgb_weight = F.softmax(rgb_topk_val, dim=1)[b, i]
#                 hsi_weight = F.softmax(hsi_topk_val, dim=1)[b, i]
#                 shared_weight = F.softmax(shared_topk_val, dim=1)[b, i]

#                 rgb_out = self.rgb_experts[rgb_topk_idx[b, i]](x[b].unsqueeze(0))
#                 hsi_out = self.hsi_experts[hsi_topk_idx[b, i]](y[b].unsqueeze(0))
#                 shared_out = self.shared_experts[shared_topk_idx[b, i]](fusion_input[b].unsqueeze(0))

#                 fused[b] += rgb_weight * rgb_out.squeeze(0) + \
#                             hsi_weight * hsi_out.squeeze(0) + \
#                             shared_weight * shared_out.squeeze(0)

#         return fused
    

class ModalitySpecificMoE(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, rgb_experts=2, hsi_experts=2, shared_experts=2, top_k=1):
        super(ModalitySpecificMoE, self).__init__()
        self.in_channels = in_channels
        self.top_k = top_k

        # Expert definitions remain the same
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
        
        # Gating network definitions remain the same
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

    def _forward_moe(self, inputs, experts, gate):
        """
        Helper function for vectorized Mixture of Experts forward pass.
        """
        # 1. Get routing scores and select top-k experts
        gate_scores = gate(inputs) # [B, num_experts]
        topk_weights, topk_indices = torch.topk(gate_scores, self.top_k, dim=1) # [B, top_k]
        topk_weights = F.softmax(topk_weights, dim=1) # [B, top_k]

        # 2. Prepare for batch processing
        B, C_in, H, W = inputs.shape
        # The output channel count is defined by the experts
        C_out = self.in_channels 
        
        # Flatten indices and weights for processing
        flat_indices = topk_indices.flatten() # [B * top_k]
        flat_weights = topk_weights.flatten() # [B * top_k]

        # Repeat inputs to match the flattened indices
        # Each input sample is repeated top_k times
        repeated_inputs = inputs.repeat_interleave(self.top_k, dim=0) # [B * top_k, C_in, H, W]

        # 3. Process inputs with their assigned experts in batches
        output = torch.zeros(B, C_out, H, W, device=inputs.device)
        
        # Loop over experts (a small, fixed number), NOT the batch
        for i, expert in enumerate(experts):
            # Find which inputs are routed to this expert
            mask = (flat_indices == i)
            if mask.any():
                # Get the original batch positions for scattering results later
                # We use torch.where to get the indices into the flattened tensor,
                # then integer divide by top_k to get the original batch index.
                original_batch_pos = torch.where(mask)[0] // self.top_k

                # Gather the inputs and weights for this expert's batch
                expert_inputs = repeated_inputs[mask]
                expert_weights = flat_weights[mask]
                
                # Run the expert on its batch of inputs
                expert_output = expert(expert_inputs)

                # Weight the outputs
                weighted_output = expert_output * expert_weights.view(-1, 1, 1, 1)

                # 4. Scatter (add) the results back to the correct positions
                output.index_add_(0, original_batch_pos, weighted_output)
        
        return output

    def forward(self, x, y):
        fusion_input = torch.cat([x, y], dim=1)

        # Process each modality group using the vectorized helper
        rgb_output = self._forward_moe(x, self.rgb_experts, self.rgb_gate)
        hsi_output = self._forward_moe(y, self.hsi_experts, self.hsi_gate)
        shared_output = self._forward_moe(fusion_input, self.shared_experts, self.shared_gate)

        # Combine the results from all expert groups
        fused = rgb_output + hsi_output + shared_output
        
        return fused
    

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):  # x: (B*N, C)
        return self.net(x)


# class ModalitySpecificMoE_ViT(nn.Module):
#     def __init__(self, 
#                  in_channels,         # input dim (C)
#                  hidden_dim=256,     # hidden dim in expert
#                  num_experts=2,  # number of experts per modality
#                  top_k=2,        # top-k experts used
#                  shared_expert=False,
#                  dropout=0.0):
#         super().__init__()

#         self.num_experts = num_experts
#         self.top_k = top_k
#         self.shared_expert = shared_expert

#         # Experts per modality
#         self.rgb_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)])
#         self.hsi_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)])
#         self.shared_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)]) if shared_expert else None

#         # Gating networks (shared across modalities)
#         self.rgb_gate = nn.Linear(in_channels, num_experts)
#         self.hsi_gate = nn.Linear(in_channels, num_experts)
#         self.shared_gate = nn.Linear(in_channels, num_experts) if shared_expert else None

#         self.dropout = nn.Dropout(dropout)

#     def route(self, x, gate_layer, experts):
#         # x: (B, C, N)
#         B, C, N = x.shape
#         x = x.permute(0, 2, 1).contiguous().view(B*N, C)  # (B*N, C)
#         scores = gate_layer(x)                            # (B*N, num_experts)
#         topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)  # (B*N, top_k)
#         topk_weights = F.softmax(topk_scores, dim=-1)                       # (B*N, top_k)

#         out = torch.zeros_like(x)
#         for i in range(self.top_k):
#             idx = topk_indices[:, i]
#             weight = topk_weights[:, i].unsqueeze(-1)  # (B*N, 1)
#             for j in range(self.num_experts):
#                 mask = (idx == j)
#                 if mask.sum() == 0:
#                     continue
#                 x_j = x[mask]
#                 out_j = experts[j](x_j)
#                 out[mask] += weight[mask] * out_j

#         out = self.dropout(out)
#         out = out.view(B, N, C).permute(0, 2, 1).contiguous()  # (B, C, N)
#         return out

#     def forward(self, x_rgb, x_hsi, x_shared=None):
#         # x_rgb, x_hsi: (B, C, N), x_shared optional
#         out_rgb = self.route(x_rgb, self.rgb_gate, self.rgb_experts)
#         out_hsi = self.route(x_hsi, self.hsi_gate, self.hsi_experts)
        
#         if self.shared_expert and x_shared is not None:
#             out_shared = self.route(x_shared, self.shared_gate, self.shared_experts)
#             out = out_rgb + out_hsi + out_shared  # or torch.cat([...], dim=1)
#         else:
#             out = out_rgb + out_hsi

#         return out  # (B, C, N)


# Assume Expert class is defined elsewhere, e.g.:
class Expert(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels)
        )
    def forward(self, x):
        return self.net(x)


class ModalitySpecificMoE_ViT(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=256,
                 num_experts=2,
                 top_k=2,
                 shared_expert=False,
                 dropout=0.0):
        super().__init__()

        # Assert that top_k is not greater than num_experts
        assert top_k <= num_experts, "top_k must be less than or equal to num_experts"

        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert = shared_expert

        # Experts per modality
        self.rgb_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)])
        self.hsi_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([Expert(in_channels, hidden_dim) for _ in range(num_experts)]) if shared_expert else None

        # Gating networks
        self.rgb_gate = nn.Linear(in_channels, num_experts)
        self.hsi_gate = nn.Linear(in_channels, num_experts)
        self.shared_gate = nn.Linear(in_channels, num_experts) if shared_expert else None

        self.dropout = nn.Dropout(dropout)

    def route(self, x, gate_layer, experts):
        """
        A fully vectorized routing function for speed.
        """
        # x: (B, C, N) -> (Batch, Channels, SequenceLength)
        B, C, N = x.shape
        
        # 1. Prepare tokens and get routing scores
        tokens = x.permute(0, 2, 1).reshape(B * N, C)  # (B*N, C)
        scores = gate_layer(tokens)                    # (B*N, num_experts)
        
        # Get top-k expert indices and weights for each token
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1) # (B*N, top_k)
        topk_weights = F.softmax(topk_weights, dim=-1)                      # (B*N, top_k)

        # 2. Flatten assignments for batch processing
        # Each token is routed top_k times, so we create a flat list of all assignments
        flat_indices = topk_indices.flatten()             # (B*N*top_k)
        flat_weights = topk_weights.flatten()             # (B*N*top_k)
        
        # Repeat tokens to match the flattened assignments
        repeated_tokens = tokens.repeat_interleave(self.top_k, dim=0) # (B*N*top_k, C)
        
        # 3. Vectorized Scatter-Gather
        final_output = torch.zeros_like(tokens) # (B*N, C)
        
        # Loop over the (small) number of experts, not the data
        for i in range(self.num_experts):
            # Find all assignments for the current expert
            mask = (flat_indices == i)
            
            if mask.any():
                # Gather the tokens and weights for this expert
                expert_inputs = repeated_tokens[mask]
                expert_weights = flat_weights[mask]
                
                # Run the expert on its batch of tokens
                expert_outputs = experts[i](expert_inputs)
                
                # Apply weights
                weighted_outputs = expert_outputs * expert_weights.unsqueeze(-1)
                
                # Scatter (add) the results back to the correct token positions
                # We find the original token position by integer dividing the mask index by top_k
                original_token_pos = torch.where(mask)[0] // self.top_k
                final_output.index_add_(0, original_token_pos, weighted_outputs)

        # 4. Apply dropout and reshape back to original dimensions
        final_output = self.dropout(final_output)
        out = final_output.view(B, N, C).permute(0, 2, 1) # (B, C, N)
        return out

    def forward(self, x_rgb, x_hsi, x_shared=None):
        # This part remains the same as it was already well-structured
        out_rgb = self.route(x_rgb, self.rgb_gate, self.rgb_experts)
        out_hsi = self.route(x_hsi, self.hsi_gate, self.hsi_experts)

        if self.shared_expert and x_shared is not None:
            out_shared = self.route(x_shared, self.shared_gate, self.shared_experts)
            out = out_rgb + out_hsi + out_shared
        else:
            out = out_rgb + out_hsi
            
        return out
    

if __name__ == "__main__":
	# a = torch.rand(2, 5, 3, 3)

	# aug1 = DropSpectral(dropout_rate=0.5, mode="bernoulli")
	# out1 = aug1(a)
	# print(out1, out1.shape)

	# aug2 = DropBlock2D(drop_prob=0.5, block_size=1)
	# out2 = aug2(a)
	# print(out2, out2.shape)
    

    xe4 = torch.rand(2, 512, 3, 3)
    ye4 = torch.rand(2, 512, 3, 3)

    moefusion = CrossAttentionFusion(dim=512, num_heads=8, dropout=0.1)
    center = moefusion(xe4, ye4)
    print(center.shape)

    moefusion = BiModalCrossAttentionFusion(dim=512)
    center = moefusion(xe4, ye4)
    print(center.shape)

    moefusion = Gated_Fusion(in_channels=512)
    center = moefusion(xe4, ye4)
    print(center.shape)

    moefusion = ModalitySpecificMoE(in_channels=512, hidden_dim=256)
    center = moefusion(xe4, ye4)
    print(center.shape)

    moefusion = ModalityAwareMoE(in_channels=512, hidden_dim=256)
    center = moefusion(xe4, ye4)
    print(center.shape)

    xe4 = torch.rand(2, 37, 384)
    ye4 = torch.rand(2, 37, 384)

    moefusion = ModalitySpecificMoE_ViT(in_channels=37, hidden_dim=256)
    center = moefusion(xe4, ye4)
    print("moefusion", center.shape)

