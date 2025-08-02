import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalLoss(nn.Module):
    def __init__(self, version='frobenius', sample_size=None):
        super().__init__()
        assert version in ['frobenius', 'dot']
        self.version = version
        self.sample_size = sample_size

    def forward(self, a1, a2):
        if a1.ndim == 4:

            # 为什么可以将 B×H×W 统一展平为一个维度？	
            # 因为在 feature-level 对齐中，“每个位置”都是语义样本
            # 我们要比较的是模态 A 和 B 在相同位置提取出的表示是否相似、正交、互补
            a1 = a1.permute(0, 2, 3, 1).reshape(-1, a1.shape[1])    # [B×H×W, C]
            a2 = a2.permute(0, 2, 3, 1).reshape(-1, a2.shape[1])    # 每个“像素点”是一个样本, 每行是一个通道向量,用于统计分布、正交性、互信息分析最合理
            # print("shape", a1.shape, a2.shape)
            
        if self.sample_size is not None and a1.size(0) > self.sample_size:
            idx = torch.randperm(a1.size(0))[:self.sample_size]
            a1, a2 = a1[idx], a2[idx]

        if self.version == 'frobenius':      # 协方差 Frobenius 范数版本
            a1 = a1 - a1.mean(dim=0, keepdim=True)
            a2 = a2 - a2.mean(dim=0, keepdim=True)
            cov = torch.matmul(a1.T, a2) / a1.size(0)
            return torch.norm(cov, p='fro') ** 2
        elif self.version == 'dot':          # 点积归一化版本
            a1 = F.normalize(a1, dim=1)
            a2 = F.normalize(a2, dim=1)
            dot = torch.sum(a1 * a2, dim=1)
            return torch.mean(dot ** 2)
        
        else:
            raise NotImplementedError("Not Implemented OrthogonalLoss")

# Example usage:
if __name__ == "__main__":

	# Simulate two feature maps with shape [batch_size, channels, height, width]
	v1 = torch.randn(8, 64, 32, 32)  # Example feature map 1
	v2 = torch.randn(8, 64, 32, 32)


	orth_loss = OrthogonalLoss(version="frobenius", sample_size=2000)
	o_loss = orth_loss(v1, v2)
	print(o_loss)