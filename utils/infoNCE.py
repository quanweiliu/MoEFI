import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List


class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        pass

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, tau=0.3):
        z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)

        refl_sim = torch.exp(self.sim(z1, z1) / tau)
        between_sim = torch.exp(self.sim(z1, z2) / tau)
        denominator = refl_sim.sum(dim=1) + between_sim.sum(dim=1) - torch.diag(refl_sim)
        return (-torch.log(torch.diag(between_sim) / denominator)).mean()

    def forward(self, input_1: torch.Tensor, input_2: torch.Tensor):
        """
        Forward pass for the InfoNCE loss.
        Args:
            input_1 (torch.Tensor): First input tensor.
            input_2 (torch.Tensor): Second input tensor.

        Returns:
            torch.Tensor: Computed InfoNCE loss.

        """
        loss1 = self.semi_loss(input_1, input_2)
        loss2 = self.semi_loss(input_2, input_1)
        return 0.5 * (loss1 + loss2)


class NT_xent_loss_W_EN(nn.Module):
    def __init__(self):
        super(NT_xent_loss_W_EN, self).__init__()
        pass


    def forward(self,
                z_view1: torch.Tensor,
                z_view2: torch.Tensor,
                z_negative: Optional[List[torch.Tensor]] = None,
                temperature: float=0.3):

        """
        NT-Xent loss with optional external negative samples (e.g., other modalities)

        Args:
            z_view1: Tensor [B, D] - first augmented view
            z_view2: Tensor [B, D] - second augmented view
            z_negative: Tensor [M, D] (optional) - additional negatives from other modalities
            temperature: float - temperature scaling

        Returns:
            Scalar contrastive loss

        """
        B = z_view1.size(0)
        # Normalize
        z_view1 = F.normalize(z_view1, dim=1)
        z_view2 = F.normalize(z_view2, dim=1)
        z_pos = torch.cat([z_view1, z_view2], dim=0)  # [2B, D]

        if z_negative is not None:
            # 适用于 z_negative 是一个单一张量的情况，简单高效。
            # z_negative = F.normalize(z_negative, dim=1)  # [M, D]
            # 适用于 z_negative 是一个包含多个张量的列表，适合多模态或复杂负样本场景。
            z_negative = torch.cat([F.normalize(z, dim=1) for z in z_negative], dim=0)  # [M_total, D]
            z_all = torch.cat([z_pos, z_negative], dim=0)  # [2B + M_total, D]
        else:
            z_all = z_pos  # fallback: standard SimCLR

        # Compute similarity matrix: [2B, 2B + M]
        sim_matrix = torch.matmul(z_pos, z_all.T) / temperature  # [2B, 2B+M]

        # Remove self-similarity only in the first 2B x 2B block
        if z_negative is None:
            mask = torch.eye(2 * B, dtype=torch.bool).to(z_view1.device)
            sim_matrix.masked_fill_(mask, float('-inf'))

        # Construct labels: positives are i ↔ i+B
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z_view1.device)  # [2B]
        # Cross-entropy: only over first 2B rows (anchors)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    batchSize = 8
    feature_dim = 128
    feature_1, out_1 = torch.rand(batchSize,feature_dim), torch.rand(batchSize,feature_dim)
    feature_2, out_2 = torch.rand(batchSize,feature_dim), torch.rand(batchSize,feature_dim)

    criterion = InfoNCE()

    loss = criterion(out_1, out_2)
    print(loss)