import torch
import torch.nn as nn
import torch.nn.functional as F


class Mutual_info(nn.Module):
    def __init__(self, input_channels, channels, latent_size=16, margin=4.5):
        """
        Mutual_info 用于比较两个特征的相关性。
        - mode='similar': 希望两个特征越相似越好（loss 趋近 0）
        - mode='dissimilar': 希望两个特征越不同越好（loss 趋近 0）

        margin: dissimilar 最小距离阈值，低于此值才会惩罚（越不相似越好）
        """
        super(Mutual_info, self).__init__()

        self.shared_conv = nn.Conv2d(input_channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.flatten = lambda x: x.reshape(x.size(0), -1)
        self.fc = nn.Linear(channels * 4 * 4, latent_size)
        self.norm = nn.LayerNorm(latent_size)

        self.mse = nn.MSELoss()
        self.margin = margin

    def extract_latent(self, feat):
        """
        提取特征的 latent 表示
        """
        x = self.relu(self.shared_conv(feat))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.norm(self.fc(x))
        # print(x.shape)
        return x

    def forward(self, feat1, feat2, mode='similar'):
        """
        计算 mutual info loss。
        参数：
            feat1, feat2: 输入特征图
            mode: 'similar' or 'dissimilar'
        返回：
            一个 scalar loss
        """
        # print("feat1", feat1.shape, "feat2", feat2.shape)  # [128, 1024, 4]  / [128, 1024, 2, 2]

        if len(feat1.size()) == 3:
            feat1 = feat1.reshape(feat2.shape)

        z1 = self.extract_latent(feat1)
        z2 = self.extract_latent(feat2)

        # # 三个相似性指标
        # p1 = F.log_softmax(z1, dim=1)
        # p2 = F.softmax(z2, dim=1)

        # kl_loss = (F.kl_div(p1, p2, reduction='batchmean') +
        #            F.kl_div(F.log_softmax(z2, dim=1), F.softmax(z1, dim=1), reduction='batchmean')) / 2
        
        # 衡量两个向量方向有多“不相似”。 越不相似，值越大。
        cosine_dist = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
        mse_dist = self.mse(z1, z2)

        similarity_score = cosine_dist + mse_dist
        # similarity_score = kl_loss + cosine_dist + mse_dist

        if mode == 'similar':
            # 希望越相似越好 → 直接最小化相似性距离
            loss = similarity_score
        elif mode == 'dissimilar':
            # 希望越 dissimilar 越好 → 让 similarity_score 尽可能大 → 使用 margin 保护
            loss = F.relu(self.margin - similarity_score)
            # loss = - similarity_score
        else:
            raise ValueError(f"mode must be 'similar' or 'dissimilar', got: {mode}")
        # print("cosine_dist, mse_dist, similarity_score", cosine_dist.item(), mse_dist.item(), similarity_score.item(), loss.item())

        return loss
    

class Mutual_info_cnn(nn.Module):
    def __init__(self, input_channels, output_channels, latent_size=16, margin=4.5):
        """
        Mutual_info 用于比较两个特征的相关性。
        - mode='similar': 希望两个特征越相似越好（loss 趋近 0）
        - mode='dissimilar': 希望两个特征越不同越好（loss 趋近 0）

        margin: dissimilar 最小距离阈值，低于此值才会惩罚（越不相似越好）
        """
        super(Mutual_info_cnn, self).__init__()

        self.shared_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.flatten = lambda x: x.reshape(x.size(0), -1)
        # self.fc = nn.Linear(output_channels * 2 * 2, latent_size)  # output_channels * h * w
        self.fc = nn.Linear(output_channels * 4 * 4, latent_size)
        self.norm = nn.LayerNorm(latent_size)

        self.mse = nn.MSELoss()
        self.margin = margin

    def extract_latent(self, feat):
        """
        提取特征的 latent 表示
        """
        x = self.relu(self.shared_conv(feat))
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)     # 128, 4096
        x = self.norm(self.fc(x))
        # print(x.shape)
        return x

    def forward(self, feat1, feat2, mode='similar'):
        """
        计算 mutual info loss。
        参数：
            feat1, feat2: 输入特征图
            mode: 'similar' or 'dissimilar'
        返回：
            一个 scalar loss
        """
        # print("feat1", feat1.shape, "feat2", feat2.shape)  # [128, 1024, 4]  / [128, 1024, 2, 2]

        if len(feat1.size()) == 3:
            feat1 = feat1.reshape(feat2.shape)

        z1 = self.extract_latent(feat1)
        z2 = self.extract_latent(feat2)

        # # 三个相似性指标
        # p1 = F.log_softmax(z1, dim=1)
        # p2 = F.softmax(z2, dim=1)

        # kl_loss = (F.kl_div(p1, p2, reduction='batchmean') +
        #            F.kl_div(F.log_softmax(z2, dim=1), F.softmax(z1, dim=1), reduction='batchmean')) / 2
        
        # 衡量两个向量方向有多“不相似”。 越不相似，值越大。
        cosine_dist = 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()
        mse_dist = self.mse(z1, z2)

        similarity_score = cosine_dist + mse_dist
        # similarity_score = kl_loss + cosine_dist + mse_dist

        if mode == 'similar':
            # 希望越相似越好 → 直接最小化相似性距离
            loss = similarity_score
        elif mode == 'dissimilar':
            # 希望越 dissimilar 越好 → 让 similarity_score 尽可能大 → 使用 margin 保护
            loss = F.relu(self.margin - similarity_score)
            # loss = - similarity_score
        else:
            raise ValueError(f"mode must be 'similar' or 'dissimilar', got: {mode}")
        # print("cosine_dist, mse_dist, similarity_score", cosine_dist.item(), mse_dist.item(), similarity_score.item(), loss.item())

        return loss
    

if __name__ == "__main__":

    # a = torch.randn(64, 128, 16)
    # a1 = torch.randn(64, 128, 4, 4)
    # a2 = torch.randn(64, 128, 4, 4)
    # model = Mutual_info(128, 128)

    a1 = torch.randn(64, 1024, 2, 2)
    a2 = torch.randn(64, 1024, 2, 2)
    model = Mutual_info(1024, 1024)
    c = model(a1, a2, mode='similar')

    # print(c)
    # print(c.shape)