import torch
from torch import nn
from torchvision import models
from functools import partial
import torch.nn.functional as F

from vision_transformer_dino import vit_small, vit_base, vit_large, vit_giant2



class DINO(nn.Module):
	def __init__(self, channell=3, channel2=1, is_pretrained="ResNet18_Weights.DEFAULT"):
		super(DINO, self).__init__()
		self.channell = channell
		self.channel2 = channel2
		self.is_pretrained = is_pretrained

		# Load the pre-trained DINO model
		self.dino_model = vit_small(pretrained=True)

	def forward(self, x, y):
		# Process the first modality (e.g., RGB)
		x_features = self.dino_model(x)

		# Process the second modality (e.g., LiDAR)
		y_features = self.dino_model(y)

		return x_features, y_features



if __name__=="__main__":
    # model=SEBlock(128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    x = torch.randn(4, 144, 28, 28, device=device)
    y = torch.randn(4, 3, 28, 28, device=device)

    model = DINO(channell=144, channel2=3).to(device)
    # print(model)
    output, x_put, y_put = model(x, y)
    print("output", output.shape, x_put.shape, y_put.shape)
