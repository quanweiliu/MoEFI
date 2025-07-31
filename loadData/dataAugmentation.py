import random
import torch

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from PIL import ImageFilter, ImageOps


class SpectralDropout(object):

    """
    Args:
        x: Tensor of shape [C, H, W]
        dropout_rate: Ratio of channels (spectral bands) to drop
        mode: "bernoulli" or "fixed"
        bernoulli 模式：每个波段以一定概率独立被丢弃, 有可能某些通道保留，有些通道被丢弃，甚至全部被保留或全部被丢弃。
        fixed 模式：固定丢弃 int(C * dropout_rate) 个通道。
    Returns:
        Tensor of shape [C, H, W]

    """

    def __init__(self, p=0.5, dropout_rate=0.1, mode="bernoulli"):
        self.prob = p
        self.dropout_rate = dropout_rate
        self.mode = mode

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        C, H, W = img.shape
        if self.mode == "bernoulli":
            drop_mask = torch.rand(C, 1, 1, device=img.device) > self.dropout_rate
        elif self.mode == "fixed":  # fixed mode
            drop_mask = torch.ones(C, device=img.device)
            num_drop = int(C * self.dropout_rate)
            drop_indices = torch.randperm(C)[:num_drop]
            drop_mask[drop_indices] = 0
            drop_mask = drop_mask.view(C, 1, 1)
        else:
            raise ValueError("Invalid mode. Use 'bernoulli' or 'fixed'.")
        return img * drop_mask


class Spectral_mixing(object):
    '''
    在高光谱图像的某一个随机波段（band）上，将 x1 和 x2 的该波段做加权混合。
    模拟光谱信号的不确定性，比如大气干扰或传感器噪声。
    '''
    def __init__(self, p=0.5, mix_band=3):
        self.prob = p
        self.mix_band = mix_band

    def __call__(self, img1, img2):
        img1 = img1.clone()
        img2 = img2.clone()

        do_it = random.random() <= self.prob
        if not do_it:
            return img1
        C, H, W = img1.shape
        assert self.mix_band < C, "mix_band must be less than or equal to the number of spectral bands C"

        alpha = torch.rand(1).item()
        # band = torch.randint(0, C, (1,)).item()
        # img1[band] = alpha * img1[band] + (1 - alpha) * img2[band]

        bands = torch.randperm(C)[:self.mix_band]
        img1[bands] = alpha * img1[bands] + (1 - alpha) * img2[bands]

        return img1

# if __name__ == "__main__":
#     # Example usage
#     img1 = torch.randn(3, 64, 64)  # Simulated image with 3 spectral bands
#     img2 = torch.randn(3, 64, 64)  # Another image with the same number of bands

#     spectral_mixing = Spectral_mixing(p=0.5, mix_band=2)
#     mixed_img = spectral_mixing(img1, img2)
#     print("Mixed Image Shape:", mixed_img.shape)


class Spectral_cutmix(object):
    """
    选择一个随机波段 band，用 x2 的这个波段替换掉 x1 中对应的波段。
    把每个通道看作一层彩色塑料薄片叠在一起构成图像。
    spectral_cutmix 就是从图像 x1 中抽掉第 band 层塑料片，换上 x2 中的那一层。
    """

    def __init__(self, p=0.5, mix_band=3):
        self.prob = p
        self.mix_band = mix_band

    def __call__(self, img1, img2):
        do_it = random.random() <= self.prob
        if not do_it:
            return img1
        C, H, W = img1.shape
        assert self.mix_band < C, "mix_band must be less than or equal to the number of spectral bands C"
        band = torch.randint(0, C-self.mix_band, (1,)).item()
        return torch.cat([img1[:band], img2[band:band+self.mix_band], img1[band+self.mix_band:]], dim=0)

           
class AddNoise(object):
    def __init__(self, p=0.5, noise_std=0.01):
        self.prob = p
        self.noise_std = noise_std

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img + torch.randn_like(img) * self.noise_std


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class SpectralShift(object):
    """
    高光谱图像的光谱维度（通道维度）进行“频谱平移”
    光谱维度上的“滑动窗口变换”，但是“环形滑动”。
    """
    def __init__(self, p=0.5):

        self.prob = p

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        shift = torch.randint(-2, 3, (1,)).item()
        return torch.roll(img, shifts=shift, dims=0)


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


# class DataAugmentationDINO(object):
#     def __init__(self, randCrop=28, local_crops_number=0):
#         # first global crop
#         self.global_transfo1 = transforms.Compose([
#                 transforms.RandomResizedCrop(randCrop, antialias=True, interpolation=InterpolationMode.BICUBIC),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 # transforms.RandomRotation(90),
#                 transforms.GaussianBlur((3)),
#                 # transforms.RandomErasing(0.5)
#                 # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ])
#         # second global crop
#         self.global_transfo2 = transforms.Compose([
#                 transforms.RandomResizedCrop(randCrop, antialias=True, interpolation=InterpolationMode.BICUBIC),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 # transforms.RandomRotation(90)
#                 transforms.GaussianBlur((3)),
#                 # transforms.RandomErasing(0.5)
#                 # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),

#                 ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#                 # transforms.RandomResizedCrop(24, antialias=True, interpolation=InterpolationMode.BICUBIC),
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomVerticalFlip(p=0.5),
#                 # GaussianBlur(1.0),
#                 # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                 ])

#     def __call__(self, image):
#         crops = []
#         crops.append(self.global_transfo1(image))
#         crops.append(self.global_transfo2(image))
#         # for _ in range(self.local_crops_number):
#         #     crops.append(self.local_transfo(image))
#         return crops
    
class DataAugmentationDINO2(object):
    def __init__(self, args):
        # first global crop
        self.global_transfo = transforms.Compose([
                # transforms.RandomResizedCrop(randCrop, antialias=True, interpolation=InterpolationMode.BICUBIC),
                # transforms.RandomCrop(args.randomCrop, padding=1, padding_mode='reflect'),
                transforms.RandomCrop(args.randomCrop),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                AddNoise(),
                SpectralDropout(),
                SpectralShift(),
                # transforms.RandomRotation(90),
                # transforms.GaussianBlur((3)),
                # transforms.RandomErasing(0.5)
                ])

    def __call__(self, image, x_pair=None):

        if x_pair is not None:
            image = Spectral_mixing()(image, x_pair)
            image = Spectral_cutmix()(image, x_pair)

        image = self.global_transfo(image)
        # print(image.shape)
        return image