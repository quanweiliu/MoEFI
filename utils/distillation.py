import torch
import torch.nn as nn
import torch.nn.functional as F


f_s = torch.randn(3, 5)
f_t = torch.randn(3, 5)


#  Kullback-Leibler 散度  /  通常用于 知识蒸馏（Knowledge Distillation, KD） 场景
# 这里的 f_s 已经应用过一次 softmax
def KL_loss(f_s, f_t, T=2, reduction='sum'):
    p_s = F.log_softmax(f_s / T, dim=1)
    p_t = F.softmax(f_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction=reduction) / f_t.shape[0]* (T**2)
    return loss


# print(KL_loss(f_s, f_t))

# def KL_loss(f_s, f_t, T=2):
#     p_s = F.log_softmax(f_s / T, dim=1)
#     p_t = F.softmax(f_t / T, dim=1)
#     loss = torch.sum(p_t * (p_t.log() - p_s)) / p_s.size()[0] * (T**2)
#     return loss


# print(KL_loss(f_s, f_t))