import torch
import torch.nn as nn
import torch.nn.functional as F


#  Kullback-Leibler 散度  /  通常用于 知识蒸馏（Knowledge Distillation, KD） 场景
# 这里的 KL 散度和下面的是等价的
def KL_loss(f_s, f_t, T=2, reduction='sum'):
    p_s = F.log_softmax(f_s / T, dim=1)
    p_t = F.softmax(f_t / T, dim=1)
    loss = F.kl_div(p_s, p_t, reduction=reduction) / f_t.shape[0]* (T**2)
    return loss


def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7, reduction='batchmean'):
    """
    混合蒸馏损失，包括 KL 散度损失（soft labels）和 CE 损失（hard labels）

    参数：
    - student_logits: 学生模型输出（未归一化 logits）
    - teacher_logits: 教师模型输出（未归一化 logits）
    - labels: 真值标签（整数形式）
    - T: 蒸馏温度（越高 softmax 越平滑）
    - alpha: 控制 KL 与 CE 的权重，alpha 越大越注重 teacher 的 soft label
    - reduction: KL loss 的 reduction 方式，推荐 'batchmean' 或 'mean'
    """

    # KL 散度（teacher 提供 soft label）
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction) * (T ** 2)

    return kl

    # 交叉熵（真实标签）
    ce = F.cross_entropy(student_logits, labels)

    # 混合损失
    return alpha * kl + (1 - alpha) * ce

if __name__ == "__main__":

    f_s = torch.randn(3, 5)
    f_t = torch.randn(3, 5)

    loss = KL_loss(f_s, f_t)
    print("KL Loss:", loss.item())

    loss = distillation_loss(f_s, f_t, torch.tensor([1, 2, 3]))
    print("Distillation Loss:", loss.item())

    