import torch
import numpy as np

def mixup(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x1.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x1.size()[0])
    mixed_x1 = lam * x1 + (1 - lam) * x1[rand_index, :]
    mixed_x2 = lam * x2 + (1 - lam) * x2[rand_index, :]
    mixed_x3 = lam * x3 + (1 - lam) * x3[rand_index, :]
    mixed_x4 = lam * x4 + (1 - lam) * x4[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x1, mixed_x2, mixed_x3, mixed_x4, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(imgs: torch.Tensor, real_labels: torch.Tensor, device_id, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(imgs.size()[0]).to(device_id)
    target_a = real_labels
    target_b = real_labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    return imgs, target_a, target_b, lam