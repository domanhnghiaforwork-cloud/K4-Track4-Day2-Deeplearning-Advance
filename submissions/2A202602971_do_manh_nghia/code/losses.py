"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Mô-đun quản lý hàm mục tiêu và xử lý mất cân bằng lớp cho bài lab:
- Cross-Entropy tiêu chuẩn
- Label Smoothing Cross-Entropy (chống over-confidence)
- Focal Loss nhiều lớp (tập trung vào mẫu khó và lớp hiếm)
- Trọng số lớp (nghịch đảo tần suất hoặc Class-Balanced số mẫu hiệu dụng)
- Trộn mẫu theo batch (Mixup và CutMix với tính toán diện tích thực)
"""
from __future__ import annotations

from typing import Tuple, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw) -> nn.Module:
    """Trả về hàm loss theo `kind`: 'ce', 'ls', 'focal', 'ce_weighted'.

    Ví dụ các tham số trong kw:
      - smoothing (float): Độ làm mịn nhãn cho label smoothing (mặc định 0.1)
      - gamma (float): Số mũ trừng phạt mẫu khó cho Focal loss (mặc định 2.0)
      - alpha / weight (Tensor): Vector trọng số theo 9 lớp
    """
    kind = kind.lower()
    weight = kw.get("weight", kw.get("alpha", None))

    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    elif kind in ("ls", "label_smoothing"):
        eps = kw.get("smoothing", kw.get("label_smoothing", 0.1))
        return LabelSmoothingCE(smoothing=eps, weight=weight)
    elif kind == "focal":
        gamma = kw.get("gamma", kw.get("focal_gamma", 2.0))
        return FocalLoss(gamma=gamma, alpha=weight)
    elif kind in ("ce_weighted", "weighted"):
        if weight is None:
            raise ValueError("Cần cung cấp tham số 'weight' cho hàm loss có trọng số 'ce_weighted'!")
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Hàm loss loại '{kind}' không được hỗ trợ!")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Khi eps = 0.0, hàm loss hoàn toàn tương đương với CrossEntropyLoss chuẩn.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = float(smoothing)
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Nếu nhãn là one-hot hoặc soft-target (2D tensor)
        if target.ndim > 1:
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target * log_probs).sum(dim=-1).mean()
            return loss

        # Sử dụng hàm tích hợp sẵn của PyTorch đã được tối ưu hóa
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Khi gamma = 0.0 và alpha = None, hàm này đồng nhất với CrossEntropyLoss thông thường.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, eps: float = 1e-7):
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        # 1. Tính log-probabilities và probabilities
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()

        # 2. Lấy p_t và log(p_t) của lớp tương ứng với nhãn đúng
        if target.ndim == 1:
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
        else:
            target_one_hot = target.float()

        p_t = (p * target_one_hot).sum(dim=-1).clamp(min=self.eps, max=1.0)
        log_p_t = (log_p * target_one_hot).sum(dim=-1)

        # 3. Tính hệ số điều biến (focal weight)
        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        # 4. Áp dụng trọng số lớp alpha_t nếu có
        if self.alpha is not None:
            alpha_t = (self.alpha * target_one_hot).sum(dim=-1)
            loss = -alpha_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        return loss.mean()


def class_weights(counts: Sequence[int] | Dict[int, int] | np.ndarray, beta: float = 0.0) -> torch.Tensor:
    """Tính trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (Quy tắc N2: chỉ dùng train).

    - beta == 0.0: Trọng số tỷ lệ nghịch với tần suất lớp: w_c = 1 / n_c, chuẩn hóa trung bình về 1.0.
    - beta > 0.0: Class-Balanced Loss dựa trên "số mẫu hiệu dụng" (Cui et al., CVPR 2019):
        E_n = (1 - beta^n_c) / (1 - beta) => w_c = 1 / E_n.
        Chuẩn hóa sao cho trung bình trọng số bằng 1.0.
    """
    if isinstance(counts, dict):
        counts_arr = np.array([counts[i] for i in range(len(counts))], dtype=np.float64)
    else:
        counts_arr = np.array(counts, dtype=np.float64)

    assert len(counts_arr) == 9, f"Kỳ vọng 9 lớp, nhưng nhận được {len(counts_arr)}"
    assert np.all(counts_arr > 0), "Số lượng mẫu của các lớp phải lớn hơn 0"

    if beta <= 0.0:
        # Nghịch đảo số lượng ảnh: 1 / n_c
        weights = 1.0 / counts_arr
    else:
        # Công thức số mẫu hiệu dụng (Class-Balanced)
        effective_num = (1.0 - np.power(beta, counts_arr)) / (1.0 - beta)
        weights = 1.0 / effective_num

    # Chuẩn hóa về trọng số trung bình bằng 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Trộn một batch ảnh và nhãn (Trục B của GUIDE.md).

    - mode='mixup': Pha trộn tuyến tính: x_mix = lam * x + (1 - lam) * x[perm]
    - mode='cutmix': Cắt một vùng hình chữ nhật từ x[perm] dán đè lên x, và tính lại
      lambda thực tế dựa trên tỷ lệ diện tích vùng cắt (slide trang 48).

    Trả về: (x_mixed, (y_a, y_b, lam))
    """
    batch_size = x.size(0)
    device = x.device

    if alpha > 0.0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    # Sinh hoán vị ngẫu nhiên trong batch
    perm = torch.randperm(batch_size, device=device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mixed = lam * x + (1.0 - lam) * x[perm]
        return x_mixed, (y_a, y_b, lam)

    elif mode == "cutmix":
        _, _, h, w = x.shape

        # Tỷ lệ kích thước vùng cắt r_w, r_h dựa trên sqrt(1 - lam)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Chọn tâm vùng cắt ngẫu nhiên
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)

        # Tọa độ hộp cắt
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        x_mixed = x.clone()
        x_mixed[:, :, bby1:bby2, bbx1:bbx2] = x[perm, :, bby1:bby2, bbx1:bbx2]

        # Điều chỉnh lại lam theo đúng DIỆN TÍCH THỰC của hộp sau khi bị cắt ngoài biên
        actual_area = (bbx2 - bbx1) * (bby2 - bby1)
        lam_adjusted = 1.0 - (actual_area / float(w * h))

        return x_mixed, (y_a, y_b, lam_adjusted)

    else:
        raise ValueError(f"Chế độ trộn '{mode}' không được hỗ trợ (chỉ chọn 'mixup' hoặc 'cutmix')!")


def mixed_loss(criterion: nn.Module, logits: torch.Tensor, targets: Tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Tính toán loss cho batch đã được trộn:

    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
