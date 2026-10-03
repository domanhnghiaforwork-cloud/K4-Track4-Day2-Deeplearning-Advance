"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Mô-đun tối ưu hóa giai đoạn suy luận (Inference):
- Chạy trích xuất Logit và xác suất với chế độ eval / torch.inference_mode
- Test-Time Augmentation (TTA) với biến đổi lật ngang (hflip) và đa góc cắt (multi-crop)
- Gộp các lượt nhìn TTA theo không gian xác suất hoặc không gian logit
- Ghép mô hình (Ensemble) trung bình xác suất
- Hiệu chuẩn độ tin cậy bằng Temperature Scaling (tối ưu hóa ECE trên tập Val)
- Gộp BatchNorm vào Conv2d (Conv-BN Fusion) để giảm thiểu độ trễ suy luận
"""
from __future__ import annotations

import copy
import sys
from typing import List, Tuple, Callable, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.fusion as fusion


def _softmax(z: np.ndarray) -> np.ndarray:
    """Hàm Softmax ổn định về mặt số học trên mảng numpy."""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def predict_logits(model: nn.Module, loader, device: str | torch.device = "cuda",
                   view: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Chạy model trên loader và thu thập danh sách tên file, nhãn thật và ma trận logit.

    `view` là hàm biến đổi tensor batch ảnh trước khi nạp vào model (ví dụ view_hflip).
    Đảm bảo: model.eval(), torch.inference_mode(), giữ nguyên thứ tự dữ liệu của loader.
    """
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_filenames: List[str] = []
    all_targets: List[int] = []
    all_logits: List[np.ndarray] = []

    use_amp = torch.cuda.is_available() and device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device, non_blocking=True)

            # Áp dụng hàm biến đổi góc nhìn (nếu có TTA)
            if view is not None:
                images = view(images)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)

            all_filenames.extend(filenames)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_logits.append(outputs.float().cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    logits = np.concatenate(all_logits, axis=0)

    return all_filenames, y_true, logits


def view_identity(x: torch.Tensor) -> torch.Tensor:
    """Góc nhìn nguyên bản không biến đổi."""
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch ảnh (N, C, H, W) dọc theo trục chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int = 224) -> List[torch.Tensor]:
    """Cắt 5 vùng ảnh kích thước (crop x crop): 4 góc (Top-Left, Top-Right, Bottom-Left, Bottom-Right) và 1 tâm.

    Trả về danh sách 5 tensor batch ảnh.
    """
    _, _, h, w = x.shape
    if h < crop or w < crop:
        raise ValueError(f"Kích thước ảnh ({h}, {w}) nhỏ hơn kích thước crop ({crop}, {crop})")

    tl = x[:, :, :crop, :crop]
    tr = x[:, :, :crop, w - crop:]
    bl = x[:, :, h - crop:, :crop]
    br = x[:, :, h - crop:, w - crop:]

    ch_start = (h - crop) // 2
    cw_start = (w - crop) // 2
    center = x[:, :, ch_start:ch_start + crop, cw_start:cw_start + crop]

    return [tl, tr, bl, br, center]


def views_multiscale(x: torch.Tensor, sizes: List[int]) -> List[torch.Tensor]:
    """Thay đổi kích thước batch ảnh sang các độ phân giải khác nhau trong `sizes`."""
    views = []
    for s in sizes:
        resized = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
        views.append(resized)
    return views


def aggregate_views(logits_per_view: List[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán xác suất cuối cùng (slide trang 62).

    - space='prob' : Tính softmax từng view trước, sau đó lấy trung bình cộng xác suất.
    - space='logit': Lấy trung bình cộng logit của các view, sau đó mới tính softmax.
    """
    if not logits_per_view:
        raise ValueError("Danh sách logits_per_view không được để trống!")

    if space == "prob":
        probs_list = [_softmax(l) for l in logits_per_view]
        probs = np.mean(probs_list, axis=0)
    elif space == "logit":
        avg_logits = np.mean(logits_per_view, axis=0)
        probs = _softmax(avg_logits)
    else:
        raise ValueError(f"Không gian gộp '{space}' không hợp lệ (chỉ chọn 'prob' hoặc 'logit')!")

    # Đảm bảo tổng xác suất bằng đúng 1.0
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return probs


def ensemble_probs(list_of_probs: List[np.ndarray]) -> np.ndarray:
    """Ghép dự đoán (Ensemble) từ nhiều mô hình khác nhau hoặc khác seed:

    Tính trung bình cộng ma trận xác suất và chuẩn hóa tổng bằng 1.0.
    """
    if not list_of_probs:
        raise ValueError("Danh sách xác suất ghép không được rỗng!")

    avg_probs = np.mean(list_of_probs, axis=0)
    avg_probs = avg_probs / np.sum(avg_probs, axis=-1, keepdims=True)
    return avg_probs


def fit_temperature(val_logits: np.ndarray | torch.Tensor, val_labels: np.ndarray | torch.Tensor) -> float:
    """Tìm hệ số nhiệt độ tối ưu T > 0 bằng cách cực tiểu hóa Negative Log-Likelihood (NLL) trên tập VAL:

    p_i = softmax(z_i / T)  (slide trang 69).
    Quy tắc N2: Chỉ khớp T trên validation, tuyệt đối không dùng tập test để tìm T.
    """
    if isinstance(val_logits, np.ndarray):
        logits = torch.from_numpy(val_logits).float()
    else:
        logits = val_logits.float().clone()

    if isinstance(val_labels, np.ndarray):
        labels = torch.from_numpy(val_labels).long()
    else:
        labels = val_labels.long().clone()

    # Hàm mất mát NLL phụ thuộc vào tham số T
    def _nll_loss(t_val: float) -> float:
        t_val = max(1e-4, float(t_val))
        scaled_logits = logits / t_val
        loss = F.cross_entropy(scaled_logits, labels).item()
        return loss

    # Sử dụng thuật toán tối ưu hóa vô hướng giới hạn trong khoảng [0.05, 10.0]
    opt_result = scipy.optimize.minimize_scalar(_nll_loss, bounds=(0.05, 10.0), method="bounded")
    optimal_t = float(opt_result.x)

    print(f"[fit_temperature] Nhiệt độ T tối ưu trên Validation: T = {optimal_t:.4f}")
    return optimal_t


def apply_temperature(logits: np.ndarray | torch.Tensor, T: float) -> np.ndarray:
    """Áp dụng nhiệt độ T đã khớp để tính xác suất softmax đã được hiệu chuẩn:

    probs = softmax(logits / T).
    """
    T = max(1e-4, float(T))
    if isinstance(logits, torch.Tensor):
        logits = logits.cpu().numpy()

    scaled_logits = logits / T
    return _softmax(scaled_logits)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp các lớp BatchNorm2d vào Conv2d liền kề trong chế độ suy luận (slide trang 71, 75):

    w' = gamma * w / sqrt(var + eps)
    b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Giúp giảm số phép tính và tối ưu hóa bộ nhớ đệm khi chạy thực tế trên vi điều khiển / robot.
    """
    fused_model = copy.deepcopy(model).eval()

    def _fuse_recursive(module: nn.Module):
        # 1. Gộp các khối Sequential chứa Conv2d -> BatchNorm2d
        if isinstance(module, nn.Sequential):
            i = 0
            while i < len(module) - 1:
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    module[i] = fusion.fuse_conv_bn_eval(module[i], module[i + 1])
                    module[i + 1] = nn.Identity()
                    i += 2
                else:
                    i += 1

        # 2. Gộp các thuộc tính thường gặp (conv + bn) trong các block Residual
        for conv_name, bn_name in [("conv", "bn"), ("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3"), ("downsample.0", "downsample.1")]:
            if hasattr(module, conv_name) and hasattr(module, bn_name):
                c = getattr(module, conv_name)
                b = getattr(module, bn_name)
                if isinstance(c, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    setattr(module, conv_name, fusion.fuse_conv_bn_eval(c, b))
                    setattr(module, bn_name, nn.Identity())

        # Đệ quy xuống các module con
        for child in module.children():
            _fuse_recursive(child)

    _fuse_recursive(fused_model)
    return fused_model
