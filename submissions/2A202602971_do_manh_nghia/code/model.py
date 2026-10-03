"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Mô-đun quản lý kiến trúc mô hình học sâu cho bài lab:
- Tải các backbone từ thư viện timm (CNN và Vision Transformer)
- Hỗ trợ 3 cơ chế khởi tạo (scratch, frozen, finetune)
- Phân chia tham số thành 3 nhóm tối ưu hóa (Backbone 2D, Backbone Norm/Bias, Classifier Head)
- Đo số tham số (Params tính bằng triệu) và số phép tính (GMACs)
"""
from __future__ import annotations

import copy
import sys
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import timm

# Danh sách ánh xạ các backbone gợi ý trong GUIDE.md
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # Vision Transformer DeiT
    "vit_small": "vit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224", # Swin Transformer
    "efficientnet_b0": "efficientnet_b0",        # Mạng nhẹ tối ưu tài nguyên
    "mobilenetv3": "mobilenetv3_large_100",      # Mạng nhẹ cho mobile/edge
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> nn.Module:
    """Tạo model phân loại 9 lớp từ thư viện timm.

    `init` (Trục A của GUIDE.md mục 3):
      - "scratch"  : pretrained=False, khởi tạo ngẫu nhiên, huấn luyện toàn bộ
      - "frozen"   : pretrained=True, đóng băng toàn bộ backbone, chỉ huấn luyện classifier head
      - "finetune" : pretrained=True, tinh chỉnh toàn bộ mô hình (công thức chuẩn)
    """
    # Ánh xạ tên gọi tắt sang tên định danh trong timm
    model_name = SUGGESTED_BACKBONES.get(name, name)
    is_pretrained = (init != "scratch") and pretrained

    print(f"[build_model] Đang tạo mô hình '{model_name}' (pretrained={is_pretrained}, init='{init}', drop_rate={drop_rate})...")
    model = timm.create_model(
        model_name,
        pretrained=is_pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )

    # Ghi nhận thông tin tag trọng số thực tế đã tải
    tag_info = getattr(model, "pretrained_cfg", {}).get("tag", "default")
    model.pretrained_tag = tag_info

    # Nếu chọn chế độ đóng băng: đóng băng thân mô hình
    if init == "frozen":
        freeze_backbone(model)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Đóng băng mọi tham số của backbone, chỉ cho phép cập nhật classifier head.

    Lưu ý: Khi đóng băng, các lớp BatchNorm vẫn cần duy trì ở chế độ eval()
    để không cập nhật running_mean và running_var (slide trang 53 & GUIDE.md mục 3.2).
    """
    # 1. Tắt gradient toàn bộ mô hình
    for p in model.parameters():
        p.requires_grad = False

    # 2. Bật lại gradient riêng cho classifier head
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for p in classifier.parameters():
            p.requires_grad = True
    elif isinstance(classifier, nn.Parameter):
        classifier.requires_grad = True

    # 3. Chuyển các lớp BatchNorm về chế độ eval
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, weight_decay: float) -> List[Dict[str, Any]]:
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52:

    1. Thân backbone với tensor > 1 chiều (weights): lr = lr_backbone, weight_decay = weight_decay
    2. Chuẩn hóa (BatchNorm/LayerNorm) và bias của backbone (tensor <= 1 chiều): lr = lr_backbone, weight_decay = 0.0
    3. Classifier head mới: lr = lr_head (thường gấp 10 lần backbone), weight_decay = weight_decay
    """
    # Thu thập các tham số thuộc classifier head
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        head_params = set(classifier.parameters())
    elif isinstance(classifier, nn.Parameter):
        head_params = {classifier}
    else:
        head_params = set()

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param in head_params:
            # Nhóm 3: Head mới
            head_decay.append(param)
        else:
            # Nhóm của backbone
            if param.ndim <= 1 or name.endswith(".bias"):
                # Nhóm 2: Bias và Norm -> weight_decay = 0
                backbone_no_decay.append(param)
            else:
                # Nhóm 1: Trọng số 2D trở lên (Conv, Linear) -> weight_decay chuẩn
                backbone_decay.append(param)

    groups = [
        {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
        {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": head_decay, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return groups


def count_params(model: nn.Module) -> float:
    """Đếm tổng số tham số của mô hình (tính theo đơn vị Triệu - Million), bao gồm cả tham số đóng băng."""
    total = sum(p.numel() for p in model.parameters())
    return round(total / 1e6, 3)


def count_gmacs(model: nn.Module, img_size: int = 224) -> float:
    """Tính toán GMACs cho một ảnh kích thước 3 x img_size x img_size (slide tính MAC, 1 MAC ≈ 2 FLOPs).

    Ưu tiên sử dụng thư viện thop nếu có sẵn, hoặc dùng bộ đếm hook fallback tích hợp.
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # Thử dùng thư viện thop nếu đã cài đặt
    try:
        import thop
        # Tạo bản sao mô hình ở eval mode để đo
        eval_model = copy.deepcopy(model).to(device)
        eval_model.eval()
        macs, _ = thop.profile(eval_model, inputs=(dummy_input,), verbose=False)
        gmacs = macs / 1e9
        return round(float(gmacs), 3)
    except Exception:
        pass

    # Thuật toán đếm thủ công fallback qua PyTorch hooks
    total_macs = [0]
    hooks = []

    def conv_hook(self, input, output):
        batch_size = input[0].size(0)
        output_channels, output_h, output_w = output.shape[1], output.shape[2], output.shape[3]
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
        macs = batch_size * output_channels * output_h * output_w * kernel_ops
        total_macs[0] += macs

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].ndim > 1 else 1
        total_macs[0] += batch_size * self.in_features * self.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception:
            pass
        finally:
            for h in hooks:
                h.remove()
            if was_training:
                model.train()

    gmacs = total_macs[0] / 1e9
    return round(float(gmacs), 3)
