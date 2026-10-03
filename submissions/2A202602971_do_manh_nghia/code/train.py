"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Mô-đun điều phối huấn luyện trung tâm của toàn bộ bài lab (RUBRIC mục H):
- Thiết kế một hàm duy nhất `run(cfg: Config)` dùng chung cho mọi kịch bản thí nghiệm.
- Quản lý siêu tham số qua Dataclass `Config` và hỗ trợ ghi đè từ dòng lệnh (`--set key=value`).
- Hỗ trợ huấn luyện độ chính xác hỗn hợp (Mixed Precision - AMP).
- Lịch trình tốc độ học: Tuyến tính Warmup kết hợp Cosine Annealing.
- Theo dõi Exponential Moving Average (EMA) cho trọng số mô hình.
- Tự động lưu Checkpoint tốt nhất dựa trên chỉ số **Macro-F1 trên tập Validation**.
- Xuất biểu đồ đường cong huấn luyện (`curves/<exp_id>.png`) và file dự đoán chuẩn (`predictions/*.csv`).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, Any, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Tìm và import eval.py từ thư mục gốc của repository
for search_dir in [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parents[3]]:
    if (search_dir / "eval.py").exists():
        if str(search_dir) not in sys.path:
            sys.path.insert(0, str(search_dir))
        break

try:
    from eval import save_predictions, compute_metrics
except ImportError:
    # Fallback dự phòng nếu import trực tiếp
    import eval as ev
    save_predictions = ev.save_predictions
    compute_metrics = ev.compute_metrics

# Import các mô-đun thành phần nội bộ
from dataset import load_split, check_split, build_transforms, make_loader
from model import build_model, freeze_backbone, param_groups, count_params, count_gmacs
from losses import build_criterion, class_weights, mix_batch, mixed_loss


@dataclass
class Config:
    # --- Định danh thí nghiệm ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- Kiến trúc mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- Dữ liệu / Augmentation ---
    img_size: int = 224
    aug: str = "basic"                # none | basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- Hàm Loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- Tối ưu hóa (Công thức nền GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- Đường dẫn dữ liệu và kết quả ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # Lưu config.json, history.csv, checkpoint, logit
    pred_dir: str = "predictions"     # Lưu file dự đoán CSV nộp bài
    # --- Chỉ bật ở Bước 4 (Chung kết): Ghi file dự đoán trên tập TEST. Mặc định TẮT (Quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục lưu trữ kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Tạo optimizer AdamW với 3 nhóm tham số (Backbone weights, Backbone norm/bias, Classifier head)."""
    groups = param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
    """Xây dựng lịch trình học: Tuyến tính Warmup sau đó Cosine Annealing về gần 0 (slide trang 55)."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        # Giảm cosine từ 1.0 về 0.001
        return max(1e-3, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W (slide trang 56)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Cập nhật trọng số EMA sau mỗi bước tối ưu."""
        d = self.decay
        model_params = dict(model.named_parameters())
        shadow_params = dict(self.shadow.named_parameters())

        for name, param in model_params.items():
            if name in shadow_params:
                shadow_params[name].data.lerp_(param.data, 1.0 - d)

        # Đồng bộ hóa các buffers (running_mean, running_var của BatchNorm)
        model_buffers = dict(model.named_buffers())
        shadow_buffers = dict(self.shadow.named_buffers())
        for name, buffer in model_buffers.items():
            if name in shadow_buffers:
                shadow_buffers[name].copy_(buffer)

    def apply_shadow(self, model: nn.Module) -> nn.Module:
        """Trả về mô hình mang trọng số EMA để đánh giá."""
        return self.shadow


def train_one_epoch(model: nn.Module, loader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, scaler,
                    cfg: Config, device: torch.device, ema: EMA | None = None) -> Dict[str, float]:
    """Huấn luyện mô hình trong một epoch duy nhất."""
    model.train()
    # Nếu chế độ là "frozen": giữ các lớp BatchNorm của backbone ở chế độ eval
    if cfg.init == "frozen":
        freeze_backbone(model)

    running_loss = 0.0
    total_samples = 0
    use_cuda_amp = cfg.amp and (device.type == "cuda")

    for batch in loader:
        images, targets, _ = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        # Xử lý trộn mẫu Mixup / CutMix
        if cfg.mix in ("mixup", "cutmix"):
            images, mixed_targets = mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)

        optimizer.zero_grad(set_to_none=True)

        # Lan truyền tiến với Mixed Precision
        with torch.autocast(device_type=device.type, enabled=use_cuda_amp):
            outputs = model(images)
            if cfg.mix in ("mixup", "cutmix"):
                loss = mixed_loss(criterion, outputs, mixed_targets)
            else:
                loss = criterion(outputs, targets)

        # Lan truyền ngược và cập nhật trọng số
        if scaler is not None and use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]

    return {"train_loss": epoch_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> Tuple[List[str], np.ndarray, np.ndarray, float]:
    """Đánh giá mô hình trên tập validation hoặc test (chế độ eval, không tính gradient)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_filenames: List[str] = []
    all_targets: List[int] = []
    all_logits: List[np.ndarray] = []

    use_cuda_amp = torch.cuda.is_available() and device.type == "cuda"

    with torch.inference_mode():
        for batch in loader:
            images, targets, filenames = batch
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = images.size(0)

            with torch.autocast(device_type=device.type, enabled=use_cuda_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_filenames.extend(filenames)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_logits.append(outputs.float().cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    y_true = np.array(all_targets, dtype=np.int64)
    logits = np.concatenate(all_logits, axis=0)

    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: List[Dict[str, Any]], path: str | Path, title: str) -> None:
    """Vẽ và lưu biểu đồ đường cong huấn luyện chuẩn (GUIDE.md mục 6.2)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]
    val_top1 = [h.get("val_top1", 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Đồ thị Loss theo Epoch
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-s", label="Val Loss", linewidth=2)
    ax1.set_title("Hàm mất mát (Loss) theo Epoch", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Đồ thị Metric theo Epoch
    ax2.plot(epochs, val_macro_f1, "g-^", label="Val Macro-F1", linewidth=2)
    ax2.plot(epochs, val_top1, "m-d", label="Val Top-1 Acc", linewidth=2)
    ax2.set_title("Độ đo chất lượng trên Validation", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric (0.0 - 1.0)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _softmax(z: np.ndarray) -> np.ndarray:
    """Hàm Softmax tính toán xác suất từ ma trận logit."""
    z_max = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - z_max)
    return e / np.sum(e, axis=-1, keepdims=True)


def run(cfg: Config) -> Dict[str, Any]:
    """Quy trình huấn luyện và đánh giá một kịch bản hoàn chỉnh từ A đến Z."""
    start_total_time = time.time()
    set_seed(cfg.seed)

    # 1. Khởi tạo các thư mục lưu kết quả
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path("curves").mkdir(parents=True, exist_ok=True)

    # Ghi file config.json
    with open(r_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, ensure_ascii=False)

    print(f"\n==================== [BẮT ĐẦU THÍ NGHIỆM {cfg.exp_id}] ====================")
    print(f"Cấu hình: Backbone={cfg.backbone}, Init={cfg.init}, Loss={cfg.loss}, Aug={cfg.aug}, Seed={cfg.seed}")

    # 2. Đọc và kiểm tra tính toàn vẹn của dữ liệu chia fold
    train_df, val_df, test_df = load_split(cfg.labels_dir, fold=cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir)

    # 3. Dựng DataLoader
    train_tf = build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = build_transforms(train=False, img_size=cfg.img_size)

    train_loader = make_loader(
        train_df, cfg.images_dir, train_tf, cfg.batch_size,
        train=True, sampler=cfg.sampler, num_workers=cfg.num_workers
    )
    val_loader = make_loader(
        val_df, cfg.images_dir, val_tf, cfg.batch_size,
        train=False, num_workers=cfg.num_workers
    )

    # 4. Khởi tạo mô hình và chuyển sang thiết bị tính toán
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        cfg.backbone,
        pretrained=True,
        num_classes=9,
        drop_rate=cfg.drop_rate,
        init=cfg.init,
    ).to(device)

    n_params = count_params(model)
    gmacs = count_gmacs(model, img_size=cfg.img_size)
    print(f"[Model Stats] Tham số: {n_params}M | GMACs: {gmacs}")

    # 5. Xây dựng hàm loss
    loss_weights = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().sort_index().to_dict()
        beta_val = cfg.class_weight_beta if cfg.class_weight_beta is not None else 0.0
        loss_weights = class_weights(counts, beta=beta_val).to(device)

    criterion = build_criterion(
        kind=cfg.loss,
        smoothing=cfg.label_smoothing,
        gamma=cfg.focal_gamma,
        weight=loss_weights,
    )
    # Criterion dùng cho đánh giá validation luôn là Cross-Entropy chuẩn
    val_criterion = nn.CrossEntropyLoss()

    # 6. Thiết lập Optimizer, Scheduler, Scaler, EMA
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and (device.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay is not None else None

    # 7. Vòng lặp huấn luyện qua từng epoch
    history: List[Dict[str, Any]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_val_data: Tuple[List[str], np.ndarray, np.ndarray] | None = None
    epoch_durations: List[float] = []

    for epoch in range(1, cfg.epochs + 1):
        t_start_epoch = time.time()
        train_res = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=ema
        )
        epoch_dur = time.time() - t_start_epoch
        epoch_durations.append(epoch_dur)

        # Lấy mô hình mang trọng số EMA nếu có để đánh giá
        eval_model = ema.apply_shadow(model) if ema is not None else model

        # Đánh giá trên tập Validation
        val_filenames, val_y_true, val_logits, val_loss = evaluate(eval_model, val_loader, val_criterion, device)
        val_probs = _softmax(val_logits)
        val_y_pred = val_probs.argmax(axis=1)

        # Tính toán chỉ số chính xác bằng compute_metrics của repo
        val_metrics = compute_metrics(val_y_true, val_y_pred, val_probs)
        macro_f1 = val_metrics["macro_f1"]
        top1_acc = val_metrics["top1"]

        log_item = {
            "epoch": epoch,
            "train_loss": round(train_res["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_top1": round(top1_acc, 4),
            "lr": train_res["lr"],
            "duration_s": round(epoch_dur, 2),
        }
        history.append(log_item)

        print(f"Epoch [{epoch:02d}/{cfg.epochs:02d}] - Train Loss: {log_item['train_loss']:.4f} | "
              f"Val Loss: {log_item['val_loss']:.4f} | Val Macro-F1: {macro_f1:.4f} | "
              f"Top-1: {top1_acc*100:.2f}% ({epoch_dur:.1f}s)")

        # Lưu Checkpoint tốt nhất theo chỉ số Macro-F1 Validation (Quy tắc N3)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_val_data = (val_filenames, val_y_true, val_probs)
            # Lưu checkpoint
            torch.save(eval_model.state_dict(), r_dir / "best_model.pth")
            np.save(r_dir / "val_logits.npy", val_logits)

    print(f"\n[Kết quả huấn luyện] Best Epoch: {best_epoch} với Val Macro-F1: {best_macro_f1:.4f}")

    # 8. Lưu file dự đoán Validation của checkpoint tốt nhất
    if best_val_data is not None:
        v_names, v_true, v_probs = best_val_data
        save_predictions(pred_path(cfg, "val"), v_names, v_true, v_probs)

    # 9. Chỉ chạy đánh giá trên TEST khi bật save_test_predictions (Bước 4 Chung kết)
    test_metrics = None
    if cfg.save_test_predictions:
        print(f"[Đánh giá TEST] Nạp checkpoint tốt nhất từ epoch {best_epoch} để đánh giá Test đúng một lần...")
        best_model_state = torch.load(r_dir / "best_model.pth", map_location=device)
        model.load_state_dict(best_model_state)

        test_loader = make_loader(
            test_df, cfg.images_dir, val_tf, cfg.batch_size,
            train=False, num_workers=cfg.num_workers
        )
        test_filenames, test_y_true, test_logits, test_loss = evaluate(model, test_loader, val_criterion, device)
        test_probs = _softmax(test_logits)
        test_y_pred = test_probs.argmax(axis=1)

        # Lưu kết quả test và logits
        save_predictions(pred_path(cfg, "test"), test_filenames, test_y_true, test_probs)
        np.save(r_dir / "test_logits.npy", test_logits)

        test_metrics = compute_metrics(test_y_true, test_y_pred, test_probs)
        print(f"[TEST KẾT QUẢ CUỐI] Macro-F1: {test_metrics['macro_f1']:.4f} | "
              f"Top-1 Acc: {test_metrics['top1']*100:.2f}% | ECE: {test_metrics['ece']:.4f}")

    # 10. Ghi log lịch sử training và vẽ đồ thị
    history_df = pd.DataFrame(history)
    history_df.to_csv(r_dir / "history.csv", index=False)

    curve_path = Path("curves") / f"{cfg.exp_id}_{cfg.backbone}.png"
    plot_curves(history, curve_path, title=f"Thí nghiệm {cfg.exp_id} ({cfg.backbone})")

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "best_epoch": best_epoch,
        "val_macro_f1": best_macro_f1,
        "avg_epoch_time_s": round(float(np.mean(epoch_durations)), 2),
        "params_m": n_params,
        "gmacs": gmacs,
        "total_time_min": round((time.time() - start_total_time) / 60.0, 2),
    }
    if test_metrics is not None:
        summary["test_macro_f1"] = test_metrics["macro_f1"]
        summary["test_top1"] = test_metrics["top1"]
        summary["test_ece"] = test_metrics["ece"]

    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Chuyển đổi danh sách tham số dạng ['key=value', ...] thành dictionary có ép kiểu theo Config."""
    cfg_fields = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}

    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Định dạng tham số sai (cần dạng KEY=VALUE): '{pair}'")
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in cfg_fields:
            raise KeyError(f"Trường '{k}' không tồn tại trong lớp Config! Các trường hợp lệ: {list(cfg_fields.keys())}")

        field_type = cfg_fields[k]

        # Xử lý các giá trị đặc biệt
        if v.lower() in ("none", "null"):
            overrides[k] = None
        elif v.lower() == "true":
            overrides[k] = True
        elif v.lower() == "false":
            overrides[k] = False
        else:
            # Ép kiểu tự động theo kiểu dữ liệu khai báo
            if "int" in str(field_type):
                overrides[k] = int(v)
            elif "float" in str(field_type):
                overrides[k] = float(v)
            elif "bool" in str(field_type):
                overrides[k] = (v.lower() in ("1", "true", "yes"))
            else:
                overrides[k] = str(v)

    return overrides


def main() -> None:
    """Điểm vào chạy kịch bản từ dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện thí nghiệm DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Ghi đè siêu tham số dạng KEY=VALUE")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    run(cfg)


if __name__ == "__main__":
    main()
