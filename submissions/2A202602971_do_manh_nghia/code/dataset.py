"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Mô-đun quản lý dữ liệu DeepWeeds cho toàn bộ quy trình:
- Đọc phân chia tập train/val/test theo Fold 0 (Quy tắc S1-S6)
- Kiểm tra tính toàn vẹn (không trùng lặp, hợp đủ 17.509 ảnh)
- Xây dựng pipeline tiền xử lý và tăng cường dữ liệu (Data Augmentation)
- Tạo Dataset và DataLoader với hỗ trợ sampler cân bằng lớp
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Chuẩn hóa ImageNet
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (Quy tắc S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame: (train_df, val_df, test_df).
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    for p in (train_file, val_file, test_file):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file nhãn chia fold: {p}")

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Các kiểm tra, nếu có lỗi sẽ raise AssertionError để dừng chương trình ngay:
      1. Số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
      2. Giao của từng cặp tập theo Filename phải RỖNG (train ∩ val, train ∩ test, val ∩ test)
      3. Hợp ba tập phải bằng đúng 17.509 ảnh
      4. Mọi Filename đều tồn tại trong `images_dir` (nếu images_dir tồn tại)
    Trả về dict số liệu thống kê để đưa vào báo cáo và log.
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_imgs = n_train + n_val + n_test

    # 1. Kiểm tra tổng số ảnh và tính duy nhất
    files_train = set(train_df["Filename"])
    files_val = set(val_df["Filename"])
    files_test = set(test_df["Filename"])

    assert len(files_train) == n_train, "Có tên file trùng lặp trong train_df!"
    assert len(files_val) == n_val, "Có tên file trùng lặp trong val_df!"
    assert len(files_test) == n_test, "Có tên file trùng lặp trong test_df!"

    # 2. Giao của từng cặp tập theo Filename phải RỖNG
    inter_train_val = files_train.intersection(files_val)
    inter_train_test = files_train.intersection(files_test)
    inter_val_test = files_val.intersection(files_test)

    assert len(inter_train_val) == 0, f"Rò rỉ dữ liệu giữa train và val: {len(inter_train_val)} ảnh!"
    assert len(inter_train_test) == 0, f"Rò rỉ dữ liệu giữa train và test: {len(inter_train_test)} ảnh!"
    assert len(inter_val_test) == 0, f"Rò rỉ dữ liệu giữa val và test: {len(inter_val_test)} ảnh!"

    # 3. Hợp ba tập phải bằng đúng 17.509 ảnh
    union_all = files_train.union(files_val).union(files_test)
    EXPECTED_TOTAL = 17509
    assert len(union_all) == EXPECTED_TOTAL, f"Hợp ba tập là {len(union_all)} ảnh, kỳ vọng {EXPECTED_TOTAL}!"

    # 4. Kiểm tra sự tồn tại của thư mục và file trong images_dir
    img_dir_path = Path(images_dir)
    assert img_dir_path.exists(), (
        f"LỖI ĐƯỜNG DẪN: Thư mục chứa ảnh '{images_dir}' không tồn tại!\n"
        f"Gợi ý trên Colab: Hãy kiểm tra bằng `!ls {images_dir}` hoặc đặt đường dẫn tuyệt đối, ví dụ: '/content/data/images'."
    )

    missing_files = []
    for fn in list(union_all)[:100]:  # Kiểm tra nhanh 100 file đầu tiên
        if not (img_dir_path / fn).exists():
            missing_files.append(fn)

    if missing_files:
        raise FileNotFoundError(
            f"Thư mục '{images_dir}' không chứa các file ảnh của dataset!\n"
            f"Thiếu các file ví dụ: {missing_files[:5]}.\n"
            f"Gợi ý: Nếu ảnh bị giải nén thẳng vào 'data/', hãy chạy: '!mkdir -p data/images && mv data/*.jpg data/images/'."
        )

    # Thống kê phân bố lớp
    per_class = {
        "train": train_df["Label"].value_counts().sort_index().to_dict(),
        "val": val_df["Label"].value_counts().sort_index().to_dict(),
        "test": test_df["Label"].value_counts().sort_index().to_dict(),
    }

    stats = {
        "n": {"train": n_train, "val": n_val, "test": n_test, "total": total_imgs},
        "ratio": {
            "train": round(n_train / total_imgs, 4),
            "val": round(n_val / total_imgs, 4),
            "test": round(n_test / total_imgs, 4),
        },
        "overlap": {
            "train_val": len(inter_train_val),
            "train_test": len(inter_train_test),
            "val_test": len(inter_val_test),
        },
        "per_class": per_class,
        "missing_files_count": len(missing_files),
    }

    print(f"[check_split] Đạt chuẩn: Train={n_train} ({stats['ratio']['train']*100:.1f}%), "
          f"Val={n_val} ({stats['ratio']['val']*100:.1f}%), "
          f"Test={n_test} ({stats['ratio']['test']*100:.1f}%), Tổng={total_imgs}")
    return stats


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo torchvision transform cho train hoặc eval.

    Tham số `aug` định nghĩa mức độ augmentation (Trục B của GUIDE.md):
      - "basic": RandomResizedCrop(img_size, scale=(0.08, 1.0)) + RandomHorizontalFlip()
      - "color": Basic + ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
      - "trivial": Basic + TrivialAugmentWide()
      - "randaug": Basic + RandAugment(num_ops=2, magnitude=9)
      - "none": Resize(img_size) (không crop, không flip ngẫu nhiên)

    Quy tắc eval/test: Không áp dụng biến đổi ngẫu nhiên.
      Resize 256 -> CenterCrop(img_size) -> ToTensor -> Normalize.
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        transform_list: List[Any] = []
        if aug == "none":
            transform_list.append(T.Resize((img_size, img_size)))
        else:
            # Cơ bản: RandomResizedCrop và lật ngang
            transform_list.append(T.RandomResizedCrop(img_size, scale=(0.08, 1.0)))
            transform_list.append(T.RandomHorizontalFlip(p=0.5))

            if aug == "color":
                transform_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            elif aug == "trivial":
                transform_list.append(T.TrivialAugmentWide())
            elif aug == "randaug":
                transform_list.append(T.RandAugment(num_ops=2, magnitude=9))
            elif aug == "basic":
                pass
            else:
                raise ValueError(f"Chế độ aug '{aug}' không được hỗ trợ!")

        transform_list.extend([
            T.ToTensor(),
            normalize,
        ])
        return T.Compose(transform_list)

    else:
        # Tiền xử lý chuẩn cho Validation và Test (không có tính ngẫu nhiên)
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize,
            ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về bộ 3 giá trị: (image_tensor, int(label), filename: str).
    Tên file cần được bảo toàn để phục vụ ghi file dự đoán theo chuẩn của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames: List[str] = self.df["Filename"].tolist()
        self.labels: List[int] = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fn = self.filenames[i]
        img_path = self.images_dir / fn
        # Mở ảnh bằng PIL và ép kiểu sang RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[i], fn


def _seed_worker(worker_id: int):
    """Cố định seed cho từng worker của DataLoader để đảm bảo tính tái lập."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2) -> DataLoader:
    """Tạo DataLoader hoàn chỉnh cho tập train hoặc val/test.

    - train=True: Bật shuffle (hoặc dùng WeightedRandomSampler nếu sampler="balanced")
    - train=False: Giữ nguyên thứ tự tuần tự để khớp logit với tên file
    - sampler="balanced": Trọng số tỷ lệ nghịch với số mẫu của lớp (xử lý mất cân bằng lớp)
    """
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    data_sampler = None
    shuffle = False

    if train:
        if sampler == "balanced":
            # Tính trọng số nghịch đảo với số mẫu lớp: w_c = 1 / N_c
            class_counts = df["Label"].value_counts().to_dict()
            sample_weights = [1.0 / class_counts[y] for y in df["Label"]]
            data_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        else:
            shuffle = True

        drop_last = len(dataset) > batch_size  # Tránh batch cuối cùng có 1 mẫu làm hỏng BatchNorm
    else:
        shuffle = False
        drop_last = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if train else None,
    )
    return loader
