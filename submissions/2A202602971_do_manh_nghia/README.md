# Bài Nộp Lab Day 2: Backbone, Huấn Luyện và Suy Luận trên DeepWeeds

- **Sinh viên**: Đỗ Mạnh Nghĩa
- **Mã số sinh viên**: 2A202602971
- **Khoá học**: Track 4 · Ngày 2 · Thị giác Máy tính Nâng cao (VinUniversity / VinAI)
- **Thư mục nộp bài**: `submissions/2A202602971_do_manh_nghia/`

---

## 1. Cấu Trúc Thư Mục Nộp Bài

Thư mục bài nộp tuân thủ nghiêm ngặt theo quy định tại `README.md` (mục 5) và điều kiện tiên quyết P1 của `RUBRIC.md`:

```
submissions/2A202602971_do_manh_nghia/
├── README.md               # File này: thông tin sinh viên, môi trường, hướng dẫn chạy
├── results.xlsx            # Bảng so sánh đầy đủ 7 sheets (Backbones, Training, Inference, Final, PerClass, Latency, Summary)
├── report.md               # Báo cáo thực nghiệm chuyên sâu (Dàn ý 9 phần chuẩn theo GUIDE.md)
├── curves/                 # 15 ảnh biểu đồ huấn luyện PNG (B01..B05, T00..T08, F01)
├── predictions/            # File dự đoán test và val cho chung kết (F01 3 seeds), mốc (T00) và các ablation
└── code/                   # Toàn bộ code hoàn thiện từ bộ khung starter/
    ├── dataset.py          # Xử lý dữ liệu, kiểm tra S1-S6, transforms, DataLoader
    ├── model.py            # Khởi tạo backbone qua timm, 3 nhóm tham số, đếm params/GMACs
    ├── losses.py           # CrossEntropy, Label Smoothing, Focal Loss, CutMix/Mixup loss
    ├── train.py            # Hàm train.run(cfg) hợp nhất: AMP, Cosine Warmup, EMA, checkpoint theo Macro-F1 val
    ├── inference.py        # TTA, FixRes, Temperature Scaling, gộp BatchNorm
    ├── benchmark.py        # Đo độ trễ p50/p95/p99 đúng chuẩn (warmup, cuda.synchronize)
    └── lab_day2.ipynb      # Notebook thực nghiệm toàn diện trên Colab / Kaggle
```

---

## 2. Môi Trường Thực Nghiệm & Phiên Bản Thư Viện

Thực nghiệm được chạy trên GPU NVIDIA Tesla T4 16GB (Google Colab / Kaggle):
- **Hệ điều hành**: Linux Ubuntu 22.04 LTS (x86_64) / Windows 11
- **Python**: `3.10+` hoặc `3.11`
- **PyTorch**: `2.1.0` hoặc `2.2.0` (CUDA 12.1 / CUDA 13.0)
- **Thư viện chính**:
  ```bash
  pip install timm>=0.9.16 openpyxl pandas numpy scipy matplotlib scikit-learn
  ```

---

## 3. Hướng Dẫn Tái Lập Kết Quả (Reproducibility)

### Cách 1: Chạy trực tiếp qua Notebook
Mở notebook [`code/lab_day2.ipynb`](code/lab_day2.ipynb) trên Google Colab hoặc Kaggle GPU:
1. Chạy ô **Cài đặt môi trường** và **Tải dữ liệu**: Notebook sẽ tự động tải dataset DeepWeeds từ Zenodo và nhãn fold 0 từ GitHub của tác giả Olsen et al.
2. Thực hiện tuần tự các bước:
   - **Bước 0**: EDA và Sanity Check (Loss ban đầu $\approx 2.197$, overfit 1 batch nhỏ).
   - **Bước 1**: Sàng lọc 5 backbone (ResNet-50, ConvNeXt-Tiny, ResNeXt-50, DeiT-Small, MobileNetV3).
   - **Bước 2**: Tinh chỉnh công thức huấn luyện 4 trục (Khởi tạo, Augmentation CutMix, Loss LS/Focal, EMA).
   - **Bước 3**: Kỹ thuật suy luận và đo độ trễ (1-view, TTA, FixRes, Temperature Scaling).
   - **Bước 4**: Huấn luyện cấu hình chung kết F01 trên 3 seed (Seed 0, 1, 2) và xuất file dự đoán test.
   - **Bước 5**: Xuất file tổng hợp `results.xlsx` đầy đủ 7 sheets.

### Cách 2: Chạy kiểm tra và tự chấm điểm bằng CLI (qua `eval.py`)
Từ thư mục gốc của repository:

```bash
# 1. Tính chỉ số cho cấu hình chung kết F01 (3 seed) trên tập Test
python eval.py score \
    --pred "submissions/2A202602971_do_manh_nghia/predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv \
    --tag F01

# 2. Tính chỉ số cho cấu hình mốc nền T00 trên tập Test
python eval.py score \
    --pred "submissions/2A202602971_do_manh_nghia/predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv \
    --tag T00

# 3. Tự chấm điểm mục I (Chất lượng mô hình - tối đa 20 điểm)
python eval.py grade \
    --final "submissions/2A202602971_do_manh_nghia/predictions/F01_cal_seed*_test.csv" \
    --uncal "submissions/2A202602971_do_manh_nghia/predictions/F01_seed*_test.csv" \
    --baseline "submissions/2A202602971_do_manh_nghia/predictions/T00_seed*_test.csv" \
    --final-val "submissions/2A202602971_do_manh_nghia/predictions/F01_seed*_val.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv \
    --latency-p95-ms 10.81
```

---

## 4. Tóm Tắt Kết Quả Chính Thức Đạt Được

- **Điểm Phần I (Chất lượng mô hình)**: **20 / 20 điểm tuyệt đối** khi chạy với công cụ tự chấm [`eval.py`](../../eval.py).
  - **I1 - Top-1 Accuracy**: **97.54% ± 0.27%** ($\ge 95.7\% \rightarrow$ **7/7** điểm).
  - **I2 - Macro-F1 Cải thiện**: Final $0.9686$ vs Mốc $0.8054$, $\Delta = +0.1633 \gg s = 0.0039$ ($\rightarrow$ **5/5** điểm).
  - **I3 - Recall hai lớp khó**: Chinee apple đạt **92.9%** (mốc 88.5%), Snake weed đạt **95.8%** (mốc 88.8%) ($\rightarrow$ **4/4** điểm).
  - **I4a - Hiệu chuẩn ECE**: Giảm từ $0.0904$ xuống **$0.0073$** sau Temperature Scaling ($\rightarrow$ **1/1** điểm).
  - **I4b - Độ ổn định Val/Test**: Chênh lệch Macro-F1 giữa Val ($0.9709$) và Test ($0.9686$) chỉ $0.0023 \le 0.02$ ($\rightarrow$ **1/1** điểm).
  - **I5 - Độ trễ thời gian thực**: Batch 1 phân vị $p95 = \mathbf{10.81\text{ ms}} \le 100\text{ ms}$ trên GPU Tesla T4 ($\rightarrow$ **2/2** điểm).
