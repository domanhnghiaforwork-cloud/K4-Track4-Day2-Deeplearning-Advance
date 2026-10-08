# Báo Cáo Thực Nghiệm Lab Day 2: Backbone, Công Thức Huấn Luyện và Kỹ Thuật Suy Luận trên Dataset DeepWeeds

- **Học viên**: Đỗ Mạnh Nghĩa
- **Mã số sinh viên**: 2A202602971
- **Khoá học**: Track 4 · Ngày 2 · Thị giác Máy tính Nâng cao (VinUniversity / VinAI)
- **Thư mục bài nộp**: `submissions/2A202602971_do_manh_nghia/`

---

## 1. Tóm Tắt (Executive Summary)

Bài thực nghiệm này khảo sát toàn diện bài toán phân loại đa lớp thực vật cỏ dại ngoài thực địa trên bộ dữ liệu **DeepWeeds** (17.509 ảnh RGB, 9 lớp gồm 8 loài cỏ mục tiêu và lớp `Negative` chiếm 52,01%). Chúng tôi tiến hành thực nghiệm so sánh công bằng trên **5 kiến trúc backbone** (ResNet-50, ConvNeXt-Tiny, ResNeXt-50, DeiT-Small, MobileNetV3-Large), bóc tách riêng biệt đóng góp của **4 trục công thức huấn luyện** (khởi tạo, augmentation CutMix, hàm mất mát Label Smoothing/Focal, điều hòa trọng số EMA) và đánh giá **4 kỹ thuật suy luận** gắn liền với độ trễ thời gian thực.

Cấu hình chung kết **F01** kết hợp backbone hiện đại **ConvNeXt-Tiny**, kỹ thuật tăng cường **CutMix** ($\alpha=1.0$), mất mát **Label Smoothing** ($\varepsilon=0.1$) và **EMA** (decay $0.999$), kèm hiệu chuẩn **Temperature Scaling** ($T \approx 0.60$) đạt kết quả xuất sắc trên tập kiểm tra (Test fold 0, trung bình 3 seed):
- **Top-1 Accuracy**: **97.54% ± 0.27%** (vượt xa mốc ResNet-50 95.7% của bài báo gốc Olsen et al., 2019).
- **Macro-F1**: **0.9686 ± 0.0039** (cải thiện $\Delta = +0.1633$ so với mốc nền $0.8054$, vượt xa độ lệch chuẩn nhiễu hạt giống $s = 0.0039$).
- **Hai lớp khó nhất**: Recall của **Chinee apple** đạt **92.9% ± 1.2%** (mốc bài báo 88.5%) và **Snake weed** đạt **95.8% ± 1.0%** (mốc bài báo 88.8%).
- **Độ trễ suy luận**: Độ trễ phân vị $p95 = \mathbf{10.81\text{ ms}}$ ở kích thước lô $1$ trên GPU Tesla T4 (đáp ứng xuất sắc ngân sách $\le 100\text{ ms}$ cho chu kỳ cảm biến robot nông nghiệp tự hành).

---

## 2. Dữ Liệu và Thiết Lập Thực Nghiệm

### 2.1 Tập dữ liệu DeepWeeds và Phân bố lớp (EDA)
Bộ dữ liệu DeepWeeds gồm $17.509$ ảnh kích thước $256 \times 256$ thu thập qua robot nông nghiệp tại 8 địa điểm nông trường ở miền bắc Queensland, Úc. 

| Mã Lớp | Tên Loài Thực Vật | Số Ảnh (Toàn tập) | Số Ảnh Train (Fold 0) | Số Ảnh Val (Fold 0) | Số Ảnh Test (Fold 0) | Tỷ lệ (%) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple (*Ziziphus mauritiana*) | 1.125 | 674 | 225 | 226 | 6.43% |
| 1 | Lantana (*Lantana camara*) | 1.064 | 638 | 213 | 213 | 6.08% |
| 2 | Parkinsonia (*Parkinsonia aculeata*) | 1.031 | 619 | 205 | 207 | 5.89% |
| 3 | Parthenium (*Parthenium hysterophorus*) | 1.022 | 613 | 204 | 205 | 5.84% |
| 4 | Prickly acacia (*Vachellia nilotica*) | 1.062 | 637 | 212 | 213 | 6.07% |
| 5 | Rubber vine (*Cryptostegia grandiflora*) | 1.009 | 606 | 201 | 202 | 5.76% |
| 6 | Siam weed (*Chromolaena odorata*) | 1.074 | 644 | 215 | 215 | 6.13% |
| 7 | Snake weed (*Stachytarpheta spp.*) | 1.016 | 609 | 203 | 204 | 5.80% |
| 8 | **Negative** (Cây cỏ bản địa không phải loài đích) | **9.106** | **5.464** | **1.820** | **1.822** | **52.01%** |
| **Tổng** | **9 lớp** | **17.509** | **10.504** | **3.498** | **3.507** | **100%** |

**Nhận xét phân bố và quy tắc đo lường:**
- Lớp `Negative` chiếm ưu thế áp đảo (~52%). Nếu chỉ đo đơn thuần bằng **Top-1 Accuracy**, mô hình đoán thiên lệch hoàn toàn về `Negative` vẫn có thể đạt trên $52\%$. Do đó, chỉ số đánh giá cốt lõi bắt buộc phải là **Macro-F1** (trung bình không trọng số của 9 lớp) để bảo đảm các lớp cỏ dại có số lượng ít vẫn được đánh giá công bằng.
- Tỷ lệ chia tập thực tế khớp tỉ lệ phân tầng 60/20/20 của fold 0: $10.504$ train / $3.498$ val / $3.507$ test.
- Kiểm tra toàn vẹn (S1–S4): Giao của từng cặp $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$; hợp ba tập đúng bằng $17.509$ ảnh; $100\%$ file trong CSV tồn tại hợp lệ trên đĩa.

### 2.2 Kiểm tra Tính Đúng Đắn Pipeline (Sanity Check)
Trước khi tiến hành huấn luyện hàng loạt, chúng tôi thực hiện các bài kiểm tra gỡ lỗi theo khuyến nghị của Slide (trang 59):
1. **Initial Loss Check**: Với 9 lớp phân loại, hàm mất mát Cross-Entropy với phân bố ngẫu nhiên lý thuyết phải là $-\ln(1/9) \approx 2.1972$. Giá trị đo được trên mô hình khởi tạo ngẫu nhiên là $2.2014 \approx 2.197$, xác nhận đầu ra logits và loss function được khởi tạo chuẩn xác.
2. **Overfit 1 batch nhỏ**: Thử nghiệm huấn luyện trên một mini-batch gồm 8 ảnh mẫu. Sau 40 bước cập nhật bằng AdamW, loss giảm từ $2.21$ xuống $0.0084 < 0.1$, chứng minh gradient lan truyền ngược chính xác, không bị triệt tiêu hay nổ gradient.
3. **Focal Loss Contract**: Đã chạy unit test kiểm tra $\text{FocalLoss}(\gamma=0) \equiv \text{CrossEntropyLoss}$, độ lệch tuyệt đối $< 10^{-6}$.
4. **Độ nhất quán chế độ mô hình**: Đảm bảo gọi `model.train()` trong chu kỳ cập nhật và chuyển `model.eval()`, `torch.inference_mode()` trong mọi lượt đánh giá validation và test.

### 2.3 Công thức nền (Baseline Recipe - T00)
Tất cả các mô hình trong bước so sánh backbone đều tuân thủ cùng một công thức nền:
- **Khởi tạo**: Trọng số tiền huấn luyện ImageNet-1k, thay lớp phân loại cuối (classifier head) 9 lớp.
- **Tiền xử lý & Augmentation**: Train dùng `RandomResizedCrop(224, scale=(0.8, 1.0))` + lật ngang ngẫu nhiên (`RandomHorizontalFlip(p=0.5)`). Val/Test dùng `Resize(256)` kèm `CenterCrop(224)`.
- **Chuẩn hóa**: ImageNet mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`.
- **Bộ tối ưu hóa**: AdamW, chia 2 nhóm tham số: LR backbone $= 1 \times 10^{-4}$, LR head $= 1 \times 10^{-3}$ (gấp 10 lần). Weight decay $= 0.05$ (loại trừ các tham số bias và normalization layer theo slide trang 52).
- **Lịch trình LR**: Tuyến tính Warmup 1 epoch đầu, sau đó suy giảm Cosine Annealing về $10^{-6}$.
- **Thời lượng**: 12 epoch, batch size 64 (DeiT dùng batch size 32 do giới hạn bộ nhớ GPU), bật Automatic Mixed Precision (AMP FP16).
- **Phần cứng**: GPU NVIDIA Tesla T4 16GB, PyTorch 2.11.0+cu130, `timm` 0.9.16.

---

## 3. Bước 1: So Sánh Backbone (≥ 5 Kiến Trúc)

Chúng tôi lựa chọn 5 backbone đại diện cho các họ kiến trúc then chốt trong thị giác máy tính hiện đại:
1. **ResNet-50** (`resnet50.a1_in1k`): Kiến trúc residual kinh điển, mốc so sánh cơ sở.
2. **ConvNeXt-Tiny** (`convnext_tiny.fb_in1k`): Kiến trúc tích chập hiện đại hóa (kernel 7x7 lớn, 75% khối đảo ngược, LayerNorm, GELU).
3. **ResNeXt-50 32x4d** (`resnext50_32x4d.a1_in1k`): Mở rộng ResNet qua trục Cardinality (32 nhánh song song).
4. **DeiT-Small** (`deit_small_patch16_224.fb_in1k`): Đại diện Vision Transformer (patch 16x16, self-attention toàn cục, chưng cất tri thức).
5. **MobileNetV3-Large** (`mobilenetv3_large_100.ra_in1k`): Đại diện mạng tích chập siêu nhẹ với depthwise separable convolution và khối SE.

### Bảng 1: Kết quả so sánh 5 Backbone trên tập Validation (cùng Seed 0, 12 Epochs)

| exp_id | Backbone | Họ Kiến Trúc | Số Tham Số (M) | GMAC (GFLOPs) | Macro-F1 Val | Top-1 Val | Thời Gian Train/Epoch (s) | Độ Trễ Suy Luận p95 Batch 1 (ms) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B01** | `resnet50` | CNN Residual | 23.53 | 4.13 | 0.8023 | 0.8872 | 49.34 | 6.82 |
| **B02** | `convnext_tiny` | Modernized CNN | 27.83 | 4.46 | **0.9680** | **0.9749** | 55.14 | 10.81 |
| **B03** | `resnext50_32x4d` | Multi-branch CNN | 23.00 | 4.29 | 0.8180 | 0.8995 | 63.12 | 9.45 |
| **B04** | `deit_small` | Vision Transformer | 21.67 | 4.24 | 0.9448 | 0.9572 | 43.80 | 8.12 |
| **B05** | `mobilenetv3` | Lightweight Edge | 4.21 | 0.22 | 0.7649 | 0.8655 | 37.09 | **3.42** |

### Nhận xét & Phân tích chuyên sâu:
1. **Sự vượt trội của ConvNeXt-Tiny (B02)**: ConvNeXt-Tiny đạt Macro-F1 **0.9680**, vượt trội áp đảo ResNet-50 (**+0.1657** điểm F1) dù số tham số chỉ nhỉnh hơn không đáng kể ($27.8\text{M}$ so với $23.5\text{M}$). Điều này minh chứng cho phát hiện trong slide trang 37 và bài báo *A ConvNet for the 2020s*: việc tái cấu trúc mạng tích chập theo tư duy transformer (inverted bottleneck, kernel $7 \times 7$ mở rộng trường tiếp nhận hiệu dụng, thay thế BatchNorm bằng LayerNorm) giúp mạng trích xuất đặc trưng hình thái thực vật vượt trội.
2. **DeiT-Small (B04) vs ConvNeXt (B02)**: DeiT-Small đạt F1 rất cao ($0.9448$), chứng minh khả năng mô hình hóa quan hệ toàn cục qua attention patch. Tuy nhiên, nó vẫn xếp sau ConvNeXt-Tiny ($0.9680$). Nguyên nhân là Vision Transformer thiếu thiên kiến quy nạp về tính cục bộ và bất biến dịch chuyển (slide trang 32), đòi hỏi dữ liệu huấn luyện lớn hơn nhiều so với quy mô ~10k ảnh của DeepWeeds.
3. **Nghịch lý FLOPs vs Độ trễ (FLOPs $\neq$ Latency)**:
   - ResNeXt-50 có GMAC ($4.29$) tương đương ResNet-50 ($4.13$) nhưng độ trễ thực tế lại chậm hơn đáng kể ($9.45\text{ ms}$ vs $6.82\text{ ms}$) và thời gian train mỗi epoch cao nhất ($63.12\text{ s}$). Lý do: cấu trúc đa nhánh song song làm tăng lưu lượng truy cập bộ nhớ (Memory Access Cost - MAC) và phân mảnh luồng tính toán GPU (slide trang 43).
   - MobileNetV3-Large có chi phí tính toán cực thấp ($0.22\text{ GMAC}$) và tốc độ siêu nhanh ($3.42\text{ ms}$), nhưng Macro-F1 chỉ đạt $0.7649$, do năng lực biểu diễn của các kênh depthwise bị thu hẹp quá mức trên dữ liệu phức tạp ngoài đồng ruộng.

> **Quyết định lựa chọn**: Chọn **ConvNeXt-Tiny** làm backbone chủ lực để đi tiếp vào Bước 2 (Công thức huấn luyện) và Bước 3 (Kỹ thuật suy luận), vì mô hình đem lại chất lượng biểu diễn cao nhất và độ trễ $10.81\text{ ms}$ vẫn nằm sâu trong ngưỡng an toàn thời gian thực ($\le 100\text{ ms}$).

---

## 4. Bước 2: Tinh Chỉnh Công Thức Huấn Luyện (≥ 3 Trục)

Với backbone ConvNeXt-Tiny đã chọn, chúng tôi giữ nguyên số epoch (12), seed khởi đầu (0), và thực hiện các thí nghiệm có kiểm soát theo nguyên tắc **một thay đổi mỗi lần** (N1).

### Bảng 2: Kết quả Thí nghiệm Công thức Huấn luyện trên ConvNeXt-Tiny

| exp_id | Trục Khảo Sát | Yếu Tố Thay Đổi | Macro-F1 Val | Top-1 Val | $\Delta$ so với Nền | Ghi chú & Đánh giá |
|:---:|---|---|:---:|:---:|:---:|---|
| **T00** | Mốc gốc | ResNet-50 (Finetune, CE, Basic Aug) | 0.8023 | 0.8872 | $0.0000$ | Mốc so sánh cơ sở ban đầu |
| **B02** | Nền ConvNeXt | ConvNeXt-Tiny (Finetune, CE, Basic Aug) | 0.9680 | 0.9749 | $+0.1657$ | Nền chuẩn cho các ablation |
| **T01** | A. Khởi tạo | Train from Scratch (Khởi tạo ngẫu nhiên) | 0.2971 | 0.5825 | $-0.6709$ | Thất bại nặng nề do dữ liệu ít |
| **T02** | A. Khởi tạo | Frozen Backbone (Chỉ train classification head) | 0.8536 | 0.9015 | $-0.1144$ | Đặc trưng ImageNet chưa tối ưu cho cỏ |
| **T03** | B. Augmentation | Thêm ColorJitter (đổi màu, độ tương phản) | 0.9652 | 0.9737 | $-0.0028$ | Biến đổi màu mạnh làm nhiễu sắc thái lá cỏ |
| **T04** | B. Augmentation | **CutMix** ($\alpha = 1.0$, trộn vùng ảnh và nhãn mềm) | **0.9716** | **0.9774** | **$+0.0036$** | Giảm quá khớp bối cảnh, ép học đặc trưng đa vùng |
| **T05** | C. Hàm Loss | **Label Smoothing** ($\varepsilon = 0.1$) | **0.9687** | **0.9754** | $+0.0007$ | Chống overconfidence, cải thiện biên quyết định |
| **T06** | C. Hàm Loss | Focal Loss ($\gamma = 2.0$) | 0.9709 | 0.9769 | $+0.0029$ | Tập trung vào mẫu khó, cân bằng gradient |
| **T07** | F. Chính quy hoá | **EMA Trọng Số** (Decay $0.999$) | 0.9675 | 0.9743 | $-0.0005$ | Giảm dao động tối ưu ở các epoch cuối |
| **T08** | **Tổ Hợp Tối Ưu** | **CutMix + Label Smoothing + EMA** | **0.9721** | **0.9780** | **$+0.0041$** | **Hiệu ứng cộng dồn tốt nhất, chọn vào Chung kết** |

### Phân tích chi tiết từng trục thực nghiệm:
1. **Trục Khởi tạo (A)**:
   - *From scratch (T01 - F1: 0.2971)*: Huấn luyện từ đầu với ~10k ảnh hoàn toàn thất bại. Mạng không đủ mẫu để tự học các bộ lọc tầng thấp (cạnh, góc, texture), minh chứng cho tầm quan trọng sống còn của chuyển giao tri thức từ ImageNet khi số lượng dữ liệu bị giới hạn (slide trang 51).
   - *Linear probe / Frozen (T02 - F1: 0.8536)*: Đóng băng backbone cho kết quả chấp nhận được nhưng kém finetune toàn bộ hơn 11% F1. Điều này chứng tỏ miền ảnh thực vật ngoài đồng cỏ có phân bố thị giác khác biệt đáng kể so với ảnh tổng quát của ImageNet; tinh chỉnh toàn bộ mô hình (với learning rate head gấp 10 lần backbone) là bắt buộc.
2. **Trục Augmentation (B)**:
   - *CutMix (T04 - F1: 0.9716)*: Tăng cường hiệu quả nhất. Bằng cách cắt một mảng từ ảnh này dán sang ảnh khác và chia tỷ lệ nhãn theo diện tích $\lambda$, mô hình không thể chỉ dựa vào một vùng đặc trưng duy nhất (như một nhánh hoa hay góc nền đất) mà bị ép phải học các chi tiết phân biệt trên toàn bộ bức ảnh.
   - *ColorJitter (T03)*: Không mang lại lợi ích rõ rệt do sắc thái xanh và vàng của cỏ dại là đặc trưng phân loại quan trọng; việc biến dạng màu sắc làm mất đi tín hiệu sinh học tự nhiên của thực vật.
3. **Trục Loss (C) và Regularization (F)**:
   - *Label Smoothing (T05)* và *Focal Loss (T06)* đều mang lại cải thiện tích cực so với Cross-Entropy thuần túy. Đặc biệt, Label Smoothing phạt các xác suất đầu ra cực đoan, ngăn head sinh ra logits quá lớn và trực tiếp hỗ trợ bài toán hiệu chuẩn độ tin cậy.
   - *Tổ hợp T08*: Khi kết hợp CutMix + Label Smoothing + EMA, Macro-F1 val đạt đỉnh **0.9721** (Top-1: 97.80%). Ba kỹ thuật này tương hỗ cộng dồn cho nhau: CutMix làm phong phú không gian dữ liệu, Label Smoothing làm mềm mục tiêu tối ưu, và EMA trung bình hóa các biến động ngẫu nhiên trên bề mặt mất mát.

---

## 5. Bước 3: Phương Pháp Suy Luận và Đo Độ Trễ

Chúng tôi cố định checkpoint tốt nhất từ thí nghiệm `T08` (không huấn luyện lại) và kiểm tra 4 phương pháp suy luận trên tập Validation. Độ trễ được đo nghiêm ngặt theo hướng dẫn `GUIDE.md` mục 4.1:
- Chạy **10 lần warmup** trước khi ghi nhận.
- Gọi lệnh đồng bộ **`torch.cuda.synchronize()`** trước và sau mỗi lượt đo.
- Đo liên tục **100 lần lặp**, xuất giá trị phân vị $p50$, $p95$, $p99$ và trung bình.

### Bảng 3: So sánh Đánh đổi Độ chính xác – Hiệu chuẩn – Độ trễ Suy luận (ConvNeXt-Tiny)

| exp_id | Kỹ Thuật Suy Luận | K (views) | Macro-F1 Val | Top-1 Val | ECE Val (15 bins) | p50 (ms) | p95 (ms) | p99 (ms) | Thông Lượng (ảnh/s) | Chi Phí Tương Đối |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **I00** | 1-view chuẩn (mốc) | 1 | 0.9721 | 0.9780 | 0.0897 | 7.55 | **10.81** | 11.79 | 132.50 | 1.0x |
| **I01** | TTA Lật ngang (K=2) | 2 | 0.9741 | 0.9797 | 0.0915 | 15.10 | 21.62 | 23.58 | 66.25 | 2.0x |
| **I04** | FixRes (Test 256x256) | 1 | **0.9757** | **0.9814** | 0.1260 | 9.82 | 14.05 | 15.33 | 101.90 | 1.3x |
| **I07** | **Temperature Scaling** | 1 | 0.9721 | 0.9780 | **0.0040** | 7.55 | **10.81** | 11.79 | 132.50 | **1.0x** |

### Nhận xét & Đánh giá Đánh đổi (Trade-off):
1. **FixRes (I04) cho kịch bản Ngoại tuyến (Offline Analysis)**: Khi kiểm tra ở độ phân giải gốc $256 \times 256$ (thay vì center-crop 224), Macro-F1 đạt đỉnh **0.9757** (Top-1: 98.14%). Hiện tượng này hoàn toàn khớp với lý thuyết FixRes (slide trang 68): mô hình được huấn luyện với `RandomResizedCrop(224)` đã học được các tỷ lệ vật thể nhỏ, nên khi test ở độ phân giải lớn hơn, các chi tiết lá cỏ trở nên rõ nét hơn. Độ trễ chỉ tăng nhẹ từ $10.81\text{ ms} \rightarrow 14.05\text{ ms}$.
2. **TTA Lật ngang (I01)**: TTA giúp tăng nhẹ Macro-F1 lên $0.9741$ (+0.0020), nhưng độ trễ tăng đúng gấp đôi ($21.62\text{ ms}$). Trên robot thời gian thực, chi phí này không đáng đánh đổi so với lợi ích nhỏ; TTA phù hợp hơn cho các hệ thống hậu kiểm hoặc xử lý dữ liệu hàng loạt.
3. **Hiệu chuẩn qua Temperature Scaling (I07)**:
   - Nhiệt độ tối ưu khớp trên Validation: $T \approx 0.6035$.
   - **ECE giảm ngoạn mục từ $0.0897 \rightarrow 0.0040$ (giảm $95.5\%$)** trong khi **giữ nguyên $100\%$ độ chính xác Top-1 và F1** (do phép chia vô hướng bảo toàn thứ tự argmax của logit).
   - Chi phí tính toán là phép chia $z / T$ trong thời gian tính bằng micro-giây (chi phí tăng thêm tương đối là $1.0\times$).

---

## 6. Bước 4: Vòng Chung Kết và Đánh Giá Kiểm Tra (Test Set)

### 6.1 Thiết lập Chung kết F01
- Cấu hình: **ConvNeXt-Tiny + CutMix ($\alpha=1.0$) + Label Smoothing ($0.1$) + EMA ($0.999$) + 1-view Inference**.
- Chạy trên **3 seed độc lập** (Seed 0, Seed 1, Seed 2).
- Đánh giá trên **toàn bộ 3.507 ảnh tập Test Fold 0** đúng một lần duy nhất cho mỗi seed.
- Sử dụng công cụ chuẩn [`eval.py`](file:///d:/nghia/vin/TRACK4/day2/K4-Track4-Day2-Deeplearning-Advance/eval.py) của môn học để tính toán số liệu và tự chấm.

### Bảng 4: Kết quả Vòng Chung kết (Test Fold 0, Đối chiếu Mốc Nền và Bài Báo)

| Tiêu Chí Đánh Giá | Mốc Nền T00 (Seed 0) | Chung Kết F01 (Seed 0) | Chung Kết F01 (Seed 1) | Chung Kết F01 (Seed 2) | **Chung Kết F01 (Mean ± Std)** | Mốc Bài Báo Gốc (Olsen et al., 2019) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top-1 Accuracy** | 0.8872 | 0.9778 | 0.9760 | 0.9723 | **0.9754 ± 0.0027** | 95.7% (ResNet-50) / 95.1% (Incep-v3) |
| **Macro-F1 (Chính)** | 0.8054 | 0.9728 | 0.9678 | 0.9652 | **0.9686 ± 0.0039** | Không báo cáo Macro-F1 |
| Balanced Accuracy | 0.8267 | 0.9752 | 0.9726 | 0.9664 | **0.9714 ± 0.0045** | - |
| ECE Test (Chưa TS) | 0.1284 | 0.0868 | 0.0889 | 0.0955 | **0.0904 ± 0.0045** | Không báo cáo |
| **ECE Test (Sau TS)** | 0.1284 | 0.0079 | 0.0068 | 0.0073 | **0.0073 ± 0.0006** | Không báo cáo |
| **Chinee apple Recall** | 0.7035 | 0.9204 | 0.9248 | 0.9425 | **0.929 ± 0.012** | **88.5%** |
| **Snake weed Recall** | 0.7255 | 0.9608 | 0.9657 | 0.9461 | **0.958 ± 0.010** | **88.8%** |
| Độ trễ p95 batch-1 | 6.82 ms | 10.81 ms | 10.81 ms | 10.81 ms | **10.81 ms** | 180 ms (TX2-TF), 53.4 ms (TensorRT) |

### Bảng 5: Hiệu năng Từng Lớp trên Tập Test (F01 Chung Kết vs Mốc T00)

| Mã | Tên Loài Cây | Số Ảnh Test | Precision (T00) | Recall (T00) | F1 (T00) | Precision (F01 Mean) | Recall (F01 Mean) | F1 (F01 Mean) | Đối chiếu Bài Báo |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Chinee apple | 226 | 0.726 | 0.704 | 0.715 | **0.972 ± 0.005** | **0.929 ± 0.012** | **0.950 ± 0.007** | Vượt mốc (+4.4%) |
| 1 | Lantana | 213 | 0.795 | 0.822 | 0.808 | **0.954 ± 0.016** | **0.975 ± 0.003** | **0.964 ± 0.009** | Xuất sắc |
| 2 | Parkinsonia | 207 | 0.845 | 0.865 | 0.855 | **0.989 ± 0.007** | **0.982 ± 0.006** | **0.985 ± 0.004** | Vượt mốc (97.2%) |
| 3 | Parthenium | 205 | 0.832 | 0.844 | 0.838 | **0.982 ± 0.005** | **0.984 ± 0.007** | **0.983 ± 0.002** | Rất cao |
| 4 | Prickly acacia | 213 | 0.815 | 0.831 | 0.823 | **0.925 ± 0.005** | **0.981 ± 0.009** | **0.952 ± 0.007** | Rất cao |
| 5 | Rubber vine | 202 | 0.854 | 0.837 | 0.845 | **0.982 ± 0.003** | **0.970 ± 0.013** | **0.976 ± 0.006** | Rất cao |
| 6 | Siam weed | 215 | 0.862 | 0.870 | 0.866 | **0.977 ± 0.008** | **0.983 ± 0.007** | **0.980 ± 0.006** | Rất cao |
| 7 | Snake weed | 204 | 0.742 | 0.725 | 0.733 | **0.929 ± 0.027** | **0.958 ± 0.010** | **0.943 ± 0.009** | Vượt mốc (+7.0%) |
| 8 | **Negative** | 1.822 | 0.945 | 0.956 | 0.950 | **0.987 ± 0.002** | **0.980 ± 0.002** | **0.984 ± 0.001** | Vượt mốc (97.6%) |

### 6.2 Phân tích Ma trận Nhầm lẫn và Các Lỗi Tiêu Biểu
Ma trận nhầm lẫn đo trên 3.507 ảnh tập Test (Seed 0):

```
Nhãn Thực \ Nhãn Đoán     0    1    2    3    4    5    6    7    8   (Tổng)
0: Chinee apple         208    1    0    1    1    0    0    8    7    (226)
1: Lantana                0  208    0    0    0    0    0    3    2    (213)
2: Parkinsonia            1    0  202    0    2    0    0    0    2    (207)
3: Parthenium             0    0    0  200    4    0    0    0    1    (205)
4: Prickly acacia         0    0    2    1  207    0    0    0    3    (213)
5: Rubber vine            0    0    0    0    0  194    0    2    6    (202)
6: Siam weed              0    1    0    0    0    0  210    0    4    (215)
7: Snake weed             2    2    0    0    0    0    1  196    3    (204)
8: Negative               4    7    0    1   11    3    3    6 1787   (1822)
```

**Phân tích nguyên nhân nhầm lẫn sinh học giữa Chinee apple $\leftrightarrow$ Snake weed:**
1. Trong ma trận nhầm lẫn, lỗi nhầm lẫn lớn nhất giữa hai loài cỏ xảy ra giữa **Chinee apple** và **Snake weed** (8 ảnh Chinee apple bị đoán nhầm thành Snake weed, và ngược lại 2 ảnh Snake weed bị đoán thành Chinee apple).
2. Hiện tượng này hoàn toàn trùng khớp với phân tích trong bài báo gốc của Olsen et al. (2019):
   - Cả hai loài đều là cây bụi mọc thấp trên nền đồng cỏ khô, phiến lá có răng cưa nhỏ và màu xanh lục sẫm tương đồng.
   - Khi chụp ngoài thực địa dưới góc nhìn của camera robot hướng từ trên xuống (top-down view) với điều kiện ánh sáng nắng gắt hoặc bóng râm che khuất, cấu trúc vân lá và cụm hoa bị bệt, khiến biên quyết định thị giác giữa hai loài trở nên rất mỏng.
3. Tuy nhiên, nhờ kỹ thuật CutMix và cấu trúc kernel $7 \times 7$ của ConvNeXt, mô hình F01 đã đẩy Recall của Chinee apple lên **92.9%** (bài báo: 88.5%) và Snake weed lên **95.8%** (bài báo: 88.8%), giảm tới hơn $60\%$ số lượng ca đoán sai so với mô hình tham chiếu gốc.

---

## 7. Kết Luận và Khuyến Nghị Triển Khai Thực Tế

### 7.1 Trả lời các câu hỏi nghiên cứu cốt lõi
1. **Cấu hình nào tốt nhất? Cải thiện bao nhiêu so với mốc?**
   - Cấu hình tốt nhất là **ConvNeXt-Tiny + CutMix + Label Smoothing + EMA + 1-view Inference (F01)**.
   - So với mốc nền ban đầu (T00 ResNet-50), Macro-F1 tăng từ $0.8054 \rightarrow 0.9686$ (tăng **$+0.1633$**, tương đương $+20.3\%$). Mức tăng này vượt xa độ lệch chuẩn hạt giống ($s = 0.0039$, tức tăng gấp $\approx 42$ lần độ lệch chuẩn), khẳng định đây là sự cải tiến bản chất, không phải do nhiễu seed.
2. **Yếu tố nào đóng góp nhiều nhất: Backbone, Huấn luyện hay Suy luận?**
   - **Backbone** đóng góp bước nhảy vọt lớn nhất ban đầu: Việc đổi từ ResNet-50 sang ConvNeXt-Tiny mang lại mức tăng $+0.1657$ F1 ngay trong công thức nền.
   - **Công thức huấn luyện** mang tính quyết định để bẻ gãy quá khớp và tối ưu hóa biên phân loại: Thiếu khởi tạo ImageNet mô hình sụp đổ hoàn toàn; bổ sung CutMix + LS + EMA giúp gia tăng thêm $+0.0041$ F1 và đặc biệt nâng vọt độ nhạy ở các loài cỏ hiếm.
   - **Kỹ thuật suy luận** mang lại giải pháp toàn vẹn: Temperature Scaling không tăng F1 nhưng giải quyết triệt để rủi ro mô hình tự tin thái quá, giảm ECE xuống mức tiệm cận 0 ($0.0073$).
3. **Độ trễ và Khuyến nghị triển khai trên Robot Nông nghiệp**:
   - Trong ứng dụng thực tế trên robot phun thuốc trừ sâu tự hành, cảm biến camera thường hoạt động ở tần số $10\text{ Hz}$ (ngân sách chu kỳ tính toán $\le 100\text{ ms}$).
   - Cấu hình F01 (ConvNeXt-Tiny, FP32, batch 1) có thời gian xử lý phân vị $p95 = \mathbf{10.81\text{ ms}}$, chiếm chưa đầy $11\%$ ngân sách thời gian, đạt thông lượng **132 khung hình/giây** trên GPU phổ thông.
   - **Khuyến nghị**: Triển khai trực tiếp cấu hình **F01 1-view + Temperature Scaling**. Không nên dùng TTA trên robot do tăng độ trễ lên gấp đôi mà cải thiện không đáng kể.

---

## 8. Hạn Chế và Hướng Đi Tiếp Theo

1. **Hạn chế về chia dữ liệu (Data Split)**: Bộ dữ liệu DeepWeeds được chia ngẫu nhiên theo ảnh (random stratified split), không chia theo vị trí địa lý nông trường hay mùa vụ. Vì vậy, các ảnh trong tập train và test có thể được chụp tại cùng một nông trường trong cùng một buổi thu thập, dẫn đến việc điểm số trên test có phần lạc quan hơn so với thực tế khi robot đi sang cánh đồng hoàn toàn mới.
2. **Rủi ro Lệch phân phối (Domain Shift)**: Trong điều kiện thời tiết khắc nghiệt (mưa, bùn bám trên ống kính camera, ánh sáng chập tối), độ chính xác của mô hình có thể bị suy giảm.
3. **Hướng phát triển trong tương lai**:
   - Áp dụng **Chưng cất tri thức (Knowledge Distillation)**: Dùng ConvNeXt-Tiny làm giáo viên để huấn luyện MobileNetV3-Large, nâng cao F1 của MobileNetV3 từ $0.76$ lên $>0.90$ nhằm triển khai trên các chip AI biên siêu tiết kiệm điện (như Jetson Orin Nano).
   - Khảo sát **K-fold Cross Validation** (Fold 1–4) để đo độ tin cậy của mô hình qua nhiều tập con dữ liệu độc lập.
   - Thử nghiệm **Test-Time Adaptation (TTA-Tent)** nhằm tự động cập nhật thống kê chuẩn hóa khi điều kiện ánh sáng camera thay đổi đột ngột.

---

## 9. Phụ Lục

### 9.1 Bảng Tra Cứu Toàn Bộ Thí Nghiệm (Experiment ID Mapping)
- `B01`: ResNet-50, nền T00.
- `B02`: ConvNeXt-Tiny, nền T00.
- `B03`: ResNeXt-50 32x4d, nền T00.
- `B04`: DeiT-Small patch16, nền T00.
- `B05`: MobileNetV3-Large 100, nền T00.
- `T00`: ResNet-50 baseline.
- `T01`: ConvNeXt-Tiny, scratch.
- `T02`: ConvNeXt-Tiny, frozen backbone.
- `T03`: ConvNeXt-Tiny, ColorJitter.
- `T04`: ConvNeXt-Tiny, CutMix ($\alpha=1.0$).
- `T05`: ConvNeXt-Tiny, Label Smoothing ($\varepsilon=0.1$).
- `T06`: ConvNeXt-Tiny, Focal Loss ($\gamma=2.0$).
- `T07`: ConvNeXt-Tiny, EMA decay $0.999$.
- `T08`: ConvNeXt-Tiny, CutMix + LS + EMA.
- `I00`: 1-view chuẩn.
- `I01`: TTA lật ngang.
- `I04`: FixRes (độ phân giải 256).
- `I07`: Temperature Scaling.
- `F01`: Cấu hình chung kết 3 seed (seed 0, 1, 2) trên Test.

### 9.2 Liên kết Tái Lập (Reproducibility)
- Toàn bộ pipeline được đóng gói và tái lập thông qua file notebook `code/lab_day2.ipynb`.
- File kết quả tổng hợp: [`results.xlsx`](file:///d:/nghia/vin/TRACK4/day2/K4-Track4-Day2-Deeplearning-Advance/submissions/2A202602971_do_manh_nghia/results.xlsx) gồm đủ 7 sheets tiêu chuẩn.
- Thư mục biểu đồ huấn luyện: [`curves/`](file:///d:/nghia/vin/TRACK4/day2/K4-Track4-Day2-Deeplearning-Advance/submissions/2A202602971_do_manh_nghia/curves/) gồm 15 ảnh PNG.
- Thư mục dự đoán: [`predictions/`](file:///d:/nghia/vin/TRACK4/day2/K4-Track4-Day2-Deeplearning-Advance/submissions/2A202602971_do_manh_nghia/predictions/) gồm các file CSV theo định dạng chuẩn của `eval.py`.
