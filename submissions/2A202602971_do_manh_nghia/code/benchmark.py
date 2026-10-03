"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Mô-đun đo kiểm độ trễ phần cứng chuẩn khoa học:
- Khởi động (Warmup >= 10 lần) để tránh nhiễu do nạp thư viện và cuBLAS
- Đồng bộ GPU (torch.cuda.synchronize) trước và sau mỗi lượt bấm giờ
- Đo lường tối thiểu 50-100 lần, tính toán các phân vị p50, p95, p99
- Đo trên cả Batch 1 (độ trễ robot) và Batch 32 (thông lượng ảnh/giây)
- Đánh giá trên nhiều định dạng (FP32, AMP autocast, FP16 half)
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100, sync: Callable[[], None] | None = None) -> Dict[str, float]:
    """Đo thời gian thực thi của một hàm `fn()` (không tham số), trả về mili-giây (ms).

    `sync` là hàm đồng bộ GPU (ví dụ torch.cuda.synchronize) hoặc None trên môi trường CPU.
    """
    # 1. Warmup: Bỏ qua các lần chạy đầu tiên để ổn định xung nhịp GPU và cache
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo thời gian thực tế qua từng vòng lặp
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()

        fn()

        if sync is not None:
            sync()
        t1 = time.perf_counter()

        times.append((t1 - t0) * 1000.0)  # Quy đổi sang mili-giây (ms)

    times_arr = np.array(times, dtype=np.float64)
    p50, p95, p99 = np.percentile(times_arr, [50, 95, 99])

    return {
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "mean": float(np.mean(times_arr)),
        "min": float(np.min(times_arr)),
        "max": float(np.max(times_arr)),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> Dict[str, Any]:
    """Đo độ trễ lan truyền tiến (forward) của `model` với dữ liệu ngẫu nhiên (batch_size, 3, img_size, img_size).

    Trả về dict có thể ghi trực tiếp vào sheet `Latency` của results.xlsx.
    """
    model.eval()

    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = torch.device(device if use_cuda else "cpu")
    model.to(target_device)

    # Chuẩn bị dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=target_device)

    # Thiết lập kiểu dữ liệu dtype
    dtype_str = dtype.lower()
    sync_fn = torch.cuda.synchronize if use_cuda else None
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    if dtype_str == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()

    @torch.inference_mode()
    def _run_forward():
        if dtype_str == "amp" and use_cuda:
            with torch.autocast(device_type="cuda", enabled=True):
                return model(dummy_input)
        else:
            return model(dummy_input)

    # Chạy đo kiểm qua hàm bench
    res = bench(_run_forward, warmup=warmup, iters=iters, sync=sync_fn)

    # Tính thông lượng (throughput: số ảnh xử lý được trong một giây)
    # throughput = batch_size / (thời gian tính bằng giây)
    p50_sec = max(res["p50"] / 1000.0, 1e-7)
    images_per_s = batch_size / p50_sec

    report = {
        "gpu": gpu_name,
        "dtype": dtype_str,
        "batch": batch_size,
        "img_size": img_size,
        "p50": round(res["p50"], 2),
        "p95": round(res["p95"], 2),
        "p99": round(res["p99"], 2),
        "mean": round(res["mean"], 2),
        "images_per_s": round(images_per_s, 2),
        "torch": torch.__version__,
    }

    print(f"[latency_report] Batch={batch_size}, Dtype={dtype_str} -> p50={report['p50']}ms, "
          f"p95={report['p95']}ms, Thông lượng={report['images_per_s']} ảnh/s")
    return report


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", **kw) -> Dict[str, Any]:
    """Đo độ trễ của phương pháp TTA với K góc nhìn (K-views):

    So sánh giữa thời gian đo thực tế và ước lượng lý thuyết (K * p50 của 1 view).
    """
    # 1. Đo độ trễ cơ sở của 1 view
    base_report = latency_report(model, batch_size=batch_size, img_size=img_size, dtype=dtype, device=device, **kw)
    base_p50 = base_report["p50"]

    # 2. Đo mô phỏng K views (chạy K lần forward cho mỗi batch)
    use_cuda = torch.cuda.is_available() and device.startswith("cuda")
    target_device = torch.device(device if use_cuda else "cpu")
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=target_device)
    sync_fn = torch.cuda.synchronize if use_cuda else None

    @torch.inference_mode()
    def _run_k_views():
        for _ in range(k_views):
            _ = model(dummy)

    tta_res = bench(_run_k_views, warmup=kw.get("warmup", 10), iters=kw.get("iters", 50), sync=sync_fn)

    return {
        "k_views": k_views,
        "1_view_p50_ms": base_p50,
        "tta_p50_ms": round(tta_res["p50"], 2),
        "tta_p95_ms": round(tta_res["p95"], 2),
        "ratio_vs_1view": round(tta_res["p50"] / max(base_p50, 1e-5), 2),
        "expected_k_ratio": k_views,
    }
