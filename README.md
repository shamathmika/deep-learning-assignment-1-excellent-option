# CMPE 258 — Homework 1 Excellent: CUDA Core vs Tensor Core GEMM Benchmark

Benchmarks FP32 GEMM (CUDA cores) vs Tensor Core GEMM on an NVIDIA T4, using both PyTorch and raw cuBLAS.

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/deep-learning-assignment-1-excellent-option/blob/main/gemm_benchmark.ipynb)

> Requires **GPU runtime** (Runtime → Change runtime type → T4 GPU).

## Overview

The notebook benchmarks GEMM across two frameworks and two precision modes:

| Framework | FP32 Baseline (CUDA Cores) | Tensor Core Path |
|-----------|---------------------------|-----------------|
| PyTorch | `nn.Linear` with `allow_tf32=False` | `nn.Linear` with `allow_tf32=True` |
| cuBLAS (CUDA C++) | `cublasSgemm` | `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16F` |

**Setup:** Square matrices (M=K=N) at sizes 256–8192, 50 warmup + 200 timed iterations, GPU-side event timing, on T4 (sm_75).

### T4 and TF32

The T4 doesn't support TF32 natively (that's Ampere sm_80+). On T4:
- PyTorch's `allow_tf32` gives inconsistent speedup. i.e, it helps at mid-sizes (1.21x at 1024) but is slower at extremes (0.87x at 8192)
- cuBLAS with `CUBLAS_COMPUTE_32F_FAST_16F` uses FP16 Tensor Cores (FP32 in → FP16 multiply → FP32 accumulate) and gives consistent 1.52-2.81x speedup

## Results

### Latency Comparison
![Latency Comparison](outputs/gemm_benchmark/1_latency_comparison.png)

### GFLOPS Throughput
![GFLOPS Throughput](outputs/gemm_benchmark/2_gflops_throughput.png)

### Speedup (Tensor Core / FP32)
![Speedup](outputs/gemm_benchmark/3_speedup_barchart.png)

## To Run

1. Click the Colab badge above
2. Set runtime to T4 GPU
3. Run all cells
4. Plots are saved to `outputs/gemm_benchmark/`
