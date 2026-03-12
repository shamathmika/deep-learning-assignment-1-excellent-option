# CMPE 258 — Homework 1 Excellent: CUDA Core vs Tensor Core GEMM Benchmark

Measures and explains the performance difference between a **traditional CUDA-core FP32 GEMM** path and a **modern Tensor Core TF32 GEMM** path using a simple fully connected layer.

## Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shamathmika/deep-learning-assignment-1-excellent-option/blob/main/gemm_benchmark.ipynb)

> **Requirements:** Select **GPU runtime** (Runtime → Change runtime type → T4 GPU) before running.

## What's Inside

The notebook (`gemm_benchmark.ipynb`) benchmarks GEMM (General Matrix Multiply) across two frameworks and two precision modes:

| Framework | FP32 Baseline (CUDA Cores) | TF32 / Tensor Core Path |
|-----------|---------------------------|------------------------|
| **PyTorch** | `nn.Linear` with `allow_tf32=False` | `nn.Linear` with `allow_tf32=True` |
| **cuBLAS (CUDA C++)** | `cublasSgemm` | `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_16F` |

### Benchmark Details

- **Matrix sizes:** 256, 512, 1024, 2048, 4096, 8192 (square M=K=N)
- **Warmup:** 50 iterations
- **Timed:** 200 iterations per configuration
- **Timing:** GPU-side events (`torch.cuda.Event` / `cudaEventElapsedTime`)
- **Target GPU:** NVIDIA T4 (Turing, sm_75, Tensor Cores)

### Generated Plots (saved to `outputs/gemm_benchmark/`)

1. **Latency comparison** — log-log plot of all 4 series
2. **GFLOPS throughput** — with T4 FP32 peak line at 8.1 TFLOPS
3. **Speedup bar chart** — TF32/FP32 ratio per matrix size
4. **Combined roofline overview** — latency + throughput side-by-side

### Analysis Covers

- Why small matrices don't benefit from Tensor Cores
- At what size the Tensor Core advantage kicks in
- PyTorch overhead vs raw cuBLAS
- TF32 precision trade-off (10-bit mantissa vs 23-bit FP32)

## How to Run

1. Click the **Open in Colab** badge above
2. Set runtime to **T4 GPU**
3. Run all cells (Runtime → Run all)
4. Results and plots will be generated inline and saved to `outputs/gemm_benchmark/`
