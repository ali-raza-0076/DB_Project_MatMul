# Dense Baseline Comparison

## Purpose

Compare sparse matrix multiplication (scipy CSR×CSC) vs dense matrix multiplication (numpy) to understand when sparse algorithms provide benefits.

## Setup

**Matrix Size**: 1000×1000  
**Entries**: 1000 each (~0.1% density, 99.9% zeros)  
**Algorithm**: Scipy sparse (CSR×CSC) vs numpy dense (matmul)

## Quick Start

```bash
cd dense_baseline_comparison

# 1. Generate small matrices
python generate_small_data.py

# 2. Run benchmark
python benchmark_comparison.py
```

## What Gets Measured

- **Execution time**: Sparse vs dense multiplication  
- **Memory usage**: Storage requirements for each format  
- **Speedup**: How much faster sparse is for highly sparse data

## Results

For 1000×1000 matrices with 99.9% zeros:

- **Sparse time**: ~0.000115s (115 microseconds)
- **Dense time**: ~0.025s (25 milliseconds)
- **Speedup**: ~215× faster with sparse!
- **Memory**: Sparse uses 500× less memory (31 KB vs 15,625 KB)

## Key Findings

✓ Sparse multiplication is dramatically faster for highly sparse data  
✓ Sparse format uses significantly less memory  
✓ Dense multiplication wastes computation on zeros  
✓ Sparse algorithms essential for real-world sparse data

## Benchmark Output

Results saved to `benchmarks/`:
- `comparison_results.json` - Detailed metrics
- `comparison_results.txt` - Human-readable report

## Note

Small numerical differences (<1e-4) may appear due to duplicate coordinate handling between scipy and numpy. This doesn't affect the performance comparison.

## Use Case

This baseline establishes:
- When sparse algorithms are beneficial (high sparsity)
- Memory/time tradeoffs for different approaches
- Baseline for comparing with GPU implementation later
