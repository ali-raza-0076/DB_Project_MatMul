# GPU Benchmark Results

This folder contains the results from GPU benchmarks performed on an **NVIDIA GeForce RTX 5070 Ti Laptop GPU** using PyTorch with CUDA 12.4.

## Files

### JSON Results (Machine-Readable)
- **`gpu_sparsity_results.json`** - Raw sparsity comparison data
- **`gpu_gnn_results.json`** - Raw GNN benchmark data

### TXT Reports (Human-Readable)
- **`gpu_sparsity_results.txt`** - Formatted sparsity comparison results
- **`gpu_gnn_results.txt`** - Formatted GNN benchmark results
- **`COMPARISON_SUMMARY.txt`** - Complete CPU vs GPU analysis and recommendations

## Quick Summary

### Sparsity Benchmark (1000×1000 matrices)
| Sparsity | GPU Time | vs Sparse CPU |
|----------|----------|---------------|
| 50%      | 1.71ms   | **364x faster** |
| 90%      | 1.18ms   | **72x faster** |
| 95%      | 1.26ms   | **20x faster** |
| 99%      | 1.25ms   | **1x (tied)** |

### GNN Benchmark
| Graph Size | Nodes | GPU Time | vs Sparse CPU |
|------------|-------|----------|---------------|
| Small      | 500   | 0.60ms   | **4.7x faster** |
| Medium     | 1000  | 1.70ms   | **2.9x faster** |
| Large      | 1500  | 3.91ms   | **4.9x faster** |

## Key Findings

1. **GPU dominates at < 99% sparsity** with 20-364x speedups
2. **Crossover point at 99% sparsity** where CPU sparse becomes competitive
3. **GNN operations: GPU is 3-5x faster** across all graph sizes
4. **Consistent sub-4ms execution times** for all tested scenarios

## View Full Analysis

See `COMPARISON_SUMMARY.txt` for complete analysis, insights, and recommendations.
