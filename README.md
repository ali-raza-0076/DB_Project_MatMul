# Sparse Matrix Operations - Database Systems Project

## Overview

Implementation of sparse matrix operations using database principles: external sorting, parallel processing, and compressed sparse formats (COO, CSR, CSC). Tests super sparse matrices (90-99.9% sparsity, ≤10% density) with dynamic graph update capabilities.

**Indexing Convention**: CSV files use 1-based indexing; internal operations use 0-based indexing.

---

## Benchmark Execution

Generate all benchmark results:

```bash
# Dense vs Sparse CPU (90%, 99%, 99.9% sparsity)
python dense_baseline_comparison/sparsity_comparison.py

# GNN: CPU sparse vs dense (static graphs)
python gnn_benchmark_comparison/gnn_benchmark.py

# GNN: Dynamic graph updates (1, 2, 3 edge additions)
python gnn_benchmark_comparison/gnn_benchmark_dynamic.py

# GNN: GPU dense vs CPU sparse
python gnn_benchmark_comparison/gnn_benchmark_gpu.py

# GPU sparsity tests
python google_colab_gpu/run_gpu_sparsity_torch.py
python google_colab_gpu/run_gpu_gnn_torch.py
python google_colab_gpu/compare_results.py
```

---

## Project Structure

```
DB_Project_MatMul/
├── sparse_addition.py / sparse_addition_parallel.py
├── sparse_multiplication.py / sparse_multiplication_parallel.py
├── external_sort.py
├── matrix_formats.py
├── generate_data.py
├── dense_baseline_comparison/    # CPU sparse vs dense
├── gnn_benchmark_comparison/     # Graph operations
└── google_colab_gpu/              # GPU benchmarks
```

---

## Benchmark Results

### Test Configuration
- **CPU**: AMD Ryzen 9 8940HX (16 cores)
- **GPU**: NVIDIA RTX 5070 Ti (5888 CUDA cores, 12GB VRAM)
- **Matrix Size**: 1000×1000
- **Sparsity Levels**: 90%, 99%, 99.9% (super sparse: ≤10% density)

### Dense vs Sparse CPU (Super Sparse)

| Sparsity | Non-Zeros | Sparse | Dense | Speedup | Memory Ratio |
|----------|-----------|--------|-------|---------|--------------|
| 90% | 95,178 | 0.063s | 1.184s | **18.7×** | 2.6× |
| 99% | 9,954 | 0.001s | 1.114s | **826×** | 25× |
| 99.9% | 999 | 0.0002s | 1.209s | **7,591×** | 250× |

**Result**: Sparse CSR×CSC wins at all super sparse levels. Speedup increases exponentially with sparsity.

### GNN Dynamic Graph Updates (CPU)

**Incremental Edge Addition vs Full Recomputation**

| Sparsity | New Edges | Full Recomp (CPU) | Incremental (CPU) | Speedup | Winner |
|----------|-----------|-------------------|-------------------|---------|--------|
| 90% | 1 | 68.7ms | 5.7ms | **12.0×** | Incremental |
| 90% | 2 | 73.5ms | 5.8ms | **12.8×** | Incremental |
| 90% | 3 | 73.8ms | 6.4ms | **11.6×** | Incremental |
| 99% | 1 | 8.3ms | 1.8ms | **4.5×** | Incremental |
| 99% | 2 | 8.1ms | 1.7ms | **4.8×** | Incremental |
| 99% | 3 | 8.1ms | 2.9ms | **2.8×** | Incremental |
| 99.9% | 1 | 0.95ms | 1.64ms | 0.58× | Full Recomp |
| 99.9% | 2 | 0.87ms | 1.30ms | 0.67× | Full Recomp |
| 99.9% | 3 | 1.12ms | 1.50ms | 0.75× | Full Recomp |

**Result**: CPU incremental updates (LIL→CSR) win at 90-99% sparsity. At extreme sparsity (99.9%), full recomputation faster due to format conversion overhead.

**Note**: GPU multicore dynamic benchmark script available (`gnn_benchmark_dynamic_gpu.py`) but cannot execute due to PyTorch limitation (RTX 5070 Ti compute capability sm_120 > PyTorch max sm_90).

### GNN: GPU Dense vs CPU Sparse

| Graph | Nodes | Sparsity | Edges | CPU Sparse | GPU Dense | Speedup | Winner |
|-------|-------|----------|-------|------------|-----------|---------|--------|
| Small | 500 | 96.08% | 9,799 | 0.0048s | 0.0005s | **9.8×** | GPU |
| Medium | 1,000 | 98.02% | 19,799 | 0.0087s | 0.0026s | **3.4×** | GPU |
| Large | 1,500 | 98.02% | 44,537 | 0.0298s | 0.0041s | **7.2×** | GPU |

**Result**: GPU wins at all graph sizes (3-10×) for typical GNN sparsity (~96-98%).

### GPU Sparsity Tests

| Sparsity | GPU Time | Consistency |
|----------|----------|-------------|
| 90% | 1.98ms ± 0.99ms | ✓ |
| 99% | 2.04ms ± 0.70ms | ✓ |
| 99.9% | 2.37ms ± 0.48ms | ✓ |

**Result**: GPU performance stable across super sparse levels (~2ms). Sparsity doesn't benefit dense GPU operations.

---

## Key Findings

### Sparsity Analysis
1. **CPU Sparse dominates at all super sparse levels** (90-99.9%)
2. **Speedup increases exponentially** with sparsity: 18× → 826× → 7,591×
3. **Memory advantage grows** with sparsity: 2.6× → 25× → 250×

### Dynamic Graph Updates (CPU)
4. **CPU incremental updates win at moderate sparsity** (90-99%): 3-13× faster than full recomputation
5. **Full recomputation wins at extreme sparsity** (99.9%): format conversion overhead (LIL→CSR) exceeds rebuild cost
6. **Use incremental update method**: LIL format for adding edges, then convert to CSR once

### GPU vs CPU
7. **GPU optimal for typical GNN graphs** (96-98% sparsity): 3-10× speedup over CPU sparse
8. **GPU advantage consistent across graph sizes**: 500-1500 nodes all show GPU wins
9. **PyTorch limitation**: RTX 5070 Ti (sm_120) exceeds PyTorch support (max sm_90), limiting some GPU operations

### Practical Recommendations
- **Social networks** (90-98% sparse): Use incremental updates (CPU) + GPU for matrix ops
- **Citation graphs** (99% sparse): Use incremental updates on CPU sparse
- **Molecular structures** (99.9% sparse): Use full recomputation on CPU sparse

---

## Dependencies

```bash
pip install -r requirements.txt
```

Requirements: numpy, scipy, numba, torch, tqdm, tabulate

---

## Sparse Matrix Formats

- **COO**: (row, col, value) triplets for I/O
- **CSR**: Compressed rows for efficient matrix operations
- **CSC**: Compressed columns for efficient column access

---

## Documentation

Detailed documentation in `documentation/`:
- `DATA_GENERATION.md`
- `SPARSE_OPERATIONS.md`
- `PARALLEL_CPU.md`
- `VERIFICATION.md`
- `NUMBA_SCIPY_INTEGRATION.md`

Each benchmark folder contains a README with execution instructions and result interpretation.
