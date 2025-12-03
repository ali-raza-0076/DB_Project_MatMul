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

### GNN Dynamic Graph Updates

**Incremental Edge Addition vs Full Recomputation**

| Density | Sparsity | New Edges | Full Recomp | Incremental | Speedup | Winner |
|---------|----------|-----------|-------------|-------------|---------|--------|
| 10% | 90% | 1 | 70.2ms | 7.7ms | **9.1×** | Incremental |
| 10% | 90% | 2 | 70.9ms | 5.9ms | **12.1×** | Incremental |
| 10% | 90% | 3 | 68.7ms | 5.5ms | **12.5×** | Incremental |
| 1% | 99% | 1 | 6.8ms | 1.7ms | **4.0×** | Incremental |
| 1% | 99% | 2 | 7.9ms | 1.6ms | **4.9×** | Incremental |
| 1% | 99% | 3 | 6.8ms | 2.4ms | **2.8×** | Incremental |
| 0.1% | 99.9% | 1 | 0.98ms | 1.66ms | 0.59× | Full Recomp |
| 0.1% | 99.9% | 2 | 0.91ms | 1.57ms | 0.58× | Full Recomp |
| 0.1% | 99.9% | 3 | 1.22ms | 1.36ms | 0.90× | Full Recomp |

**Result**: Incremental updates (LIL→CSR) win at 90-99% sparsity. At extreme sparsity (99.9%), full recomputation is faster due to overhead of format conversion.

### GNN: GPU Dense vs CPU Sparse

| Nodes | Density | Sparsity | Non-Zeros | CPU Sparse | GPU Dense | Speedup | Winner |
|-------|---------|----------|-----------|------------|-----------|---------|--------|
| 1,000 | 10% | 90% | 95,178 | 0.058s | 0.002s | **27.6×** | GPU |
| 1,000 | 1% | 99% | 9,954 | 0.002s | 0.002s | **1.0×** | GPU |
| 1,000 | 0.1% | 99.9% | 1,000 | 0.0002s | 0.002s | 0.15× | **CPU** |
| 2,000 | 10% | 90% | 380,453 | 0.317s | 0.010s | **33.3×** | GPU |
| 2,000 | 1% | 99% | 39,802 | 0.013s | 0.010s | **1.3×** | GPU |

**Result**: GPU dominates at 90-99% sparsity. CPU sparse wins at extreme sparsity (99.9%).

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

### Dynamic Graph Updates
4. **Incremental updates win at moderate sparsity** (90-99%): 3-12× faster
5. **Full recomputation wins at extreme sparsity** (99.9%): format conversion overhead exceeds rebuild cost
6. **Use LIL→CSR for dynamic graphs** with <99% sparsity

### GPU vs CPU
7. **GPU optimal for 90-99% sparsity**: 27-33× speedup for larger graphs
8. **CPU sparse optimal for ≥99.9% sparsity**: minimal computation negates GPU parallelism benefit
9. **Crossover point**: GPU wins with >1,000 non-zeros

### Practical Recommendations
- **Social networks** (90% sparse): Use incremental updates on GPU
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
