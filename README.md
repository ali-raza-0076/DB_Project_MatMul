# Sparse Matrix Operations - Database Systems Project

<div align="center">

**High-Performance Sparse Matrix Operations Using Database Principles**

*External Sorting | Parallel Processing | Compressed Sparse Formats*

</div>

---

## 📋 Overview

This project implements efficient sparse matrix operations using database system principles. It focuses on super sparse matrices (90-99.9% sparsity) with applications in Graph Neural Networks (GNNs), demonstrating significant performance improvements through CSR/CSC formats, parallel processing, and GPU acceleration.

**Key Features:**
- Sparse matrix formats: COO, CSR, CSC
- CPU and GPU benchmarking
- Dynamic graph update optimization
- External sorting for large-scale operations
- Parallel processing with Numba

**Indexing Convention:** CSV files use 1-based indexing; internal operations use 0-based indexing.

---

## 🚀 Quick Start

### Run All Benchmarks

```bash
# CPU: Dense vs Sparse comparison
python dense_baseline_comparison/sparsity_comparison.py

# GNN: Dynamic graph updates (CPU)
python gnn_benchmark_comparison/gnn_benchmark_dynamic.py

# GNN: Dynamic graph updates (GPU) - PRIMARY
python gnn_benchmark_comparison/gnn_benchmark_dynamic_gpu.py

# GNN: GPU vs CPU comparison
python gnn_benchmark_comparison/gnn_benchmark_gpu.py
```

---

## 📊 Benchmark Results

### 🖥️ Test Hardware

<div align="center">

| Component | Specification |
|:---------:|:-------------:|
| **CPU** | AMD Ryzen 9 8940HX (16 cores) |
| **GPU** | NVIDIA RTX 5070 Ti (5888 CUDA cores, 12GB VRAM) |
| **RAM** | 32GB DDR5 |

</div>

---

### 1️⃣ CPU: Dense vs Sparse Performance

<div align="center">

**Matrix Size: 1000×1000**

| Sparsity | Non-Zeros | Sparse Time | Dense Time | **Speedup** | Memory Ratio |
|:--------:|:---------:|:-----------:|:----------:|:-----------:|:------------:|
| **90%** | 95,178 | 0.063s | 1.184s | **18.7×** | 2.6× |
| **99%** | 9,954 | 0.001s | 1.114s | **826×** | 25× |
| **99.9%** | 999 | 0.0002s | 1.209s | **7,591×** | 250× |

**✅ Result:** Sparse CSR×CSC dominates at all super sparse levels. Speedup increases exponentially with sparsity.

</div>

---

### 2️⃣ GNN: Dynamic Graph Updates (GPU - Primary)

<div align="center">

**GPU Accelerated | Early Stopping: 120s**

| Graph Size | Sparsity | Edges | Full Recomp | Incremental | **Speedup** | Winner |
|:----------:|:--------:|:-----:|:-----------:|:-----------:|:-----------:|:------:|
| **500** | 90% | 25,000 | 0.022s | 0.004s | **5.0×** | ✅ Incremental |
| **500** | 99% | 249 | 0.0004s | 0.0003s | **1.4×** | ✅ Incremental |
| **500** | 99.9% | 249 | 0.0003s | 0.0002s | **1.2×** | ✅ Incremental |
| **1000** | 90% | 100,000 | 0.002s | 0.002s | **1.4×** | ✅ Incremental |
| **1000** | 99% | 999 | 0.0007s | 0.0004s | **1.8×** | ✅ Incremental |
| **1000** | 99.9% | 999 | 0.0004s | 0.0003s | **1.3×** | ✅ Incremental |
| **1500** | 90% | 225,000 | 0.002s | 0.0006s | **3.6×** | ✅ Incremental |
| **1500** | 99% | 2,249 | 0.0006s | 0.0002s | **2.6×** | ✅ Incremental |
| **1500** | 99.9% | 2,249 | 0.0006s | 0.0002s | **2.5×** | ✅ Incremental |

**✅ Result:** GPU incremental updates consistently outperform full recomputation across all sparsity levels.

</div>

---

### 3️⃣ GNN: Dynamic Graph Updates (CPU - Alternative)

<div align="center">

**LIL→CSR Format Conversion | Early Stopping: 120s**

| Sparsity | New Edges | Full Recomp | Incremental | **Speedup** | Winner |
|:--------:|:---------:|:-----------:|:-----------:|:-----------:|:------:|
| **90%** | 1 | 0.083s | 0.008s | **10.7×** | ✅ Incremental |
| **90%** | 2 | 0.078s | 0.006s | **12.2×** | ✅ Incremental |
| **90%** | 3 | 0.087s | 0.009s | **9.2×** | ✅ Incremental |
| **99%** | 1 | 0.009s | 0.003s | **3.5×** | ✅ Incremental |
| **99%** | 2 | 0.010s | 0.003s | **3.6×** | ✅ Incremental |
| **99%** | 3 | 0.009s | 0.002s | **3.8×** | ✅ Incremental |
| **99.9%** | 1 | 0.001s | 0.003s | **0.5×** | ⚠️ Full Recomp |
| **99.9%** | 2 | 0.001s | 0.002s | **0.6×** | ⚠️ Full Recomp |
| **99.9%** | 3 | 0.001s | 0.002s | **0.5×** | ⚠️ Full Recomp |

**✅ Result:** CPU incremental updates (LIL→CSR) win at 90-99% sparsity.  
**⚠️ Note:** At extreme sparsity (99.9%), format conversion overhead makes full recomputation faster.

</div>

---

### 4️⃣ GNN: GPU vs CPU Comparison

<div align="center">

**Typical GNN Sparsity (~96-98%)**

| Graph | Nodes | Sparsity | Edges | CPU Sparse | GPU Dense | **Speedup** | Winner |
|:-----:|:-----:|:--------:|:-----:|:----------:|:---------:|:-----------:|:------:|
| **Small** | 500 | 96.08% | 9,799 | 0.0048s | 0.0005s | **9.8×** | 🚀 GPU |
| **Medium** | 1,000 | 98.02% | 19,799 | 0.0087s | 0.0026s | **3.4×** | 🚀 GPU |
| **Large** | 1,500 | 98.02% | 44,537 | 0.0298s | 0.0041s | **7.2×** | 🚀 GPU |

**✅ Result:** GPU wins at all graph sizes with 3-10× speedup for typical GNN sparsity levels.

</div>

---

### 5️⃣ GPU Sparsity Tests

<div align="center">

| Sparsity | GPU Time | Consistency |
|:--------:|:--------:|:-----------:|
| **90%** | 1.98ms ± 0.99ms | ✓ |
| **99%** | 2.04ms ± 0.70ms | ✓ |
| **99.9%** | 2.37ms ± 0.48ms | ✓ |

**✅ Result:** GPU performance remains stable across super sparse levels (~2ms).

</div>

---

## 🎯 Key Findings

### Performance Analysis

| Scenario | Recommendation | Speedup | Optimal Approach |
|:--------:|:--------------|:-------:|:-----------------|
| **Social Networks** (90-98% sparse) | GPU incremental updates | **3-10×** | Matrix-based GPU operations |
| **Citation Graphs** (99% sparse) | CPU incremental updates | **3-4×** | LIL→CSR format conversion |
| **Molecular Structures** (99.9% sparse) | CPU full recomputation | **1.5-2×** | Avoid format conversion overhead |

### Dynamic Graph Updates

✅ **GPU (Primary):** Consistent incremental advantage across all sparsity levels  
✅ **CPU (Alternative):** Incremental wins at 90-99% sparsity  
⚠️ **Threshold:** At 99.9% sparsity, format conversion overhead matters

---

## 📁 Project Structure

```
DB_Project_MatMul/
├── sparse_addition.py              # Sparse matrix addition
├── sparse_addition_parallel.py     # Parallel addition
├── sparse_multiplication.py         # Sparse matrix multiplication
├── sparse_multiplication_parallel.py # Parallel multiplication
├── external_sort.py                # External sorting for large datasets
├── matrix_formats.py               # COO, CSR, CSC conversions
├── generate_data.py                # Test data generation
├── dense_baseline_comparison/      # CPU sparse vs dense benchmarks
├── gnn_benchmark_comparison/       # Graph Neural Network benchmarks
│   ├── gnn_benchmark_dynamic_gpu.py  # GPU dynamic updates (PRIMARY)
│   ├── gnn_benchmark_dynamic.py      # CPU dynamic updates
│   ├── gnn_benchmark_gpu.py          # GPU vs CPU comparison
│   └── generate_graph_data.py        # Graph data generation
└── google_colab_gpu/               # GPU-specific benchmarks
```

---

## 🔧 Dependencies

```bash
pip install numpy scipy numba torch tqdm tabulate
```

**Requirements:**
- Python 3.9+
- NumPy 2.0+
- SciPy 1.16+
- Numba 0.62+
- PyTorch 2.6+ (CUDA 13.0+)
- TQDM, Tabulate

---

## 📖 Documentation

Comprehensive documentation available in `documentation/`:

- **DATA_GENERATION.md** - Test data generation procedures
- **SPARSE_OPERATIONS.md** - Sparse matrix operation details
- **PARALLEL_CPU.md** - Parallel processing implementation
- **VERIFICATION.md** - Correctness verification methods
- **NUMBA_SCIPY_INTEGRATION.md** - Numba optimization techniques

Each benchmark folder contains detailed README with execution instructions and result interpretation.

---

## 🎓 Academic Context

**Database Systems Project**  
Focus: Applying database principles to sparse matrix operations

**Key Concepts:**
- External sorting for out-of-core operations
- Index structures (CSR/CSC similar to database indexes)
- Parallel query processing techniques
- GPU acceleration (hardware-aware optimization)

---

<div align="center">

**Made with ❤️ for High-Performance Computing**

*For questions or contributions, please refer to the documentation folder.*

</div>
