# Sparse Matrix Operations - Database Project

## Overview

Implements sparse matrix operations (addition, multiplication) using database principles: external sorting, multiprocessing, and memory-efficient algorithms.

**Key Feature**: All CSV files use **1-based indexing** (mathematical standard) while internal Python computations use 0-based (standard practice).

---

## Project Structure

```
DB_Project_MatMul/
├── generate_data.py              # Create random sparse matrices
├── external_sort.py              # Sort large CSV files
├── matrix_formats.py             # Convert COO ↔ CSR ↔ CSC
├── sparse_addition.py            # Sequential A+B
├── sparse_addition_parallel.py   # Parallel A+B (8 cores)
├── sparse_multiplication.py      # Sequential A×B
├── sparse_multiplication_parallel.py  # Parallel A×B (8 cores)
├── parallel_cpu.py               # Parallel benchmarking
├── data/
│   ├── input/                    # Generated matrices (unsorted)
│   └── ouput/                    # Sorted matrices and results
├── verification/                 # Test scripts (scipy validation)
├── documentation/                # README guides
├── dense_baseline_comparison/    # Sparse vs Dense CPU benchmarks
└── gnn_benchmark_comparison/     # Graph Neural Network use cases
```

---

## Quick Start

### 1. Generate Data

Create two 50K×50K sparse matrices with 100K entries each:

```bash
python generate_data.py
```

**Output**: `data/input/matrix_a.csv`, `data/input/matrix_b.csv` (1-based, unsorted)

### 2. Sort Matrices

```bash
python external_sort.py data/input/matrix_a.csv data/ouput/matrix_a_sorted.csv
python external_sort.py data/input/matrix_b.csv data/ouput/matrix_b_sorted.csv
```

### 3. Run Operations

**Addition (A+B)**:
```bash
python sparse_addition_parallel.py
```
- Output: `data/ouput/matrix_sum.csv` (~200K entries, 1-based)

**Multiplication (A×B)**:
```bash
python sparse_multiplication_parallel.py
```
- Output: `data/ouput/matrix_product.csv` (~197K entries, 1-based)

### 4. Verify Correctness

```bash
python verification/verify_addition.py
python verification/verify_multiplication.py
```

Both should report **100% CORRECT**.

---

## Core Concepts

### Sparse Formats

- **COO** (Coordinate): (row, col, value) triplets - used for I/O
- **CSR** (Compressed Sparse Row): Efficient row operations - used for multiplication
- **CSC** (Compressed Sparse Column): Efficient column operations

### 1-Based vs 0-Based Indexing

**External (CSV files)**: Indices 1 to N (mathematical standard)
**Internal (Python/NumPy)**: Indices 0 to N-1 (standard practice)

Conversion happens automatically at I/O boundaries in `matrix_formats.py`.

### Parallelization

- **Addition**: Chunk input files, merge partial results
- **Multiplication**: Chunk rows of A, compute partial products, merge
- **Performance**: ~87-90% parallel efficiency on 8 cores

### External Sorting

Large files sorted via chunk-based merge sort (handles files larger than RAM).

---

## Benchmark Comparisons

### Dense Baseline Comparison

**Location**: `dense_baseline_comparison/`

Compares sparse (CSR×CSC) vs dense (numpy) matrix multiplication at different sparsity levels to answer: **When is sparse actually faster?**

**Tests**: 50%, 90%, 95%, 99% sparsity on 1000×1000 matrices

**Results**:
- 50% sparse: Sparse 3× faster
- 90% sparse: Sparse 21× faster
- 95% sparse: Sparse 82× faster
- 99% sparse: Sparse 1439× faster

**Key Finding**: Sparse wins at ALL sparsity levels, even dense matrices!

**Run it**:
```bash
cd dense_baseline_comparison
python sparsity_comparison.py
```

### GNN Benchmark Comparison

**Location**: `gnn_benchmark_comparison/`

Tests graph adjacency matrices simulating Graph Neural Networks (GNNs) - social networks, knowledge graphs, molecular structures.

**Graph Sizes**: 500, 1000, 1500 nodes (96-98% sparse)

**Results**:
- Small (500 nodes): Sparse 86× faster
- Medium (1000 nodes): Sparse 403× faster
- Large (1500 nodes): Sparse 394× faster

**Key Finding**: GNN use cases dramatically favor sparse representations with 6-13× memory savings.

**Run it**:
```bash
cd gnn_benchmark_comparison
python generate_graph_data.py  # Generate graph data
python gnn_benchmark.py         # Run benchmark
```

**Note**: GPU comparison pending (requires GPU hardware). Current benchmarks show sparse CPU vs dense CPU.

---

## Performance Metrics

**Matrix Size**: 50,000 × 50,000 (2.5 billion possible entries)
**Sparsity**: 100,000 entries each (~0.04% density)

| Operation       | Sequential | Parallel (8 cores) | Speedup |
|-----------------|------------|-------------------|---------|
| Addition        | ~7.2s      | ~0.95s            | 7.6×    |
| Multiplication  | ~22.4s     | ~2.9s             | 7.7×    |

---

## Documentation

- **DATA_GENERATION.md**: How to generate matrices with different sizes/sparsity
- **SPARSE_OPERATIONS.md**: Addition, multiplication, format conversions
- **PARALLEL_CPU.md**: Parallelization strategies and performance
- **VERIFICATION.md**: How test scripts work
- **NUMBA_SCIPY_INTEGRATION.md**: Technical details on Numba JIT compilation

### Benchmark Folders

- **dense_baseline_comparison/README.md**: Sparse vs Dense CPU comparison details
- **gnn_benchmark_comparison/README.md**: Graph Neural Network use case details

---

## Dependencies

```
numpy
scipy
numba
tabulate
```

Install: `pip install -r requirements.txt`

---

## Validation

All operations verified against **scipy** (industry-standard library):
- ✓ Addition: 199,996 entries, 100% match
- ✓ Multiplication: 197,421 entries, 100% match

---

## Notes

- Matrices A and B are generated with different random seeds (42 vs 123)
- Duplicate (row, col) entries are automatically merged (values summed)
- All output files use 1-based indexing per academic requirements
- No header rows in operation result files (pure triplets)
