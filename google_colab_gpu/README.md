# Google Colab GPU Benchmarks

## Purpose

Run GPU benchmarks on Google Colab to compare with CPU results. Uses **exact same test data and metrics** as CPU benchmarks for fair comparison.

## What to Test

### 1. Dense Baseline Comparison (GPU)
Compare: **Dense GPU** vs **Sparse CPU** vs **Dense CPU**

**Test Parameters** (must match CPU tests):
- Matrix size: 1000×1000
- Sparsity levels: 50%, 90%, 95%, 99%
- Same random seeds (42, 123)
- Integer values (1-10)
- 3 runs per test

### 2. GNN Benchmark (GPU)
Compare: **Dense GPU** vs **Sparse CPU**

**Test Parameters** (must match CPU tests):
- Small: 500 nodes, 20 edges/node
- Medium: 1000 nodes, 20 edges/node
- Large: 1500 nodes, 30 edges/node
- Same random seeds (42, 123, 456)

## Files in This Folder

### Benchmarking (Original):
1. **`gpu_sparsity_comparison.ipynb`** - Colab notebook for dense baseline
2. **`gpu_gnn_benchmark.ipynb`** - Colab notebook for GNN graphs
3. **`data/`** - Copy of test data (uploaded to Colab)
4. **`results/`** - GPU benchmark results (download from Colab)

### **NEW: Block-Based Multiplication for Large Matrices**:
5. **`gpu_block_multiplication.py`** - Block multiplication module (handles matrices larger than GPU memory)
6. **`gpu_block_multiplication.ipynb`** - 🔥 **Main Colab notebook** for large matrix multiplication
7. **`test_block_multiplication.py`** - Test script for verification
8. **`BLOCK_MULTIPLICATION_README.md`** - 📖 **Complete documentation** for block-based approach

## How to Use

### Step 1: Upload to Google Colab
1. Upload notebooks to Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Upload data files from `data/` folder

### Step 2: Run Benchmarks
1. Run `gpu_sparsity_comparison.ipynb` - Tests 50%, 90%, 95%, 99%
2. Run `gpu_gnn_benchmark.ipynb` - Tests 3 graph sizes
3. Download result JSON/CSV files

### Step 3: Merge Results
1. Copy GPU results to `results/` folder
2. Run `merge_results.py` to combine CPU + GPU results
3. Generate final comparison tables

## Expected Outputs

### Sparsity Comparison
```
Results: gpu_sparsity_results.json
Contains: Dense GPU times for each sparsity level
```

### GNN Benchmark
```
Results: gpu_gnn_results.json
Contains: Dense GPU times for each graph size
```

## Metrics to Collect

For each test, record:
- Execution time (seconds)
- Memory usage (MB)
- Speedup vs baseline
- GPU model used (T4, V100, etc.)

## Consistency Requirements

✅ **Same data files** (copy from CPU tests)  
✅ **Same sparsity levels** (50%, 90%, 95%, 99%)  
✅ **Same graph sizes** (500, 1000, 1500 nodes)  
✅ **Same number of runs** (3 runs, average ± std)  
✅ **Same random seeds** (42, 123, 456)  
✅ **Same value ranges** (integers 1-10)

## Final Comparison Table

After merging results, you'll get:

| Sparsity | Sparse CPU | Dense CPU | Dense GPU | Best Method |
|----------|------------|-----------|-----------|-------------|
| 50%      | 0.624s     | 1.898s    | ???       | ???         |
| 90%      | 0.084s     | 1.785s    | ???       | ???         |
| 95%      | 0.025s     | 2.083s    | ???       | ???         |
| 99%      | 0.001s     | 1.881s    | ???       | ???         |

This will answer: **Does sparse CPU beat dense GPU at extreme sparsity?**

---

## 🚀 NEW: Block-Based GPU Multiplication

### Problem: Large Matrices Don't Fit in GPU Memory

- **100,000 × 100,000 matrix** = 80GB RAM needed (if dense)
- **Google Colab GPU**: Only 12-15GB RAM
- **Current approach**: Load entire matrix → **CRASHES** ❌

### Solution: Process in Blocks

Instead of loading the entire matrix:
1. Load small **block** of rows from A (fits in GPU)
2. Load small **block** of columns from B
3. Multiply **on GPU**
4. Save result **chunk to disk**
5. Repeat for all blocks

**Result**:
- ✅ Can multiply matrices of **ANY size** (even 1TB!)
- ✅ Only need GPU memory for small chunks
- ✅ Uses GPU speed without memory crashes

### Quick Start

1. **Upload to Google Colab**: `gpu_block_multiplication.ipynb`
2. **Enable GPU**: Runtime → Change runtime type → GPU
3. **Upload data**: `matrix_a.csv`, `matrix_b.csv` from `data/input/`
4. **Run all cells** - it will automatically:
   - Estimate optimal block size
   - Multiply matrices in chunks
   - Save results to disk
5. **Download results**: `result_matrix.npz`

### Documentation

See **`BLOCK_MULTIPLICATION_README.md`** for:
- Detailed usage instructions
- Performance comparisons
- Memory optimization tips
- Troubleshooting guide
- Theory and examples

### Test Locally (Optional)

Before running on large matrices, test with small data:

```bash
python test_block_multiplication.py
```

This verifies:
- GPU is working correctly
- Block multiplication matches CPU results
- Performance is as expected

---

## 📊 When to Use Each Method

| Matrix Size | Sparsity | Best Method |
|------------|----------|-------------|
| < 10K × 10K | Any | Dense GPU (existing notebooks) |
| < 50K × 50K | > 99% | Sparse CPU (main project) |
| **> 50K × 50K** | **Any** | **🔥 Block GPU (NEW!)** |
| **100K × 100K+** | **Any** | **🔥 Block GPU (NEW!)** |
