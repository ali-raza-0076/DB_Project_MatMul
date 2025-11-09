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

1. **`gpu_sparsity_comparison.ipynb`** - Colab notebook for dense baseline
2. **`gpu_gnn_benchmark.ipynb`** - Colab notebook for GNN graphs
3. **`data/`** - Copy of test data (uploaded to Colab)
4. **`results/`** - GPU benchmark results (download from Colab)

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
