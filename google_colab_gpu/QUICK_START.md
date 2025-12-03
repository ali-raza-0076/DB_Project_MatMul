# 🚀 Quick Start Guide - Block GPU Multiplication

## For Google Colab (Most Common)

### Step 1: Setup (2 minutes)
```
1. Go to: https://colab.research.google.com/
2. Upload: gpu_block_multiplication.ipynb
3. Runtime → Change runtime type → GPU
4. Click "Connect"
```

### Step 2: Upload Data (1 minute)
```
1. Run first cell (installs CuPy)
2. Run cell that says "Upload files"
3. Upload: matrix_a.csv and matrix_b.csv
```

### Step 3: Configure (30 seconds)
```python
# Edit these values in the notebook:
MATRIX_SIZE = 50000      # Your matrix dimensions
GPU_MEMORY_GB = 12       # Colab has 12-15GB
```

### Step 4: Run (5-10 minutes)
```
1. Click "Runtime" → "Run all"
2. Watch progress: [1/625], [2/625], ...
3. Wait for completion
```

### Step 5: Download (30 seconds)
```
1. Run last cell
2. Download: result_matrix.npz
3. Also download: metadata.json
```

---

## Expected Output

### During Execution:
```
✓ GPU Available: NVIDIA Tesla T4
✓ Total GPU Memory: 15.0GB
✓ Using: 8.4GB (70%)

Memory Estimation:
============================================================
Matrix size: 50,000 × 50,000
Block size: 2,000 × 2,000
Number of blocks: 25 × 25 = 625
Memory per block: ~0.80GB
============================================================

Loading: matrix_a.csv...
  ✓ 100,000 non-zero entries
  ✓ 99.6000% sparse

Loading: matrix_b.csv...
  ✓ 100,000 non-zero entries
  ✓ 99.6000% sparse

============================================================
Starting Block Multiplication
============================================================
Total blocks: 25 × 25 = 625
Output: ./block_results
============================================================

[1/625] Block (0,0): rows [0:2000], cols [0:2000] → 1,234 nnz, 0.523s ✓
[2/625] Block (0,1): rows [0:2000], cols [2000:4000] → 987 nnz, 0.498s ✓
...
[625/625] Block (24,24): rows [48000:50000], cols [48000:50000] → 456 nnz, 0.512s ✓

============================================================
✓ Complete! Total: 315.2s, Avg: 0.504s/block
✓ Metadata: ./block_results/metadata.json
============================================================

Reconstructing result matrix...
✓ Result: 2,456,789 non-zeros, 99.0153% sparse

✓ Final result saved: ./block_results/result_matrix.npz
  Shape: (50000, 50000)
  Non-zeros: 2,456,789
  File size: 58.32 MB
```

---

## Timing Estimates

| Matrix Size | Sparsity | Blocks | Time | GPU Memory |
|-------------|----------|--------|------|------------|
| 10K × 10K | 99% | 25 | ~30s | <1GB |
| 50K × 50K | 99% | 625 | ~5min | ~0.8GB/block |
| 100K × 100K | 99% | 2,500 | ~20min | ~0.8GB/block |
| 200K × 200K | 99% | 10,000 | ~1.5hr | ~0.8GB/block |

---

## Troubleshooting

### Problem: "No GPU detected"
**Solution**: Runtime → Change runtime type → GPU

### Problem: "CUDA out of memory"
**Solution**: Change this line:
```python
block_size = 1000  # Reduce from 2000
```

### Problem: "CuPy not found"
**Solution**: Run this cell:
```python
!pip install cupy-cuda11x
```

### Problem: "File too large to download"
**Solution**: Result is in sparse format (should be <100MB). If larger, check:
```python
# Use sparse format (default)
result_csr = multiplier.reconstruct_result(metadata_path)
sp.save_npz("result.npz", result_csr)  # Compressed sparse
```

---

## Quick Configuration Guide

### Small matrices (<10K × 10K):
```python
block_size = 1000
# Fast, uses minimal GPU memory
```

### Medium matrices (50K × 50K):
```python
block_size = 2000
# Balanced speed and memory
```

### Large matrices (100K × 100K):
```python
block_size = 2000
# May take 20-30 minutes
```

### Very large matrices (>200K × 200K):
```python
block_size = 2000
# May take hours, but will work!
```

### GPU Memory Issues:
```python
# Conservative (slower but safer)
block_size = 500

# Or adjust safety factor:
multiplier = BlockMatrixMultiplier(
    gpu_memory_gb=12,
    safety_factor=0.5  # Use only 50% of GPU memory
)
```

---

## File Locations

### On Google Colab:
```
/content/
├── matrix_a.csv               (uploaded by you)
├── matrix_b.csv               (uploaded by you)
└── block_results/
    ├── block_0_0.npy
    ├── block_0_1.npy
    ├── ...
    ├── metadata.json          (download this)
    └── result_matrix.npz      (download this)
```

### Your Local Computer:
```
google_colab_gpu/
├── gpu_block_multiplication.ipynb     ← Upload to Colab
├── gpu_block_multiplication.py        ← Reference
├── test_block_multiplication.py       ← Test locally first
├── BLOCK_MULTIPLICATION_README.md     ← Full docs
└── IMPLEMENTATION_SUMMARY.md          ← Overview

data/input/
├── matrix_a.csv                       ← Upload to Colab
└── matrix_b.csv                       ← Upload to Colab
```

---

## Verification

### How to verify results are correct:

1. **Automatic verification** (in notebook):
```python
# Compares first 100×100 block with CPU
Maximum difference: 0.0001
✓ Verification PASSED! Results match.
```

2. **Manual verification**:
```python
# On CPU (for small sample)
import scipy.sparse as sp

A = sp.load_npz("matrix_a.npz")[:100, :100]
B = sp.load_npz("matrix_b.npz")[:100, :100]
expected = A @ B

result = sp.load_npz("result_matrix.npz")[:100, :100]
diff = abs(expected - result).max()
print(f"Max difference: {diff}")  # Should be < 0.001
```

---

## Performance Tips

### Faster:
1. ✅ Increase block size (if memory allows)
2. ✅ Use faster GPU (T4 → V100 → A100)
3. ✅ Process during off-peak hours (Colab gives better GPUs)

### More Memory Efficient:
1. ✅ Decrease block size
2. ✅ Lower safety factor
3. ✅ Use sparse output format

### More Reliable:
1. ✅ Use conservative block size (1000)
2. ✅ Higher safety factor (0.5)
3. ✅ Save blocks incrementally (default)

---

## Comparison with Your Existing Code

| Method | File | Matrix Size Limit | Speed | Complexity |
|--------|------|-------------------|-------|------------|
| Sparse CPU | `sparse_multiplication.py` | ~100K | Medium | Low |
| Sparse Parallel | `sparse_multiplication_parallel.py` | ~100K | Fast | Medium |
| Dense GPU | `gpu_sparsity_comparison.ipynb` | ~10K | Very Fast | Low |
| **Block GPU** | `gpu_block_multiplication.ipynb` | **Unlimited** | **Fast** | **Medium** |

---

## When to Use What

```
Matrix < 10K × 10K?
  └─→ Use: gpu_sparsity_comparison.ipynb (simple, fast)

Matrix < 50K × 50K AND > 99% sparse?
  └─→ Use: sparse_multiplication_parallel.py (CPU, optimized for sparse)

Matrix > 50K × 50K OR < 99% sparse?
  └─→ Use: gpu_block_multiplication.ipynb (handles anything!)

Matrix > 1M × 1M?
  └─→ Use: gpu_block_multiplication.ipynb (only option that scales!)
```

---

## Success Checklist

- [ ] Uploaded notebook to Colab
- [ ] Enabled GPU runtime
- [ ] Uploaded matrix CSV files
- [ ] Adjusted MATRIX_SIZE parameter
- [ ] Ran all cells
- [ ] Saw progress: [1/625], [2/625], ...
- [ ] Completed without errors
- [ ] Downloaded result_matrix.npz
- [ ] Downloaded metadata.json
- [ ] (Optional) Verified results match CPU

---

## What's Next?

After successful run:

1. **Compare with CPU**:
   ```python
   # Run your existing sparse_multiplication_parallel.py
   # Compare times
   ```

2. **Try larger matrices**:
   ```python
   # If 50K works, try 100K
   # If that works, try 200K
   ```

3. **Benchmark**:
   ```python
   # Record times for different sizes
   # Plot: size vs time
   # Compare: GPU vs CPU
   ```

4. **Document**:
   ```
   # Add results to your project report
   # Include: size, sparsity, time, method
   ```

---

## 🎉 You're Ready!

**Everything you need is in place. Just upload and run!**

Questions? Check:
1. `BLOCK_MULTIPLICATION_README.md` (detailed guide)
2. `IMPLEMENTATION_SUMMARY.md` (overview)
3. Notebook comments (inline help)

**Good luck! 🚀**
