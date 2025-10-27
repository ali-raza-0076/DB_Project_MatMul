# Data Generation Guide

## Quick Start

### Generate All Preset Matrices (Recommended)

```bash
python generate_data.py --preset
```

This generates test matrices of various sizes:
- **Small** (100×100, 500 nnz) - for quick tests
- **Medium** (1,000×1,000, 10K nnz) - for algorithm testing
- **Large** (10,000×10,000, 100K nnz) - for performance testing
- **XLarge** (50,000×50,000, 1M nnz) - for stress testing
- **Banded** and **Block-sparse** patterns

All files saved to `data/input/` by default.

---

## Custom Generation

### 1. Random Matrix (1 Million Entries)

```bash
python generate_data.py --random \
  --rows 50000 \
  --cols 50000 \
  --nnz 1000000 \
  -o my_matrix_1M.csv
```

**Memory estimate:** ~25 MB file, ~16 MB RAM during generation

### 2. Smaller Random Matrix (Safe for Any Computer)

```bash
python generate_data.py --random \
  --rows 10000 \
  --cols 10000 \
  --nnz 50000 \
  -o safe_matrix.csv
```

**Memory estimate:** ~1.25 MB file, ~0.8 MB RAM

### 3. Very Large Matrix (10 Million Entries)

```bash
python generate_data.py --random \
  --rows 100000 \
  --cols 100000 \
  --nnz 10000000 \
  -o huge_matrix_10M.csv
```

**Memory estimate:** ~250 MB file, ~160 MB RAM

⚠️ **Safety limit:** Default max is 500 MB. Override with `--max-memory 1000`

---

## Special Patterns

### Banded Matrix (Diagonal Structure)

```bash
python generate_data.py --banded \
  --size 5000 \
  --bandwidth 10 \
  -o banded_matrix.csv
```

Use for: Differential equations, tridiagonal systems

### Block-Sparse Matrix

```bash
python generate_data.py --block \
  --num-blocks 20 \
  --block-size 100 \
  --density 0.1 \
  -o block_sparse.csv
```

Use for: Graph partitioning, neural networks

### Power-Law Distribution (Social Networks)

```bash
python generate_data.py --power-law \
  --rows 10000 \
  --cols 10000 \
  --nnz 100000 \
  --alpha 2.5 \
  -o powerlaw.csv
```

Use for: Web graphs, citation networks

---

## Size Reference

| nnz | Rows × Cols | File Size | RAM Usage | Generation Time |
|-----|-------------|-----------|-----------|-----------------|
| 1,000 | 100×100 | ~25 KB | <1 MB | <1 sec |
| 10,000 | 1K×1K | ~250 KB | ~1 MB | 1 sec |
| 100,000 | 10K×10K | ~2.5 MB | ~10 MB | 5 sec |
| 1,000,000 | 50K×50K | ~25 MB | ~100 MB | 30 sec |
| 10,000,000 | 100K×100K | ~250 MB | ~500 MB | 5 min |

---

## Safety Features

### Automatic Memory Check

The script estimates memory before generation:
```
Memory estimate: 42.5 MB (SAFE)
```

If too large:
```
ERROR: Matrix too large! Estimated memory: 750 MB
Maximum allowed: 500 MB
Suggestion: Reduce nnz to 666666
```

### Override Safety Limit

```bash
python generate_data.py --random \
  --rows 100000 --cols 100000 --nnz 10000000 \
  --max-memory 1000 \
  -o huge.csv
```

---

## Output Format

All matrices are saved as CSV in COO format:

```
row,col,value
0,5,1.234567
0,12,-0.567890
1,3,2.345678
...
```

- One entry per line
- Row and column indices (0-based)
- Float values (6 decimal places)

---

## Example Workflow

### 1. Generate Test Data

```bash
# Create directory structure
mkdir -p data/input data/output data/temp

# Generate preset matrices
python generate_data.py --preset
```

### 2. Generate Custom Matrix Pair for Testing

```bash
# Matrix A (1M entries)
python generate_data.py --random \
  --rows 50000 --cols 50000 --nnz 1000000 \
  --seed 42 \
  -o matrix_A_1M.csv

# Matrix B (1M entries, same seed for reproducibility)
python generate_data.py --random \
  --rows 50000 --cols 50000 --nnz 1000000 \
  --seed 43 \
  -o matrix_B_1M.csv
```

### 3. List Generated Files

```bash
ls -lh data/input/
```

Output:
```
-rw-r--r-- 1 user user  12K Oct 27 matrix_A_1M.csv
-rw-r--r-- 1 user user  12K Oct 27 matrix_B_1M.csv
-rw-r--r-- 1 user user 250K Oct 27 large_A.csv
-rw-r--r-- 1 user user 250K Oct 27 large_B.csv
...
```

---

## Tips

### For Quick Testing
Use **small** or **medium** matrices (10-100K nnz)

### For Algorithm Validation
Use **preset** matrices with known patterns

### For Performance Benchmarking
Use **large** or **xlarge** (100K-1M nnz)

### For Stress Testing
Generate 10M+ entries (requires 1+ GB memory limit)

### For Reproducibility
Always use the same `--seed` value

---

## Troubleshooting

### "Matrix too large" Error
**Solution:** Reduce `--nnz` or increase `--max-memory`

### Slow Generation
**Progress bar shows speed:**
```
Writing entries: 100%|████████| 1000000/1000000 [00:32<00:00, 30891 entries/s]
```

### Out of Disk Space
**Check available space:**
```bash
df -h data/input/
```

**Estimate needed:** ~25 bytes per nnz

---

## Advanced: Generate Matrix Pairs

For testing addition/multiplication, generate compatible matrices:

### Addition Test (A + B)
```bash
# Both must be same size
python generate_data.py --random --rows 10000 --cols 10000 --nnz 50000 -o A.csv
python generate_data.py --random --rows 10000 --cols 10000 --nnz 50000 -o B.csv
```

### Multiplication Test (A × B)
```bash
# A is m×k, B is k×n
python generate_data.py --random --rows 5000 --cols 10000 --nnz 50000 -o A.csv
python generate_data.py --random --rows 10000 --cols 5000 --nnz 50000 -o B.csv
```

---

## Summary

**Safe sizes for most computers:**
- 100K entries: Always safe
- 1M entries: Safe (25 MB file)
- 10M entries: Use `--max-memory 1000` (250 MB file)

**Start with presets:**
```bash
python generate_data.py --preset
```

**Then customize as needed!**
