# Dense vs Sparse CPU Comparison

## Purpose

Benchmark sparse (CSR×CSC) vs dense (numpy) matrix multiplication at super sparse levels (90-99.9%, ≤10% density) to quantify performance differences.

## Execution

```bash
python sparsity_comparison.py
```

## Test Configuration

- Matrix size: 1000×1000
- Sparsity levels: **90%, 99%, 99.9%** (super sparse: ≤10% density)
- Iterations: 3 runs per test
- Formats: Sparse (SciPy CSR×CSC), Dense (NumPy)

## Results

Results in `benchmarks/`:
- `sparsity_comparison.txt` - Human-readable table
- `sparsity_comparison.json` - Machine-readable data
- `sparsity_comparison.csv` - Spreadsheet format

### Performance Summary

| Sparsity | Non-Zeros | Sparse Time | Dense Time | Speedup | Memory Ratio |
|----------|-----------|-------------|------------|---------|--------------|
| 90% | 95,178 | 0.063s | 1.184s | **18.7×** | 2.6× |
| 99% | 9,954 | 0.001s | 1.114s | **826×** | 25× |
| 99.9% | 999 | 0.0002s | 1.209s | **7,591×** | 250× |

## Analysis

Sparse format provides exponentially increasing advantages at super sparse levels:
- **18× faster** at 90% sparsity
- **826× faster** at 99% sparsity  
- **7,591× faster** at 99.9% sparsity

Memory efficiency scales similarly: sparse format uses 2.6-250× less memory depending on sparsity level.
