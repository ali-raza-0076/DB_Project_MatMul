# Data Generation Guide

## Quick Start

Generate random sparse matrices with 1-based indexing:

```bash
# Generate Matrix A (100K entries, 50K×50K)
python generate_data.py --random --rows 50000 --cols 50000 --nnz 100000 -o matrix_a.csv --seed 42

# Generate Matrix B (different seed = different matrix)
python generate_data.py --random --rows 50000 --cols 50000 --nnz 100000 -o matrix_b.csv --seed 123
```

## What It Does

Creates CSV files with sparse matrix entries in format: `row,col,value`

**Features**:
- **1-based indexing**: Rows and columns start at 1 (mathematical convention)
- **Random distribution**: Entries placed randomly across the matrix
- **Sparsity control**: Specify exact number of non-zero entries
- **Memory safety**: Checks memory before generation (won't crash your computer)

## Output Format

```
1,11828,43
2,10442,96
2,11539,-44
...
```

Each line: `row,column,value`
- Rows/columns: 1 to N (1-based)
- Values: Random integers between -100 and 100

## Parameters

- `--rows N`: Number of rows (matrix height)
- `--cols M`: Number of columns (matrix width)  
- `--nnz K`: Number of non-zero entries (K entries total)
- `--seed S`: Random seed for reproducibility
- `-o file.csv`: Output filename

## Matrix Sizes

- **Small** (testing): 1K×1K, 5K entries
- **Medium** (development): 10K×10K, 100K entries
- **Large** (production): 50K×50K, 1M entries

Adjust `--nnz` to control sparsity (lower = more sparse)

## Sorting Generated Data

After generation, sort matrices for use with operations:

```bash
python external_sort.py data/input/matrix_a.csv data/ouput/matrix_a_sorted.csv
python external_sort.py data/input/matrix_b.csv data/ouput/matrix_b_sorted.csv
```

Sorted files are required for addition and multiplication operations.
