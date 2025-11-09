# Sparse Matrix Operations

## Sparse Addition

**File**: `sparse_addition.py`

**What it does**: Adds two sparse matrices (A + B) using a two-pointer merge algorithm.

**How it works**:
1. Reads two sorted CSV files (row, col, value format)
2. Merges entries using Numba-accelerated algorithm
3. Sums values when (row, col) positions match
4. Keeps unique entries from both matrices
5. Writes sorted result to CSV

**Input**: Two CSV files with 1-based indices (row, col, value)
**Output**: CSV file with A+B result (1-based indices, no header)

---

## Sparse Multiplication

**File**: `sparse_multiplication.py`

**What it does**: Multiplies two sparse matrices (A × B) using CSR × CSC format.

**How it works**:
1. Converts matrix A to CSR (Compressed Sparse Row) format
2. Converts matrix B to CSC (Compressed Sparse Column) format
3. For each row i in A, computes dot product with each column j in B
4. Uses two-pointer algorithm for sparse dot products (Numba-accelerated)
5. Writes result to CSV

**Input**: Two CSV files with 1-based indices
**Output**: CSV file with A×B result (1-based indices, no header)

**Key optimization**: CSR × CSC is fastest format combination for matrix multiplication

---

## Format Conversions

**File**: `matrix_formats.py`

**What it does**: Converts between COO, CSR, and CSC sparse matrix formats.

**Formats**:
- **COO** (Coordinate): List of (row, col, value) triples - used for I/O
- **CSR** (Compressed Sparse Row): Efficient for row operations - used for matrix A
- **CSC** (Compressed Sparse Column): Efficient for column operations - used for matrix B

**Key functions**:
- `build_csr_from_coo()`: Converts COO → CSR
- `build_csc_from_coo()`: Converts COO → CSC (uses external sort)
- Merges duplicate entries automatically
- Handles 1-based ↔ 0-based index conversion at I/O boundaries

---

## External Sort

**File**: `external_sort.py`

**What it does**: Sorts large CSV files that don't fit in memory.

**How it works**:
1. Splits input into smaller chunks (default 50 MB)
2. Sorts each chunk in memory
3. Merges sorted chunks using k-way merge
4. Writes final sorted output

**Usage**: Used internally by `matrix_formats.py` for CSC conversion (sorts by column instead of row)
