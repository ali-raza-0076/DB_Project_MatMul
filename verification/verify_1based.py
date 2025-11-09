"""
Verify that 1-based indexing multiplication is correct.
"""
import numpy as np
from scipy import sparse as sp

print("Loading matrix A (1-based indices in file)...")
rows_a, cols_a, vals_a = [], [], []
with open('data/ouput/matrix_a_sorted.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            # Convert from 1-based (file) to 0-based (scipy)
            rows_a.append(int(parts[0]) - 1)
            cols_a.append(int(parts[1]) - 1)
            vals_a.append(float(parts[2]))

print(f"  Loaded {len(rows_a):,} entries")

print("\nComputing scipy A×A (ground truth)...")
A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(50000, 50000))
C_scipy = A.tocsr() @ A.tocsr()
print(f"  Scipy result: {C_scipy.nnz:,} non-zero entries")

print("\nLoading your result (1-based indices in file)...")
rows_out, cols_out, vals_out = [], [], []
with open('data/ouput/matrix_product_parallel.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            # Convert from 1-based (file) to 0-based (scipy)
            rows_out.append(int(parts[0]) - 1)
            cols_out.append(int(parts[1]) - 1)
            vals_out.append(float(parts[2]))

print(f"  Loaded {len(rows_out):,} entries")

print("\nComparing results...")
C_yours = sp.coo_matrix((vals_out, (rows_out, cols_out)), shape=(50000, 50000)).tocsr()

# Check if they match
mismatches = 0
position_errors = 0

# Check all scipy entries exist in your result with correct values
for i in range(50000):
    for j_idx in range(C_scipy.indptr[i], C_scipy.indptr[i+1]):
        j = C_scipy.indices[j_idx]
        scipy_val = C_scipy.data[j_idx]
        your_val = C_yours[i, j]
        
        if abs(scipy_val - your_val) > 1e-6:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Mismatch at ({i+1},{j+1}): scipy={scipy_val}, yours={your_val}")

# Check for extra entries
if C_yours.nnz != C_scipy.nnz:
    print(f"\n⚠ Entry count mismatch: scipy={C_scipy.nnz:,}, yours={C_yours.nnz:,}")

print(f"\n{'='*70}")
if mismatches == 0 and C_yours.nnz == C_scipy.nnz:
    print("✓ PERFECT! All entries match scipy exactly!")
    print(f"  - {C_scipy.nnz:,} entries verified")
    print(f"  - 1-based indexing working correctly")
else:
    print(f"✗ Found {mismatches} value mismatches")
print(f"{'='*70}")
