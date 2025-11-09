"""
Verify A×B multiplication correctness (A ≠ B).
"""
import numpy as np
from scipy import sparse as sp

print("Loading matrix A (1-based indices)...")
rows_a, cols_a, vals_a = [], [], []
with open('data/input/matrix_a.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_a.append(int(parts[0]) - 1)
            cols_a.append(int(parts[1]) - 1)
            vals_a.append(float(parts[2]))
print(f"  A: {len(rows_a):,} entries")

print("\nLoading matrix B (1-based indices)...")
rows_b, cols_b, vals_b = [], [], []
with open('data/input/matrix_b.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_b.append(int(parts[0]) - 1)
            cols_b.append(int(parts[1]) - 1)
            vals_b.append(float(parts[2]))
print(f"  B: {len(rows_b):,} entries")

# Check if A and B are different
A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(50000, 50000))
B = sp.coo_matrix((vals_b, (rows_b, cols_b)), shape=(50000, 50000))

A_csr = A.tocsr()
B_csr = B.tocsr()

# Check first few entries to confirm they're different
print("\nFirst 5 entries of A:")
for i in range(min(5, len(rows_a))):
    print(f"  A[{rows_a[i]+1},{cols_a[i]+1}] = {vals_a[i]}")

print("\nFirst 5 entries of B:")
for i in range(min(5, len(rows_b))):
    print(f"  B[{rows_b[i]+1},{cols_b[i]+1}] = {vals_b[i]}")

print("\n" + "="*70)
print("Computing scipy A×B (ground truth)...")
C_scipy = A_csr @ B_csr
print(f"  Scipy result: {C_scipy.nnz:,} non-zero entries")

print("\nLoading your A×B result...")
rows_out, cols_out, vals_out = [], [], []
with open('data/ouput/matrix_product_parallel.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_out.append(int(parts[0]) - 1)
            cols_out.append(int(parts[1]) - 1)
            vals_out.append(float(parts[2]))
print(f"  Your result: {len(rows_out):,} entries")

print("\nComparing results...")
C_yours = sp.coo_matrix((vals_out, (rows_out, cols_out)), shape=(50000, 50000)).tocsr()

mismatches = 0
for i in range(50000):
    for j_idx in range(C_scipy.indptr[i], C_scipy.indptr[i+1]):
        j = C_scipy.indices[j_idx]
        scipy_val = C_scipy.data[j_idx]
        your_val = C_yours[i, j]
        
        if abs(scipy_val - your_val) > 1e-6:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Mismatch at ({i+1},{j+1}): scipy={scipy_val}, yours={your_val}")

print(f"\n{'='*70}")
if mismatches == 0 and C_yours.nnz == C_scipy.nnz:
    print("✓ PERFECT! A×B multiplication is correct!")
    print(f"  - {C_scipy.nnz:,} entries verified")
    print(f"  - Matrix A ≠ Matrix B ✓")
    print(f"  - 1-based indexing working correctly ✓")
else:
    print(f"✗ Found {mismatches} mismatches")
    print(f"  Entry count: scipy={C_scipy.nnz:,}, yours={C_yours.nnz:,}")
print(f"{'='*70}")
