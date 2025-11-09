"""Verify addition correctness with 1-based indexing."""
import numpy as np
from scipy import sparse as sp

print("Loading matrices...")
# Load A
rows_a, cols_a, vals_a = [], [], []
with open('data/ouput/matrix_a_sorted.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_a.append(int(parts[0]) - 1)
            cols_a.append(int(parts[1]) - 1)
            vals_a.append(float(parts[2]))

# Load B
rows_b, cols_b, vals_b = [], [], []
with open('data/ouput/matrix_b_sorted.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_b.append(int(parts[0]) - 1)
            cols_b.append(int(parts[1]) - 1)
            vals_b.append(float(parts[2]))

print(f"A: {len(rows_a):,} entries")
print(f"B: {len(rows_b):,} entries")

A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(50000, 50000))
B = sp.coo_matrix((vals_b, (rows_b, cols_b)), shape=(50000, 50000))

print("\nComputing scipy A+B...")
C_scipy = (A + B).tocsr()
print(f"Scipy: {C_scipy.nnz:,} entries")

print("\nLoading your result...")
rows_out, cols_out, vals_out = [], [], []
with open('data/ouput/matrix_sum_parallel.csv') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 3:
            rows_out.append(int(parts[0]) - 1)
            cols_out.append(int(parts[1]) - 1)
            vals_out.append(float(parts[2]))
print(f"Yours: {len(rows_out):,} entries")

C_yours = sp.coo_matrix((vals_out, (rows_out, cols_out)), shape=(50000, 50000)).tocsr()

print("\nComparing...")
mismatches = 0
for i in range(50000):
    for j_idx in range(C_scipy.indptr[i], C_scipy.indptr[i+1]):
        j = C_scipy.indices[j_idx]
        if abs(C_scipy[i,j] - C_yours[i,j]) > 1e-6:
            mismatches += 1
            if mismatches <= 3:
                print(f"  Mismatch at ({i+1},{j+1})")

print(f"\n{'='*70}")
if mismatches == 0 and C_yours.nnz == C_scipy.nnz:
    print("✓ PERFECT! Addition is correct!")
    print(f"  - {C_scipy.nnz:,} entries verified")
    print(f"  - 1-based indexing ✓")
    print(f"  - No header row ✓")
else:
    print(f"✗ Errors: {mismatches} mismatches")
print(f"{'='*70}")
