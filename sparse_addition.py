"""
Sparse Matrix Addition (A + B)
Implements efficient two-pointer merge algorithm for sorted COO matrices.

Algorithm: Sort-Merge Addition
1. Both matrices must be sorted by (row, col)
2. Use two pointers to traverse both matrices simultaneously
3. Merge entries: sum if (i,j) matches, otherwise keep individual entries

Time Complexity: O(nnz(A) + nnz(B))
Space Complexity: O(nnz(A) + nnz(B))
"""

import numpy as np
import numba
import logging
from pathlib import Path
from typing import Tuple
import time
from scipy import sparse as sp

from matrix_formats import COOMatrix, CSRMatrix
from external_sort import sort_sparse_matrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Numba-Accelerated Two-Pointer Merge
# ============================================================================

@numba.jit(nopython=True, cache=True)
def _merge_sorted_coo(rows1, cols1, vals1, rows2, cols2, vals2):
    """
    Merge two sorted COO matrices using two-pointer technique.
    
    Args:
        rows1, cols1, vals1: First matrix (sorted by row, col)
        rows2, cols2, vals2: Second matrix (sorted by row, col)
    
    Returns:
        (result_rows, result_cols, result_vals)
    """
    n1, n2 = len(rows1), len(rows2)
    
    # Pre-allocate maximum possible size
    max_size = n1 + n2
    result_rows = np.empty(max_size, dtype=np.int32)
    result_cols = np.empty(max_size, dtype=np.int32)
    result_vals = np.empty(max_size, dtype=np.int32)
    
    i, j, k = 0, 0, 0
    
    # Two-pointer merge
    while i < n1 and j < n2:
        if rows1[i] < rows2[j]:
            # Entry only in A
            result_rows[k] = rows1[i]
            result_cols[k] = cols1[i]
            result_vals[k] = vals1[i]
            i += 1
            k += 1
        elif rows1[i] > rows2[j]:
            # Entry only in B
            result_rows[k] = rows2[j]
            result_cols[k] = cols2[j]
            result_vals[k] = vals2[j]
            j += 1
            k += 1
        else:  # rows1[i] == rows2[j]
            if cols1[i] < cols2[j]:
                # Entry only in A
                result_rows[k] = rows1[i]
                result_cols[k] = cols1[i]
                result_vals[k] = vals1[i]
                i += 1
                k += 1
            elif cols1[i] > cols2[j]:
                # Entry only in B
                result_rows[k] = rows2[j]
                result_cols[k] = cols2[j]
                result_vals[k] = vals2[j]
                j += 1
                k += 1
            else:  # Same (row, col) - ADD VALUES
                result_rows[k] = rows1[i]
                result_cols[k] = cols1[i]
                result_vals[k] = vals1[i] + vals2[j]
                i += 1
                j += 1
                k += 1
    
    # Copy remaining from A
    while i < n1:
        result_rows[k] = rows1[i]
        result_cols[k] = cols1[i]
        result_vals[k] = vals1[i]
        i += 1
        k += 1
    
    # Copy remaining from B
    while j < n2:
        result_rows[k] = rows2[j]
        result_cols[k] = cols2[j]
        result_vals[k] = vals2[j]
        j += 1
        k += 1
    
    # Return only filled portion
    return result_rows[:k], result_cols[:k], result_vals[:k]


# ============================================================================
# Main Addition Functions
# ============================================================================

def sparse_add_coo(coo_a: COOMatrix, coo_b: COOMatrix, output_file: str = None) -> COOMatrix:
    """
    Add two sparse matrices in COO format: C = A + B
    Uses Numba-accelerated two-pointer merge.
    
    Args:
        coo_a: First matrix (must be sorted by row, col)
        coo_b: Second matrix (must be sorted by row, col)
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info(f"Sparse addition: A({coo_a.shape}) + B({coo_b.shape})")
    
    # Check dimensions match
    if coo_a.shape != coo_b.shape:
        raise ValueError(f"Matrix dimensions don't match: {coo_a.shape} vs {coo_b.shape}")
    
    # Load data into arrays
    logger.info("Loading matrices into memory...")
    rows_a, cols_a, vals_a = [], [], []
    for chunk in coo_a.iter_entries():
        for i, j, v in chunk:
            rows_a.append(i)
            cols_a.append(j)
            vals_a.append(v)
    
    rows_b, cols_b, vals_b = [], [], []
    for chunk in coo_b.iter_entries():
        for i, j, v in chunk:
            rows_b.append(i)
            cols_b.append(j)
            vals_b.append(v)
    
    # Convert to NumPy arrays
    rows_a = np.array(rows_a, dtype=np.int32)
    cols_a = np.array(cols_a, dtype=np.int32)
    vals_a = np.array(vals_a, dtype=np.int32)
    
    rows_b = np.array(rows_b, dtype=np.int32)
    cols_b = np.array(cols_b, dtype=np.int32)
    vals_b = np.array(vals_b, dtype=np.int32)
    
    logger.info(f"A has {len(rows_a):,} entries, B has {len(rows_b):,} entries")
    
    # Perform Numba-accelerated merge
    logger.info("Running two-pointer merge (Numba-accelerated)...")
    start = time.time()
    result_rows, result_cols, result_vals = _merge_sorted_coo(
        rows_a, cols_a, vals_a,
        rows_b, cols_b, vals_b
    )
    elapsed = time.time() - start
    
    logger.info(f"✓ Addition complete in {elapsed:.4f}s")
    logger.info(f"Result has {len(result_rows):,} entries")
    
    # Create result COO matrix
    result_data = [(int(result_rows[i]), int(result_cols[i]), int(result_vals[i])) 
                   for i in range(len(result_rows))]
    
    result_coo = COOMatrix(shape=coo_a.shape, data=result_data)
    
    # Write to file if requested
    if output_file:
        logger.info(f"Writing result to {output_file}...")
        result_coo.to_csv(output_file)
    
    return result_coo


def sparse_add_csr(csr_a: CSRMatrix, csr_b: CSRMatrix) -> CSRMatrix:
    """
    Add two sparse matrices in CSR format: C = A + B
    Uses row-by-row addition.
    
    Args:
        csr_a: First matrix in CSR format
        csr_b: Second matrix in CSR format
    
    Returns:
        CSRMatrix result
    """
    logger.info(f"Sparse addition (CSR): A({csr_a.shape}) + B({csr_b.shape})")
    
    if csr_a.shape != csr_b.shape:
        raise ValueError(f"Matrix dimensions don't match: {csr_a.shape} vs {csr_b.shape}")
    
    num_rows = csr_a.shape[0]
    
    # Build result row by row
    result_cols = []
    result_vals = []
    result_row_ptr = [0]
    
    for i in range(num_rows):
        cols_a, vals_a = csr_a.get_row(i)
        cols_b, vals_b = csr_b.get_row(i)
        
        # Merge two sorted rows
        merged_cols, merged_vals = _merge_two_rows_numba(cols_a, vals_a, cols_b, vals_b)
        
        result_cols.extend(merged_cols)
        result_vals.extend(merged_vals)
        result_row_ptr.append(len(result_cols))
    
    # Convert to arrays
    result_row_ptr = np.array(result_row_ptr, dtype=np.int64)
    result_cols = np.array(result_cols, dtype=np.int32)
    result_vals = np.array(result_vals, dtype=np.int32)
    
    return CSRMatrix(csr_a.shape, result_row_ptr, result_cols, result_vals)


@numba.jit(nopython=True, cache=True)
def _merge_two_rows_numba(cols1, vals1, cols2, vals2):
    """Merge two sorted row segments."""
    n1, n2 = len(cols1), len(cols2)
    max_size = n1 + n2
    
    result_cols = np.empty(max_size, dtype=np.int32)
    result_vals = np.empty(max_size, dtype=np.int32)
    
    i, j, k = 0, 0, 0
    
    while i < n1 and j < n2:
        if cols1[i] < cols2[j]:
            result_cols[k] = cols1[i]
            result_vals[k] = vals1[i]
            i += 1
            k += 1
        elif cols1[i] > cols2[j]:
            result_cols[k] = cols2[j]
            result_vals[k] = vals2[j]
            j += 1
            k += 1
        else:  # Equal - add values
            result_cols[k] = cols1[i]
            result_vals[k] = vals1[i] + vals2[j]
            i += 1
            j += 1
            k += 1
    
    while i < n1:
        result_cols[k] = cols1[i]
        result_vals[k] = vals1[i]
        i += 1
        k += 1
    
    while j < n2:
        result_cols[k] = cols2[j]
        result_vals[k] = vals2[j]
        j += 1
        k += 1
    
    return result_cols[:k], result_vals[:k]


# ============================================================================
# Verification Against scipy
# ============================================================================

def verify_addition_scipy(coo_a: COOMatrix, coo_b: COOMatrix, result: COOMatrix) -> bool:
    """
    Verify addition result against scipy.sparse.
    
    Args:
        coo_a, coo_b: Input matrices
        result: Our result
    
    Returns:
        True if correct
    """
    logger.info("Verifying result against scipy.sparse...")
    
    try:
        # Convert to scipy
        scipy_a = coo_a.to_scipy_sparse()
        scipy_b = coo_b.to_scipy_sparse()
        scipy_result = result.to_scipy_sparse()
        
        # Compute expected
        expected = scipy_a + scipy_b
        
        # Compare
        diff = scipy_result - expected
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        
        if max_diff > 1e-9:
            logger.error(f"✗ Verification failed: max difference = {max_diff}")
            return False
        
        logger.info("✓ Verification passed! Result matches scipy.sparse")
        return True
        
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return False


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Example usage and testing."""
    logger.info("Sparse Matrix Addition - Demo")
    logger.info("=" * 70)
    
    # Example file paths - CHANGE THESE TO YOUR PATHS
    file_a = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_a_sorted.csv"
    file_b = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_b_sorted.csv"
    output_file = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_sum.csv"
    
    # Matrix dimensions (change to match your data)
    shape = (50000, 50000)
    
    logger.info(f"\nLoading matrices from:")
    logger.info(f"  A: {file_a}")
    logger.info(f"  B: {file_b}")
    
    # Load matrices
    coo_a = COOMatrix.from_csv(file_a, shape=shape)
    coo_b = COOMatrix.from_csv(file_b, shape=shape)
    
    logger.info(f"\nA: {coo_a.count_nnz():,} entries")
    logger.info(f"B: {coo_b.count_nnz():,} entries")
    
    # Perform addition
    logger.info("\n" + "=" * 70)
    result = sparse_add_coo(coo_a, coo_b, output_file=output_file)
    logger.info("=" * 70)
    
    logger.info(f"\nResult: {result.count_nnz():,} entries")
    logger.info(f"Saved to: {output_file}")
    
    # Verify (only for small matrices)
    if result.count_nnz() < 100000:
        verify_addition_scipy(coo_a, coo_b, result)


if __name__ == "__main__":
    main()
