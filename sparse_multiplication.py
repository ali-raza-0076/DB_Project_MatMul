"""
Sparse Matrix Multiplication (A × B)
Implements efficient CSR × CSC algorithm with Numba acceleration.

Algorithm: CSR × CSC Two-Pointer Inner Product
1. Convert A to CSR (efficient row access)
2. Convert B to CSC (efficient column access)
3. For each (i,k): compute dot product of A's row i with B's column k
4. Use two-pointer technique for sorted column/row indices

Time Complexity: O(nnz(A) + nnz(B) + nnz(C))
Best case: O(nnz(A) + nnz(B)) when result is sparse
Worst case: O(nnz(A) × avg_row_density) when result is dense
"""

import numpy as np
import numba
import logging
from pathlib import Path
from typing import Tuple
import time
from scipy import sparse as sp

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Numba-Accelerated Inner Product
# ============================================================================

@numba.jit(nopython=True, cache=True)
def _sparse_dot_product(indices1, vals1, indices2, vals2):
    """
    Compute dot product of two sparse vectors using two-pointer technique.
    
    Args:
        indices1, vals1: First sparse vector (sorted indices)
        indices2, vals2: Second sparse vector (sorted indices)
    
    Returns:
        Dot product value
    """
    result = 0
    i, j = 0, 0
    n1, n2 = len(indices1), len(indices2)
    
    while i < n1 and j < n2:
        if indices1[i] < indices2[j]:
            i += 1
        elif indices1[i] > indices2[j]:
            j += 1
        else:  # Match found
            result += vals1[i] * vals2[j]
            i += 1
            j += 1
    
    return result


@numba.jit(nopython=True, cache=True)
def _multiply_csr_csc_numba(
    a_row_ptr, a_col_idx, a_values,
    b_col_ptr, b_row_idx, b_values,
    num_rows_a, num_cols_b
):
    """
    Multiply CSR matrix A by CSC matrix B using Numba.
    
    Returns arrays for result in COO format.
    """
    # Pre-allocate with conservative estimate
    max_nnz = min(num_rows_a * num_cols_b, 10000000)  # Cap at 10M to avoid huge allocation
    result_rows = np.zeros(max_nnz, dtype=np.int32)
    result_cols = np.zeros(max_nnz, dtype=np.int32)
    result_vals = np.zeros(max_nnz, dtype=np.float64)
    
    count = 0
    
    # For each row of A
    for i in range(num_rows_a):
        # Get row i of A
        a_start = a_row_ptr[i]
        a_end = a_row_ptr[i + 1]
        
        if a_start == a_end:  # Empty row
            continue
        
        a_cols = a_col_idx[a_start:a_end]
        a_vals = a_values[a_start:a_end]
        
        # For each column of B
        for k in range(num_cols_b):
            # Get column k of B
            b_start = b_col_ptr[k]
            b_end = b_col_ptr[k + 1]
            
            if b_start == b_end:  # Empty column
                continue
            
            b_rows = b_row_idx[b_start:b_end]
            b_vals = b_values[b_start:b_end]
            
            # Compute dot product: A[i,:] · B[:,k]
            dot = _sparse_dot_product(a_cols, a_vals, b_rows, b_vals)
            
            if dot != 0:
                if count >= max_nnz:
                    # Need to expand arrays - shouldn't happen often
                    break
                result_rows[count] = i
                result_cols[count] = k
                result_vals[count] = dot
                count += 1
    
    # Trim to actual size
    return result_rows[:count], result_cols[:count], result_vals[:count]


# ============================================================================
# Main Multiplication Functions
# ============================================================================

def sparse_multiply(csr_a: CSRMatrix, csc_b: CSCMatrix, output_file: str = None) -> COOMatrix:
    """
    Multiply sparse matrices: C = A × B
    Uses CSR(A) × CSC(B) algorithm with Numba acceleration.
    
    Args:
        csr_a: Matrix A in CSR format
        csc_b: Matrix B in CSC format
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info(f"Sparse multiplication: A({csr_a.shape}) × B({csc_b.shape})")
    
    # Check dimensions compatible
    if csr_a.shape[1] != csc_b.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: A is {csr_a.shape}, B is {csc_b.shape}. "
            f"A's columns ({csr_a.shape[1]}) must equal B's rows ({csc_b.shape[0]})"
        )
    
    result_shape = (csr_a.shape[0], csc_b.shape[1])
    
    logger.info(f"A: {csr_a.nnz():,} nonzeros")
    logger.info(f"B: {csc_b.nnz():,} nonzeros")
    logger.info(f"Result will be {result_shape[0]} × {result_shape[1]}")
    
    # Perform multiplication
    logger.info("Running CSR×CSC multiplication (Numba-accelerated)...")
    start = time.time()
    
    result_rows, result_cols, result_vals = _multiply_csr_csc_numba(
        csr_a.row_ptr, csr_a.col_idx, csr_a.values,
        csc_b.col_ptr, csc_b.row_idx, csc_b.values,
        csr_a.shape[0], csc_b.shape[1]
    )
    
    elapsed = time.time() - start
    
    logger.info(f"✓ Multiplication complete in {elapsed:.4f}s")
    logger.info(f"Result has {len(result_rows):,} nonzeros")
    
    # Create result COO matrix
    result_data = [(int(result_rows[i]), int(result_cols[i]), int(result_vals[i])) 
                   for i in range(len(result_rows))]
    
    result_coo = COOMatrix(shape=result_shape, data=result_data)
    
    # Write to file if requested
    if output_file:
        logger.info(f"Writing result to {output_file}...")
        result_coo.to_csv(output_file)
    
    return result_coo


def sparse_multiply_from_coo(coo_a: COOMatrix, coo_b: COOMatrix, output_file: str = None) -> COOMatrix:
    """
    Multiply sparse matrices given as COO format.
    Automatically converts to CSR/CSC and performs multiplication.
    
    Args:
        coo_a: Matrix A in COO format
        coo_b: Matrix B in COO format
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info("Converting matrices to CSR/CSC...")
    
    # Convert A to CSR
    # Convert A to CSR
    logger.info("DEBUG: Starting A → CSR conversion...")
    start = time.time()
    csr_a = coo_a.to_csr()
    logger.info("DEBUG: Returned from to_csr()!")  # ← ADD THIS LINE
    elapsed = time.time() - start
    logger.info(f"A → CSR: {elapsed:.4f}s")
    
    # BUILD CSR DIRECTLY INSTEAD
    from matrix_formats import build_csr_from_coo
    csr_a = build_csr_from_coo(coo_a)
    
    logger.info("DEBUG: CSR built successfully")

    # Convert B to CSC
    logger.info("DEBUG: Starting B → CSC conversion...")  # ← ADD THIS
    start = time.time()
    csc_b = coo_b.to_csc()
    logger.info(f"B → CSC: {time.time() - start:.4f}s")
    
    # Multiply
    logger.info("DEBUG: Starting multiplication...")  # ← ADD THIS
    return sparse_multiply(csr_a, csc_b, output_file=output_file)


# ============================================================================
# Blocked Multiplication (for very large matrices)
# ============================================================================

def sparse_multiply_blocked(
    csr_a: CSRMatrix, 
    csc_b: CSCMatrix, 
    block_size: int = 1000,
    output_file: str = None
) -> COOMatrix:
    """
    Blocked sparse multiplication for very large matrices.
    Processes result in blocks to limit memory usage.
    
    Args:
        csr_a: Matrix A in CSR format
        csc_b: Matrix B in CSC format
        block_size: Number of rows to process at once
        output_file: Optional output CSV file
    
    Returns:
        COOMatrix result
    """
    logger.info(f"Blocked multiplication: block_size={block_size}")
    
    if csr_a.shape[1] != csc_b.shape[0]:
        raise ValueError(f"Incompatible dimensions")
    
    result_shape = (csr_a.shape[0], csc_b.shape[1])
    num_blocks = (csr_a.shape[0] + block_size - 1) // block_size
    
    all_rows = []
    all_cols = []
    all_vals = []
    
    # Process in row blocks
    for block_idx in range(num_blocks):
        row_start = block_idx * block_size
        row_end = min(row_start + block_size, csr_a.shape[0])
        
        logger.info(f"Processing block {block_idx + 1}/{num_blocks}: rows {row_start}-{row_end}")
        
        # Get block of A
        a_block = csr_a.get_row_block(row_start, row_end)
        
        # Multiply block with full B
        block_rows, block_cols, block_vals = _multiply_csr_csc_numba(
            a_block.row_ptr, a_block.col_idx, a_block.values,
            csc_b.col_ptr, csc_b.row_idx, csc_b.values,
            a_block.shape[0], csc_b.shape[1]
        )
        
        # Adjust row indices (add block offset)
        for r in block_rows:
            all_rows.append(r + row_start)
        all_cols.extend(block_cols)
        all_vals.extend(block_vals)
    
    logger.info(f"✓ Blocked multiplication complete: {len(all_rows):,} nonzeros")
    
    # Create result
    result_data = [(all_rows[i], all_cols[i], all_vals[i]) for i in range(len(all_rows))]
    result_coo = COOMatrix(shape=result_shape, data=result_data)
    
    if output_file:
        result_coo.to_csv(output_file)
    
    return result_coo


# ============================================================================
# Verification Against scipy
# ============================================================================

def verify_multiplication_scipy(coo_a: COOMatrix, coo_b: COOMatrix, result: COOMatrix) -> bool:
    """
    Verify multiplication result against scipy.sparse.
    
    Args:
        coo_a, coo_b: Input matrices
        result: Our result
    
    Returns:
        True if correct
    """
    logger.info("Verifying result against scipy.sparse...")
    
    try:
        # Convert to scipy
        scipy_a = coo_a.to_scipy_sparse().tocsr()
        scipy_b = coo_b.to_scipy_sparse().tocsc()
        scipy_result = result.to_scipy_sparse()
        
        # Compute expected
        expected = scipy_a @ scipy_b
        
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
    logger.info("Sparse Matrix Multiplication - Demo")
    logger.info("=" * 70)
    
    # Example file paths - CHANGE THESE TO YOUR PATHS
    file_a = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_a_sorted.csv"
    file_b = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_b_sorted.csv"
    output_file = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_product.csv"
    
    # Matrix dimensions (change to match your data)
    # For multiplication: A is m×k, B is k×n
    shape_a = (5000, 8000)  # m × k
    shape_b = (8000, 6000)  # k × n
    
    logger.info(f"\nLoading matrices from:")
    logger.info(f"  A: {file_a}")
    logger.info(f"  B: {file_b}")
    
    # Load matrices
    coo_a = COOMatrix.from_csv(file_a, shape=shape_a)
    coo_b = COOMatrix.from_csv(file_b, shape=shape_b)
    
    logger.info(f"\nA: {coo_a.shape}, {coo_a.count_nnz():,} entries")
    logger.info(f"B: {coo_b.shape}, {coo_b.count_nnz():,} entries")
    
    # Perform multiplication
    logger.info("\n" + "=" * 70)
    result = sparse_multiply_from_coo(coo_a, coo_b, output_file=output_file)
    logger.info("=" * 70)
    
    logger.info(f"\nResult: {result.shape}, {result.count_nnz():,} entries")
    logger.info(f"Saved to: {output_file}")
    
    # Verify (only for small matrices)
    if result.count_nnz() < 100000:
        verify_multiplication_scipy(coo_a, coo_b, result)


if __name__ == "__main__":
    main()
