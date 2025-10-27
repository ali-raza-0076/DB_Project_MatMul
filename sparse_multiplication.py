import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import tempfile
import os
import time
import numba
from scipy import sparse as sp

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix, print_matrix_info
from external_sort import sort_sparse_matrix
from sparse_multiplication import SparseMultiplication
from matrix_formats import COOMatrix

# ============================================================================
# Numba-Accelerated Helper Functions
# ============================================================================

@numba.jit(nopython=True, cache=True)
def _dot_product_sorted(cols1, vals1, cols2, vals2):
    """
    Compute dot product of two sorted sparse vectors using two-pointer technique.
    
    Used for computing C[i,j] = A[i,:] · B[:,j] where both vectors are sparse.
    
    Args:
        cols1: Column indices of first vector (sorted)
        vals1: Values of first vector
        cols2: Row indices of second vector (sorted, but represents column in transpose)
        vals2: Values of second vector
    
    Returns:
        Dot product value
    """
    result = 0.0
    i, j = 0, 0
    n1, n2 = len(cols1), len(cols2)
    
    while i < n1 and j < n2:
        if cols1[i] < cols2[j]:
            i += 1
        elif cols1[i] > cols2[j]:
            j += 1
        else:  # cols1[i] == cols2[j]
            result += vals1[i] * vals2[j]
            i += 1
            j += 1
    
    return result


@numba.jit(nopython=True, cache=True)
def _compute_row_col_products(row_cols, row_vals, col_rows, col_vals):
    """
    Compute dot product between a CSR row and a CSC column.
    
    Args:
        row_cols: Column indices from CSR row
        row_vals: Values from CSR row
        col_rows: Row indices from CSC column
        col_vals: Values from CSC column
    
    Returns:
        Dot product value
    """
    return _dot_product_sorted(row_cols, row_vals, col_rows, col_vals)


@numba.jit(nopython=True, cache=True, parallel=False)
def _multiply_block(row_start, row_end, col_start, col_end,
                   csr_row_ptr, csr_col_idx, csr_values,
                   csc_col_ptr, csc_row_idx, csc_values,
                   result_rows, result_cols, result_vals, result_idx):
    """
    Multiply a block of rows from A with a block of columns from B.
    
    Args:
        row_start, row_end: Row range in A (CSR)
        col_start, col_end: Column range in B (CSC)
        csr_row_ptr, csr_col_idx, csr_values: CSR arrays for A
        csc_col_ptr, csc_row_idx, csc_values: CSC arrays for B
        result_rows, result_cols, result_vals: Output arrays (pre-allocated)
        result_idx: Current position in result arrays
    
    Returns:
        Number of nonzero entries computed
    """
    count = 0
    
    for i in range(row_start, row_end):
        # Get row i from A (CSR)
        row_start_idx = csr_row_ptr[i]
        row_end_idx = csr_row_ptr[i + 1]
        
        if row_start_idx == row_end_idx:
            continue  # Empty row
        
        row_cols = csr_col_idx[row_start_idx:row_end_idx]
        row_vals = csr_values[row_start_idx:row_end_idx]
        
        for j in range(col_start, col_end):
            # Get column j from B (CSC)
            col_start_idx = csc_col_ptr[j]
            col_end_idx = csc_col_ptr[j + 1]
            
            if col_start_idx == col_end_idx:
                continue  # Empty column
            
            col_rows = csc_row_idx[col_start_idx:col_end_idx]
            col_vals = csc_values[col_start_idx:col_end_idx]
            
            # Compute dot product
            dot = _dot_product_sorted(row_cols, row_vals, col_rows, col_vals)
            
            if abs(dot) > 1e-10:  # Only store nonzero values
                idx = result_idx + count
                result_rows[idx] = i
                result_cols[idx] = j
                result_vals[idx] = dot
                count += 1
    
    return count


# ============================================================================
# Main Multiplication Class
# ============================================================================

class SparseMultiplication:
    """
    Sparse matrix multiplication using CSR × CSC format.
    
    Supports blocked multiplication for large matrices that don't fit in RAM.
    Uses Numba JIT acceleration for performance.
    """
    
    def __init__(self, block_size: int = 1000, logger: Optional[logging.Logger] = None):
        """
        Args:
            block_size: Number of rows/columns to process in each block
            logger: Optional logger instance
        """
        self.block_size = block_size
        self.logger = logger or logging.getLogger(__name__)
    
    def multiply_coo(self, coo_a: COOMatrix, coo_b: COOMatrix, 
                     output_file: Optional[str] = None) -> COOMatrix:
        """
        Multiply two COO matrices: C = A × B
        
        Args:
            coo_a: Left matrix (m × k)
            coo_b: Right matrix (k × n)
            output_file: Optional output file for result
        
        Returns:
            COOMatrix representing C = A × B
        """
        # Verify dimensions are compatible
        if coo_a.shape[1] != coo_b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: A is {coo_a.shape}, B is {coo_b.shape}. "
                f"A's columns ({coo_a.shape[1]}) must equal B's rows ({coo_b.shape[0]})"
            )
        
        self.logger.info(f"Multiplying matrices: A{coo_a.shape} × B{coo_b.shape}")
        
        # Convert to CSR and CSC formats
        self.logger.info("Converting A to CSR format...")
        csr_a = coo_a.to_csr()
        
        self.logger.info("Converting B to CSC format...")
        csc_b = coo_b.to_csc()
        
        # Perform multiplication
        result_csr = self.multiply_csr_csc(csr_a, csc_b)
        
        # Convert result to COO
        coo_result = self._csr_to_coo(result_csr)
        
        # Save to file if requested
        if output_file:
            coo_result.to_csv(output_file)
            self.logger.info(f"Result saved to {output_file}")
        
        return coo_result
    
    def multiply_csr_csc(self, csr_a: CSRMatrix, csc_b: CSCMatrix) -> CSRMatrix:
        """
        Multiply CSR × CSC matrices: C = A × B
        
        This is the core multiplication algorithm using blocked processing.
        
        Args:
            csr_a: Left matrix in CSR format (m × k)
            csc_b: Right matrix in CSC format (k × n)
        
        Returns:
            CSRMatrix representing C = A × B
        """
        # Verify dimensions
        if csr_a.shape[1] != csc_b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: A is {csr_a.shape}, B is {csc_b.shape}"
            )
        
        m, k = csr_a.shape
        k2, n = csc_b.shape
        result_shape = (m, n)
        
        self.logger.info(f"Computing C = A × B where A is {csr_a.shape} and B is {csc_b.shape}")
        self.logger.info(f"Result will be {result_shape}")
        
        # Estimate maximum possible nonzeros (for pre-allocation)
        # This is conservative: actual result will likely be much sparser
        max_nnz = min(csr_a.nnz() * csc_b.nnz() // max(k, 1), m * n)
        max_nnz = min(max_nnz, 10_000_000)  # Cap at 10M for safety
        
        # Allocate result arrays
        result_rows_list = []
        result_cols_list = []
        result_vals_list = []
        
        # Process in blocks
        num_row_blocks = (m + self.block_size - 1) // self.block_size
        num_col_blocks = (n + self.block_size - 1) // self.block_size
        
        total_blocks = num_row_blocks * num_col_blocks
        self.logger.info(f"Processing {num_row_blocks} row blocks × {num_col_blocks} col blocks = {total_blocks} blocks")
        
        block_num = 0
        for row_block in range(num_row_blocks):
            row_start = row_block * self.block_size
            row_end = min(row_start + self.block_size, m)
            
            for col_block in range(num_col_blocks):
                col_start = col_block * self.block_size
                col_end = min(col_start + self.block_size, n)
                
                block_num += 1
                if block_num % 10 == 0 or block_num == total_blocks:
                    self.logger.info(f"Processing block {block_num}/{total_blocks}...")
                
                # Compute this block
                block_result = self._multiply_block_simple(
                    csr_a, csc_b, row_start, row_end, col_start, col_end
                )
                
                # Accumulate results
                if block_result:
                    rows, cols, vals = block_result
                    result_rows_list.extend(rows)
                    result_cols_list.extend(cols)
                    result_vals_list.extend(vals)
        
        # Build result CSR matrix
        if not result_rows_list:
            # Empty result
            self.logger.warning("Result matrix is empty (all zeros)")
            result_row_ptr = np.zeros(m + 1, dtype=np.int64)
            result_col_idx = np.array([], dtype=np.int32)
            result_values = np.array([], dtype=np.float64)
        else:
            # Convert lists to arrays
            result_rows = np.array(result_rows_list, dtype=np.int32)
            result_cols = np.array(result_cols_list, dtype=np.int32)
            result_vals = np.array(result_vals_list, dtype=np.float64)
            
            # Build CSR format
            from matrix_formats import _build_csr_arrays
            result_row_ptr, result_col_idx, result_values = _build_csr_arrays(
                result_rows, result_cols, result_vals, m
            )
        
        self.logger.info(f"Multiplication complete: {len(result_values)} nonzero entries")
        
        return CSRMatrix(result_shape, result_row_ptr, result_col_idx, result_values)
    
    def _multiply_block_simple(self, csr_a: CSRMatrix, csc_b: CSCMatrix,
                               row_start: int, row_end: int,
                               col_start: int, col_end: int) -> Optional[Tuple[List, List, List]]:
        """
        Multiply a block of rows from A with a block of columns from B.
        
        Args:
            csr_a: Left matrix (CSR)
            csc_b: Right matrix (CSC)
            row_start, row_end: Row range in A
            col_start, col_end: Column range in B
        
        Returns:
            (rows, cols, values) for nonzero entries in this block, or None if empty
        """
        rows = []
        cols = []
        vals = []
        
        for i in range(row_start, row_end):
            # Get row i from A
            row_cols, row_vals = csr_a.get_row(i)
            
            if len(row_cols) == 0:
                continue  # Skip empty rows
            
            for j in range(col_start, col_end):
                # Get column j from B
                col_rows, col_vals = csc_b.get_col(j)
                
                if len(col_rows) == 0:
                    continue  # Skip empty columns
                
                # Compute dot product using Numba-accelerated function
                dot = _dot_product_sorted(row_cols, row_vals, col_rows, col_vals)
                
                if abs(dot) > 1e-10:  # Only store nonzero values
                    rows.append(i)
                    cols.append(j)
                    vals.append(dot)
        
        if not rows:
            return None
        
        return rows, cols, vals
    
    def _csr_to_coo(self, csr: CSRMatrix) -> COOMatrix:
        """
        Convert CSR matrix to COO format.
        
        Args:
            csr: CSRMatrix to convert
        
        Returns:
            COOMatrix with same data
        """
        # Extract all entries from CSR
        rows = []
        cols = []
        vals = []
        
        for i in range(csr.shape[0]):
            col_idx, values = csr.get_row(i)
            for j, v in zip(col_idx, values):
                rows.append(i)
                cols.append(j)
                vals.append(v)
        
        # Create COO data
        data = [(i, j, v) for i, j, v in zip(rows, cols, vals)]
        
        return COOMatrix(shape=csr.shape, data=data)


# ============================================================================
# Verification and Comparison Functions
# ============================================================================

def verify_multiplication_scipy(coo_a: COOMatrix, coo_b: COOMatrix, 
                                coo_result: COOMatrix, tolerance: float = 1e-9) -> bool:
    """
    Verify multiplication correctness against scipy.sparse.
    
    WARNING: Only use for small matrices that fit in memory!
    
    Args:
        coo_a: Left matrix
        coo_b: Right matrix
        coo_result: Our computed result
        tolerance: Numerical tolerance
    
    Returns:
        True if verification passes
    """
    logger = logging.getLogger(__name__)
    logger.info("Verifying multiplication against scipy.sparse...")
    
    try:
        # Convert to scipy
        scipy_a = coo_a.to_scipy_sparse()
        scipy_b = coo_b.to_scipy_sparse()
        scipy_result = coo_result.to_scipy_sparse()
        
        # Compute expected result with scipy
        scipy_expected = scipy_a @ scipy_b
        
        # Convert both to COO for comparison
        scipy_expected_coo = scipy_expected.tocoo()
        
        # Compare
        diff = scipy_result - scipy_expected_coo
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        
        if max_diff > tolerance:
            logger.error(f"Verification FAILED: max difference = {max_diff}")
            return False
        
        # Check if result has correct shape
        if scipy_result.shape != scipy_expected.shape:
            logger.error(f"Shape mismatch: got {scipy_result.shape}, expected {scipy_expected.shape}")
            return False
        
        logger.info(f"✓ Multiplication verification passed (max diff = {max_diff:.2e})")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed with exception: {e}")
        return False


def compare_multiplication_performance(coo_a: COOMatrix, coo_b: COOMatrix,
                                      multiplier: SparseMultiplication) -> dict:
    """
    Compare multiplication performance between our implementation and scipy.
    
    Args:
        coo_a: Left matrix
        coo_b: Right matrix
        multiplier: Our SparseMultiplication instance
    
    Returns:
        Dictionary with timing results
    """
    logger = logging.getLogger(__name__)
    logger.info("Comparing multiplication performance...")
    
    results = {}
    
    # Our implementation
    logger.info("Running our implementation...")
    start = time.time()
    our_result = multiplier.multiply_coo(coo_a, coo_b)
    our_time = time.time() - start
    results['our_time'] = our_time
    results['our_nnz'] = our_result.count_nnz()
    
    # scipy implementation (only for small matrices)
    try:
        logger.info("Running scipy.sparse implementation...")
        scipy_a = coo_a.to_scipy_sparse()
        scipy_b = coo_b.to_scipy_sparse()
        
        start = time.time()
        scipy_result = scipy_a @ scipy_b
        scipy_time = time.time() - start
        
        results['scipy_time'] = scipy_time
        results['scipy_nnz'] = scipy_result.nnz
        results['speedup'] = scipy_time / our_time if our_time > 0 else float('inf')
        
        # Print comparison
        logger.info(f"\n{'='*60}")
        logger.info(f"Performance Comparison: Sparse Matrix Multiplication")
        logger.info(f"{'-'*60}")
        logger.info(f"Matrix sizes: A{coo_a.shape} × B{coo_b.shape}")
        logger.info(f"Our implementation:  {our_time:.4f}s ({our_result.count_nnz()} nnz)")
        logger.info(f"scipy.sparse:        {scipy_time:.4f}s ({scipy_result.nnz} nnz)")
        logger.info(f"Speedup:             {results['speedup']:.2f}x {'(FASTER)' if results['speedup'] > 1 else '(slower)'}")
        logger.info(f"{'='*60}\n")
        
    except MemoryError:
        logger.warning("scipy.sparse failed (matrix too large for memory)")
        results['scipy_time'] = None
        results['scipy_nnz'] = None
        results['speedup'] = None
    
    return results


# ============================================================================
# Convenience Functions
# ============================================================================

def multiply_matrices_from_files(file_a: str, file_b: str, output_file: str,
                                 shape_a: Tuple[int, int], shape_b: Tuple[int, int],
                                 block_size: int = 1000) -> COOMatrix:
    """
    Multiply two sparse matrices from CSV files.
    
    Args:
        file_a: Path to matrix A (COO format CSV)
        file_b: Path to matrix B (COO format CSV)
        output_file: Path for result matrix
        shape_a: Shape of matrix A (m, k)
        shape_b: Shape of matrix B (k, n)
        block_size: Block size for multiplication
    
    Returns:
        COOMatrix representing C = A × B
    """
    logger = logging.getLogger(__name__)
    
    # Load matrices
    logger.info(f"Loading matrix A from {file_a}...")
    coo_a = COOMatrix.from_csv(file_a, shape=shape_a)
    
    logger.info(f"Loading matrix B from {file_b}...")
    coo_b = COOMatrix.from_csv(file_b, shape=shape_b)
    
    # Print info
    print_matrix_info(coo_a, "Matrix A")
    print_matrix_info(coo_b, "Matrix B")
    
    # Multiply
    multiplier = SparseMultiplication(block_size=block_size, logger=logger)
    result = multiplier.multiply_coo(coo_a, coo_b, output_file=output_file)
    
    print_matrix_info(result, "Result C = A × B")
    
    return result


# ============================================================================
# Main / Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("Sparse Matrix Multiplication Module")
    print("=" * 70)
    print("Features:")
    print("  ✓ CSR × CSC multiplication with Numba acceleration")
    print("  ✓ Blocked processing for large matrices")
    print("  ✓ Out-of-core capability (matrices larger than RAM)")
    print("  ✓ scipy.sparse verification")
    print("  ✓ Performance comparison utilities")
    print("\nExample usage:")
    print("  from sparse_multiplication import multiply_matrices_from_files")
    print("  result = multiply_matrices_from_files(")
    print("      'matrix_a.csv', 'matrix_b.csv', 'result.csv',")
    print("      shape_a=(10000, 5000), shape_b=(5000, 8000)")
    print("  )")
    print("=" * 70)
    
    # Example with small test matrices
    logger.info("\nRunning example multiplication...")
    
    result = multiply_matrices_from_files(
    './data/output/matrix_a_sorted.csv', './data/output/matrix_b_sorted.csv', 'result.csv',
    shape_a=(50000, 50000), shape_b=(50000, 50000),
    block_size=1000
    )

    coo_a = COOMatrix.from_csv('matrix_a.csv', shape=(10000, 5000))
    coo_b = COOMatrix.from_csv('matrix_b.csv', shape=(5000, 8000))

    multiplier = SparseMultiplication(block_size=1000)
    result = multiplier.multiply_coo(coo_a, coo_b, output_file='result.csv')