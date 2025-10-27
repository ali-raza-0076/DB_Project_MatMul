"""
Dense Matrix Operations - Baseline for Comparison
Implements naive 3-loop algorithms for dense matrices.

Purpose: Provide baseline to show sparse algorithms are faster.

Time Complexity:
- Addition: O(n²) - must touch every entry
- Multiplication: O(n³) - three nested loops

This is much slower than sparse algorithms for sparse matrices!
"""

import numpy as np
import time
import logging
from typing import Tuple

from matrix_formats import COOMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dense Matrix Addition (Naive)
# ============================================================================

def dense_add_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Dense matrix addition using element-wise loop.
    O(n²) complexity - touches every element.
    
    Args:
        A, B: Dense NumPy arrays
    
    Returns:
        C = A + B
    """
    m, n = A.shape
    C = np.zeros((m, n), dtype=A.dtype)
    
    for i in range(m):
        for j in range(n):
            C[i, j] = A[i, j] + B[i, j]
    
    return C


def dense_add_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Dense matrix addition using NumPy (optimized).
    Still O(n²) but much faster than naive loop.
    
    Args:
        A, B: Dense NumPy arrays
    
    Returns:
        C = A + B
    """
    return A + B


# ============================================================================
# Dense Matrix Multiplication (Naive 3-loop)
# ============================================================================

def dense_multiply_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Dense matrix multiplication using 3 nested loops.
    O(n³) complexity - very slow!
    
    This is the textbook algorithm:
    C[i,k] = sum over j of A[i,j] * B[j,k]
    
    Args:
        A: m × n matrix
        B: n × p matrix
    
    Returns:
        C: m × p matrix
    """
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError(f"Incompatible dimensions: {A.shape} × {B.shape}")
    
    C = np.zeros((m, p), dtype=A.dtype)
    
    # Three nested loops - O(n³)
    for i in range(m):
        for k in range(p):
            for j in range(n):
                C[i, k] += A[i, j] * B[j, k]
    
    return C


def dense_multiply_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Dense matrix multiplication using NumPy (highly optimized).
    Still O(n³) but uses BLAS/LAPACK for speed.
    
    Args:
        A: m × n matrix
        B: n × p matrix
    
    Returns:
        C: m × p matrix
    """
    return A @ B


# ============================================================================
# Conversion Helpers
# ============================================================================

def coo_to_dense(coo: COOMatrix) -> np.ndarray:
    """
    Convert sparse COO matrix to dense NumPy array.
    
    WARNING: Only use for small matrices!
    A 10,000 × 10,000 dense matrix needs 800 MB RAM.
    
    Args:
        coo: COOMatrix
    
    Returns:
        Dense NumPy array
    """
    m, n = coo.shape
    dense = np.zeros((m, n), dtype=np.int32)
    
    for chunk in coo.iter_entries():
        for i, j, v in chunk:
            dense[i, j] = v
    
    return dense


def dense_to_coo(dense: np.ndarray) -> COOMatrix:
    """
    Convert dense NumPy array to sparse COO matrix.
    
    Args:
        dense: NumPy array
    
    Returns:
        COOMatrix
    """
    rows, cols = np.nonzero(dense)
    values = dense[rows, cols]
    
    data = [(int(rows[i]), int(cols[i]), int(values[i])) for i in range(len(rows))]
    
    return COOMatrix(shape=dense.shape, data=data)


# ============================================================================
# Benchmarking Functions
# ============================================================================

def benchmark_addition(size: int, sparsity: float):
    """
    Compare dense vs sparse addition performance.
    
    Args:
        size: Matrix dimension (size × size)
        sparsity: Fraction of nonzeros (e.g., 0.01 = 1%)
    """
    logger.info("=" * 70)
    logger.info(f"Addition Benchmark: {size}×{size} matrix, {sparsity*100:.1f}% dense")
    logger.info("=" * 70)
    
    # Generate random matrices
    nnz = int(size * size * sparsity)
    
    np.random.seed(42)
    rows = np.random.randint(0, size, nnz)
    cols = np.random.randint(0, size, nnz)
    vals = np.random.randint(-100, 100, nnz)
    
    # Create sparse matrices
    from matrix_formats import COOMatrix
    data_a = [(rows[i], cols[i], vals[i]) for i in range(nnz)]
    data_b = [(rows[i], cols[i], vals[i] + 1) for i in range(nnz)]  # Slightly different
    
    coo_a = COOMatrix(shape=(size, size), data=data_a)
    coo_b = COOMatrix(shape=(size, size), data=data_b)
    
    # Convert to dense
    logger.info("Converting to dense...")
    dense_a = coo_to_dense(coo_a)
    dense_b = coo_to_dense(coo_b)
    
    # Benchmark dense addition (naive)
    logger.info("\nDense addition (naive 2-loop)...")
    start = time.time()
    dense_result_naive = dense_add_naive(dense_a, dense_b)
    time_dense_naive = time.time() - start
    logger.info(f"  Time: {time_dense_naive:.4f}s")
    
    # Benchmark dense addition (NumPy)
    logger.info("\nDense addition (NumPy optimized)...")
    start = time.time()
    dense_result_numpy = dense_add_numpy(dense_a, dense_b)
    time_dense_numpy = time.time() - start
    logger.info(f"  Time: {time_dense_numpy:.4f}s")
    
    # Benchmark sparse addition
    from sparse_addition import sparse_add_coo
    logger.info("\nSparse addition (our implementation)...")
    start = time.time()
    sparse_result = sparse_add_coo(coo_a, coo_b)
    time_sparse = time.time() - start
    logger.info(f"  Time: {time_sparse:.4f}s")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    logger.info(f"  Dense (naive):    {time_dense_naive:.4f}s")
    logger.info(f"  Dense (NumPy):    {time_dense_numpy:.4f}s")
    logger.info(f"  Sparse (ours):    {time_sparse:.4f}s")
    logger.info(f"  Speedup vs naive: {time_dense_naive/time_sparse:.2f}x")
    logger.info(f"  Speedup vs NumPy: {time_dense_numpy/time_sparse:.2f}x")
    logger.info("=" * 70)


def benchmark_multiplication(size: int, sparsity: float):
    """
    Compare dense vs sparse multiplication performance.
    
    Args:
        size: Matrix dimension (size × size)
        sparsity: Fraction of nonzeros (e.g., 0.01 = 1%)
    """
    logger.info("=" * 70)
    logger.info(f"Multiplication Benchmark: {size}×{size} matrix, {sparsity*100:.1f}% dense")
    logger.info("=" * 70)
    
    # Generate random matrices
    nnz = int(size * size * sparsity)
    
    np.random.seed(42)
    rows = np.random.randint(0, size, nnz)
    cols = np.random.randint(0, size, nnz)
    vals = np.random.randint(-100, 100, nnz)
    
    # Create sparse matrices
    from matrix_formats import COOMatrix
    data_a = [(rows[i], cols[i], vals[i]) for i in range(nnz)]
    data_b = [(cols[i], rows[i], vals[i] + 1) for i in range(nnz)]  # Transpose-ish
    
    coo_a = COOMatrix(shape=(size, size), data=data_a)
    coo_b = COOMatrix(shape=(size, size), data=data_b)
    
    # Convert to dense
    logger.info("Converting to dense...")
    dense_a = coo_to_dense(coo_a)
    dense_b = coo_to_dense(coo_b)
    
    # Benchmark dense multiplication (naive)
    if size <= 500:  # Only for small matrices - O(n³) is very slow!
        logger.info("\nDense multiplication (naive 3-loop)...")
        start = time.time()
        dense_result_naive = dense_multiply_naive(dense_a, dense_b)
        time_dense_naive = time.time() - start
        logger.info(f"  Time: {time_dense_naive:.4f}s")
    else:
        logger.info("\nDense multiplication (naive 3-loop): SKIPPED (too slow)")
        time_dense_naive = None
    
    # Benchmark dense multiplication (NumPy)
    logger.info("\nDense multiplication (NumPy optimized)...")
    start = time.time()
    dense_result_numpy = dense_multiply_numpy(dense_a, dense_b)
    time_dense_numpy = time.time() - start
    logger.info(f"  Time: {time_dense_numpy:.4f}s")
    
    # Benchmark sparse multiplication
    from sparse_multiplication import sparse_multiply_from_coo
    logger.info("\nSparse multiplication (our implementation)...")
    start = time.time()
    sparse_result = sparse_multiply_from_coo(coo_a, coo_b)
    time_sparse = time.time() - start
    logger.info(f"  Time: {time_sparse:.4f}s")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    if time_dense_naive:
        logger.info(f"  Dense (naive):    {time_dense_naive:.4f}s")
    else:
        logger.info(f"  Dense (naive):    SKIPPED")
    logger.info(f"  Dense (NumPy):    {time_dense_numpy:.4f}s")
    logger.info(f"  Sparse (ours):    {time_sparse:.4f}s")
    if time_dense_naive:
        logger.info(f"  Speedup vs naive: {time_dense_naive/time_sparse:.2f}x")
    logger.info(f"  Speedup vs NumPy: {time_dense_numpy/time_sparse:.2f}x")
    logger.info("=" * 70)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run benchmarks."""
    logger.info("\nDense Matrix Baseline - Performance Comparison")
    logger.info("=" * 70)
    logger.info("This compares naive dense algorithms vs our sparse algorithms")
    logger.info("=" * 70)
    
    # Small matrix benchmarks
    logger.info("\n### TEST 1: Small Matrix (500×500, 1% dense) ###\n")
    benchmark_addition(500, 0.01)
    
    logger.info("\n### TEST 2: Small Matrix Multiplication (500×500, 1% dense) ###\n")
    benchmark_multiplication(500, 0.01)
    
    # Medium matrix benchmarks
    logger.info("\n### TEST 3: Medium Matrix (1000×1000, 0.5% dense) ###\n")
    benchmark_addition(1000, 0.005)
    
    logger.info("\n### TEST 4: Medium Matrix Multiplication (1000×1000, 0.5% dense) ###\n")
    benchmark_multiplication(1000, 0.005)
    
    logger.info("\n" + "=" * 70)
    logger.info("Key Takeaways:")
    logger.info("1. Sparse addition is MUCH faster for sparse matrices")
    logger.info("2. Sparse multiplication avoids O(n³) dense computation")
    logger.info("3. Speedup increases as sparsity increases")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
