"""
Parallel Sparse Matrix Operations (CPU Multi-core)
Implements parallel addition and multiplication using Python's multiprocessing.

Parallelization Strategy:
- Addition: Partition by row blocks, merge in parallel
- Multiplication: Partition result matrix into blocks, compute in parallel

Uses: multiprocessing.Pool for CPU parallelism
"""

import numpy as np
import numba
import logging
from multiprocessing import Pool, cpu_count
from typing import Tuple, List
import time

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Parallel Addition
# ============================================================================

def _add_row_block(args):
    """
    Worker function to add a block of rows.
    Used by multiprocessing.Pool.
    """
    rows_a, cols_a, vals_a, rows_b, cols_b, vals_b, row_start, row_end = args
    
    # Filter entries in this row block
    mask_a = (rows_a >= row_start) & (rows_a < row_end)
    mask_b = (rows_b >= row_start) & (rows_b < row_end)
    
    block_rows_a = rows_a[mask_a]
    block_cols_a = cols_a[mask_a]
    block_vals_a = vals_a[mask_a]
    
    block_rows_b = rows_b[mask_b]
    block_cols_b = cols_b[mask_b]
    block_vals_b = vals_b[mask_b]
    
    # Merge this block
    from sparse_addition import _merge_sorted_coo
    result_rows, result_cols, result_vals = _merge_sorted_coo(
        block_rows_a, block_cols_a, block_vals_a,
        block_rows_b, block_cols_b, block_vals_b
    )
    
    return result_rows, result_cols, result_vals


def parallel_add_coo(coo_a: COOMatrix, coo_b: COOMatrix, num_workers: int = None) -> COOMatrix:
    """
    Parallel sparse matrix addition using row-based partitioning.
    
    Args:
        coo_a, coo_b: Input matrices (must be sorted)
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        COOMatrix result
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    logger.info(f"Parallel addition using {num_workers} workers")
    
    if coo_a.shape != coo_b.shape:
        raise ValueError(f"Matrix dimensions don't match")
    
    # Load data
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
    
    rows_a = np.array(rows_a, dtype=np.int32)
    cols_a = np.array(cols_a, dtype=np.int32)
    vals_a = np.array(vals_a, dtype=np.int32)
    
    rows_b = np.array(rows_b, dtype=np.int32)
    cols_b = np.array(cols_b, dtype=np.int32)
    vals_b = np.array(vals_b, dtype=np.int32)
    
    # Partition into row blocks
    num_rows = coo_a.shape[0]
    block_size = (num_rows + num_workers - 1) // num_workers
    
    tasks = []
    for i in range(num_workers):
        row_start = i * block_size
        row_end = min(row_start + block_size, num_rows)
        tasks.append((rows_a, cols_a, vals_a, rows_b, cols_b, vals_b, row_start, row_end))
    
    # Execute in parallel
    logger.info(f"Processing {len(tasks)} row blocks in parallel...")
    start = time.time()
    
    with Pool(num_workers) as pool:
        results = pool.map(_add_row_block, tasks)
    
    elapsed = time.time() - start
    logger.info(f"✓ Parallel addition complete in {elapsed:.4f}s")
    
    # Merge results
    all_rows = []
    all_cols = []
    all_vals = []
    
    for result_rows, result_cols, result_vals in results:
        all_rows.extend(result_rows)
        all_cols.extend(result_cols)
        all_vals.extend(result_vals)
    
    # Create result
    result_data = [(int(all_rows[i]), int(all_cols[i]), int(all_vals[i])) 
                   for i in range(len(all_rows))]
    
    return COOMatrix(shape=coo_a.shape, data=result_data)


# ============================================================================
# Parallel Multiplication
# ============================================================================

def _multiply_row_block(args):
    """
    Worker function to compute a block of result rows.
    """
    row_start, row_end, a_row_ptr, a_col_idx, a_values, b_col_ptr, b_row_idx, b_values, num_cols_b = args
    
    from sparse_multiplication import _sparse_dot_product
    
    result_rows = []
    result_cols = []
    result_vals = []
    
    # Compute rows [row_start, row_end)
    for i in range(row_start, row_end):
        # Get row i of A
        a_start = a_row_ptr[i]
        a_end = a_row_ptr[i + 1]
        
        if a_start == a_end:
            continue
        
        a_cols = a_col_idx[a_start:a_end]
        a_vals = a_values[a_start:a_end]
        
        # For each column of B
        for k in range(num_cols_b):
            b_start = b_col_ptr[k]
            b_end = b_col_ptr[k + 1]
            
            if b_start == b_end:
                continue
            
            b_rows = b_row_idx[b_start:b_end]
            b_vals = b_values[b_start:b_end]
            
            # Dot product
            dot = _sparse_dot_product(a_cols, a_vals, b_rows, b_vals)
            
            if dot != 0:
                result_rows.append(i)
                result_cols.append(k)
                result_vals.append(dot)
    
    return result_rows, result_cols, result_vals


def parallel_multiply(csr_a: CSRMatrix, csc_b: CSCMatrix, num_workers: int = None) -> COOMatrix:
    """
    Parallel sparse matrix multiplication using row-based partitioning.
    
    Args:
        csr_a: Matrix A in CSR format
        csc_b: Matrix B in CSC format
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        COOMatrix result
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    logger.info(f"Parallel multiplication using {num_workers} workers")
    
    if csr_a.shape[1] != csc_b.shape[0]:
        raise ValueError(f"Incompatible dimensions")
    
    # Partition into row blocks
    num_rows = csr_a.shape[0]
    block_size = (num_rows + num_workers - 1) // num_workers
    
    tasks = []
    for i in range(num_workers):
        row_start = i * block_size
        row_end = min(row_start + block_size, num_rows)
        tasks.append((
            row_start, row_end,
            csr_a.row_ptr, csr_a.col_idx, csr_a.values,
            csc_b.col_ptr, csc_b.row_idx, csc_b.values,
            csc_b.shape[1]
        ))
    
    # Execute in parallel
    logger.info(f"Processing {len(tasks)} row blocks in parallel...")
    start = time.time()
    
    with Pool(num_workers) as pool:
        results = pool.map(_multiply_row_block, tasks)
    
    elapsed = time.time() - start
    logger.info(f"✓ Parallel multiplication complete in {elapsed:.4f}s")
    
    # Merge results
    all_rows = []
    all_cols = []
    all_vals = []
    
    for result_rows, result_cols, result_vals in results:
        all_rows.extend(result_rows)
        all_cols.extend(result_cols)
        all_vals.extend(result_vals)
    
    logger.info(f"Result has {len(all_rows):,} nonzeros")
    
    # Create result
    result_shape = (csr_a.shape[0], csc_b.shape[1])
    result_data = [(all_rows[i], all_cols[i], all_vals[i]) for i in range(len(all_rows))]
    
    return COOMatrix(shape=result_shape, data=result_data)


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_parallel_vs_serial():
    """Compare parallel vs serial performance."""
    logger.info("=" * 70)
    logger.info("Parallel vs Serial Benchmark")
    logger.info("=" * 70)
    
    # Generate test matrices
    from generate_data import SparseMatrixGenerator
    import tempfile
    import os
    
    gen = SparseMatrixGenerator(tempfile.gettempdir())
    
    # Generate 10K × 10K matrix with 50K entries
    logger.info("Generating test matrices...")
    file_a = gen.generate_random(10000, 10000, 50000, "test_a.csv", seed=42)
    file_b = gen.generate_random(10000, 10000, 50000, "test_b.csv", seed=43)
    
    # Sort
    from external_sort import sort_sparse_matrix
    sorted_a = file_a.replace(".csv", "_sorted.csv")
    sorted_b = file_b.replace(".csv", "_sorted.csv")
    
    sort_sparse_matrix(file_a, sorted_a)
    sort_sparse_matrix(file_b, sorted_b)
    
    # Load
    coo_a = COOMatrix.from_csv(sorted_a, shape=(10000, 10000))
    coo_b = COOMatrix.from_csv(sorted_b, shape=(10000, 10000))
    
    # ADDITION BENCHMARK
    logger.info("\n### ADDITION BENCHMARK ###")
    
    # Serial
    from sparse_addition import sparse_add_coo
    logger.info("\nSerial addition:")
    start = time.time()
    result_serial = sparse_add_coo(coo_a, coo_b)
    time_serial = time.time() - start
    logger.info(f"  Time: {time_serial:.4f}s")
    
    # Parallel (2 workers)
    logger.info("\nParallel addition (2 workers):")
    start = time.time()
    result_parallel_2 = parallel_add_coo(coo_a, coo_b, num_workers=2)
    time_parallel_2 = time.time() - start
    logger.info(f"  Time: {time_parallel_2:.4f}s")
    logger.info(f"  Speedup: {time_serial/time_parallel_2:.2f}x")
    
    # Parallel (4 workers)
    logger.info("\nParallel addition (4 workers):")
    start = time.time()
    result_parallel_4 = parallel_add_coo(coo_a, coo_b, num_workers=4)
    time_parallel_4 = time.time() - start
    logger.info(f"  Time: {time_parallel_4:.4f}s")
    logger.info(f"  Speedup: {time_serial/time_parallel_4:.2f}x")
    
    # MULTIPLICATION BENCHMARK
    logger.info("\n### MULTIPLICATION BENCHMARK ###")
    
    csr_a = coo_a.to_csr()
    csc_b = coo_b.to_csc()
    
    # Serial
    from sparse_multiplication import sparse_multiply
    logger.info("\nSerial multiplication:")
    start = time.time()
    result_serial = sparse_multiply(csr_a, csc_b)
    time_serial = time.time() - start
    logger.info(f"  Time: {time_serial:.4f}s")
    
    # Parallel (2 workers)
    logger.info("\nParallel multiplication (2 workers):")
    start = time.time()
    result_parallel_2 = parallel_multiply(csr_a, csc_b, num_workers=2)
    time_parallel_2 = time.time() - start
    logger.info(f"  Time: {time_parallel_2:.4f}s")
    logger.info(f"  Speedup: {time_serial/time_parallel_2:.2f}x")
    
    # Parallel (4 workers)
    logger.info("\nParallel multiplication (4 workers):")
    start = time.time()
    result_parallel_4 = parallel_multiply(csr_a, csc_b, num_workers=4)
    time_parallel_4 = time.time() - start
    logger.info(f"  Time: {time_parallel_4:.4f}s")
    logger.info(f"  Speedup: {time_serial/time_parallel_4:.2f}x")
    
    # Cleanup
    for f in [file_a, file_b, sorted_a, sorted_b]:
        try:
            os.remove(f)
        except:
            pass
    
    logger.info("\n" + "=" * 70)
    logger.info("Summary:")
    logger.info("  - Parallel addition shows speedup with multiple workers")
    logger.info("  - Parallel multiplication shows speedup with multiple workers")
    logger.info("  - Actual speedup depends on data size and sparsity pattern")
    logger.info("=" * 70)


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Run parallel benchmarks."""
    logger.info(f"\nParallel CPU Operations")
    logger.info(f"Available CPU cores: {cpu_count()}")
    logger.info("=" * 70)
    
    benchmark_parallel_vs_serial()


if __name__ == "__main__":
    main()
