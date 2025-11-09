"""
Parallel Sparse Matrix Multiplication Module
Multi-core CPU parallelization for faster sparse matrix multiplication

Key Features:
- Parallel row block processing using multiprocessing
- Efficient work distribution across CPU cores
- Minimal overhead with shared memory where possible
- Compatible with existing sparse_multiplication.py

Usage:
    from sparse_multiplication_parallel import multiply_matrices_parallel
    
    multiply_matrices_parallel(
        file_a='data/output/matrix_a_sorted.csv',
        file_b='data/output/matrix_b_sorted.csv',
        output_file='data/output/result.csv',
        shape_a=(50000, 50000),
        shape_b=(50000, 50000),
        num_workers=8  # Use 8 CPU cores
    )
"""

import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import tempfile
import os
import time
import multiprocessing as mp
from functools import partial
import numba

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix
from sparse_multiplication import (
    _multiply_row_col,
    _numpy_argsort_2d,
    _merge_duplicates_numba
)


# ============================================================================
# Parallel Worker Functions
# ============================================================================

def _process_row_block_worker(block_idx: int, 
                              row_start: int, 
                              row_end: int,
                              csr_data: Tuple,
                              csc_data: Tuple,
                              n_cols: int,
                              temp_dir: str) -> str:
    """
    Worker function to process a single row block in parallel.
    
    Args:
        block_idx: Block index for identification
        row_start, row_end: Row range to process
        csr_data: Tuple of (row_ptr, col_idx, values) for CSR matrix A
        csc_data: Tuple of (col_ptr, row_idx, values) for CSC matrix B
        n_cols: Number of columns in B
        temp_dir: Directory for temporary files
    
    Returns:
        Path to temporary file containing results for this block
    """
    csr_row_ptr, csr_col_idx, csr_values = csr_data
    csc_col_ptr, csc_row_idx, csc_values = csc_data
    
    # Estimate max entries for this block
    num_rows = row_end - row_start
    max_entries = min(num_rows * n_cols, num_rows * 1000)  # Conservative estimate
    
    # Pre-allocate result arrays
    result_rows = np.empty(max_entries, dtype=np.int32)
    result_cols = np.empty(max_entries, dtype=np.int32)
    result_vals = np.empty(max_entries, dtype=np.float64)
    
    # Compute block using optimized Numba function
    count = _multiply_row_block_numba_parallel(
        csr_row_ptr, csr_col_idx, csr_values,
        csc_col_ptr, csc_row_idx, csc_values,
        row_start, row_end, n_cols,
        result_rows, result_cols, result_vals
    )
    
    if count == 0:
        return None  # No results for this block
    
    # Trim to actual size
    result_rows = result_rows[:count]
    result_cols = result_cols[:count]
    result_vals = result_vals[:count]
    
    # Sort results
    if count > 1:
        result_rows, result_cols, result_vals = _numpy_argsort_2d(
            result_rows, result_cols, result_vals
        )
    
    # Merge duplicates
    result_rows, result_cols, result_vals, final_count = _merge_duplicates_numba(
        result_rows, result_cols, result_vals, count
    )
    
    # Write to temporary file
    temp_file = os.path.join(temp_dir, f'block_{block_idx:04d}.csv')
    with open(temp_file, 'w', newline='') as f:
        for idx in range(final_count):
            f.write(f"{result_rows[idx]},{result_cols[idx]},{result_vals[idx]}\n")
    
    return temp_file


@numba.jit(nopython=True, cache=True, fastmath=True)
def _multiply_row_block_numba_parallel(csr_row_ptr, csr_col_idx, csr_values,
                                       csc_col_ptr, csc_row_idx, csc_values,
                                       row_start, row_end, n_cols,
                                       result_rows, result_cols, result_vals):
    """
    Numba-optimized row block multiplication for parallel processing.
    Same as original but designed for parallel execution.
    """
    count = 0
    threshold = 1e-10
    
    # Build active column list
    active_cols = np.empty(n_cols, dtype=np.int32)
    num_active = 0
    for j in range(n_cols):
        if csc_col_ptr[j] < csc_col_ptr[j + 1]:
            active_cols[num_active] = j
            num_active += 1
    
    # Process each row
    for i in range(row_start, row_end):
        row_start_idx = csr_row_ptr[i]
        row_end_idx = csr_row_ptr[i + 1]
        
        if row_start_idx == row_end_idx:
            continue
        
        row_cols = csr_col_idx[row_start_idx:row_end_idx]
        row_vals = csr_values[row_start_idx:row_end_idx]
        
        # Multiply with active columns only
        for col_idx in range(num_active):
            j = active_cols[col_idx]
            
            col_start_idx = csc_col_ptr[j]
            col_end_idx = csc_col_ptr[j + 1]
            
            col_rows = csc_row_idx[col_start_idx:col_end_idx]
            col_vals = csc_values[col_start_idx:col_end_idx]
            
            c_ij = _multiply_row_col(row_cols, row_vals, col_rows, col_vals)
            
            if abs(c_ij) > threshold:
                result_rows[count] = i
                result_cols[count] = j
                result_vals[count] = c_ij
                count += 1
    
    return count


# ============================================================================
# Parallel Matrix Multiplication Class
# ============================================================================

class ParallelSparseMultiplication:
    """
    Parallel sparse matrix multiplication using multi-core CPU.
    
    Distributes row blocks across multiple worker processes for faster computation.
    """
    
    def __init__(self, 
                 block_size: int = 1000,
                 num_workers: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            block_size: Number of rows per block
            num_workers: Number of parallel workers (default: CPU count)
            logger: Optional logger instance
        """
        self.block_size = block_size
        self.num_workers = num_workers or mp.cpu_count()
        self.logger = logger or logging.getLogger(__name__)
        self.temp_files = []
    
    def multiply_matrices(self, file_a: str, file_b: str, output_file: str,
                         shape_a: Tuple[int, int], shape_b: Tuple[int, int]) -> str:
        """
        Parallel multiplication algorithm.
        
        Args:
            file_a: Path to matrix A (COO CSV, sorted by row,col)
            file_b: Path to matrix B (COO CSV, sorted by row,col)
            output_file: Path for result matrix
            shape_a: Shape of matrix A (m, k)
            shape_b: Shape of matrix B (k, n)
        
        Returns:
            Path to output file
        """
        if shape_a[1] != shape_b[0]:
            raise ValueError(
                f"Incompatible dimensions: A is {shape_a}, B is {shape_b}"
            )
        
        self.logger.info(f"="*70)
        self.logger.info(f"PARALLEL Sparse Matrix Multiplication: A{shape_a} × B{shape_b}")
        self.logger.info(f"Workers: {self.num_workers} CPU cores")
        self.logger.info(f"="*70)
        
        # Load and convert matrices
        self.logger.info("Loading matrices...")
        start_time = time.time()
        
        coo_a = COOMatrix.from_csv(file_a, shape=shape_a)
        coo_b = COOMatrix.from_csv(file_b, shape=shape_b)
        
        self.logger.info("Converting A to CSR...")
        csr_a = coo_a.to_csr()
        
        self.logger.info("Converting B to CSC...")
        csc_b = coo_b.to_csc()
        
        load_time = time.time() - start_time
        self.logger.info(f"Matrix loading and conversion: {load_time:.2f}s")
        
        # Parallel block multiplication
        self.logger.info(f"Parallel block multiplication with {self.num_workers} workers...")
        result_files = self._multiply_blocked_parallel(csr_a, csc_b)
        
        # Merge results
        self.logger.info("Merging results from all blocks...")
        self._merge_block_results(result_files, output_file)
        
        self.logger.info(f"✓ Parallel multiplication complete: {output_file}")
        self.logger.info(f"="*70)
        
        return output_file
    
    def _multiply_blocked_parallel(self, csr_a: CSRMatrix, csc_b: CSCMatrix) -> List[str]:
        """
        Distribute row blocks across parallel workers.
        """
        m, k = csr_a.shape
        k2, n = csc_b.shape
        
        # Create work items (blocks)
        num_blocks = (m + self.block_size - 1) // self.block_size
        work_items = []
        
        for block_idx in range(num_blocks):
            row_start = block_idx * self.block_size
            row_end = min(row_start + self.block_size, m)
            work_items.append((block_idx, row_start, row_end))
        
        self.logger.info(f"Processing {num_blocks} blocks across {self.num_workers} workers...")
        
        # Prepare shared data (CSR/CSC arrays)
        csr_data = (csr_a.row_ptr, csr_a.col_idx, csr_a.values)
        csc_data = (csc_b.col_ptr, csc_b.row_idx, csc_b.values)
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='sparse_mult_parallel_')
        
        # Process blocks in parallel
        start_time = time.time()
        
        with mp.Pool(processes=self.num_workers) as pool:
            worker_func = partial(
                _process_row_block_worker,
                csr_data=csr_data,
                csc_data=csc_data,
                n_cols=n,
                temp_dir=temp_dir
            )
            
            # Map work items to workers
            result_files = pool.starmap(worker_func, work_items)
        
        compute_time = time.time() - start_time
        
        # Filter out None results (empty blocks)
        result_files = [f for f in result_files if f is not None]
        
        self.logger.info(f"Parallel computation: {compute_time:.2f}s")
        self.logger.info(f"Generated {len(result_files)} non-empty blocks")
        
        return result_files
    
    def _merge_block_results(self, result_files: List[str], output_file: str):
        """
        Merge results from all blocks into final output file.
        """
        if not result_files:
            # No results
            with open(output_file, 'w') as f:
                pass
            return
        
        # Read all results
        rows_list = []
        cols_list = []
        vals_list = []
        
        for temp_file in result_files:
            with open(temp_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        try:
                            rows_list.append(int(parts[0]))
                            cols_list.append(int(parts[1]))
                            vals_list.append(float(parts[2]))
                        except ValueError:
                            continue
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except OSError:
                pass
        
        if not rows_list:
            with open(output_file, 'w') as f:
                pass
            return
        
        self.logger.info(f"Merging {len(rows_list)} total entries...")
        
        # Convert to arrays
        rows = np.array(rows_list, dtype=np.int32)
        cols = np.array(cols_list, dtype=np.int32)
        vals = np.array(vals_list, dtype=np.float64)
        
        # Sort across blocks
        if len(rows) > 1:
            rows, cols, vals = _numpy_argsort_2d(rows, cols, vals)
        
        # Merge duplicates
        rows, cols, vals, final_count = _merge_duplicates_numba(rows, cols, vals, len(rows))
        
        # Write final result
        self.logger.info(f"Writing {final_count} final entries...")
        with open(output_file, 'w', newline='') as f:
            for idx in range(final_count):
                f.write(f"{rows[idx]},{cols[idx]},{vals[idx]}\n")


# ============================================================================
# Convenience Function
# ============================================================================

def multiply_matrices_parallel(file_a: str, file_b: str, output_file: str,
                               shape_a: Tuple[int, int], shape_b: Tuple[int, int],
                               block_size: int = 1000,
                               num_workers: Optional[int] = None) -> str:
    """
    Parallel sparse matrix multiplication using multi-core CPU.
    
    Args:
        file_a: Path to matrix A (COO CSV, sorted by row,col)
        file_b: Path to matrix B (COO CSV, sorted by row,col)
        output_file: Path for result matrix
        shape_a: Shape of matrix A (m, k)
        shape_b: Shape of matrix B (k, n)
        block_size: Number of rows per block (default 1000)
        num_workers: Number of CPU cores to use (default: all available)
    
    Returns:
        Path to output file
    """
    logger = logging.getLogger(__name__)
    
    multiplier = ParallelSparseMultiplication(
        block_size=block_size,
        num_workers=num_workers,
        logger=logger
    )
    
    return multiplier.multiply_matrices(
        file_a=file_a,
        file_b=file_b,
        output_file=output_file,
        shape_a=shape_a,
        shape_b=shape_b
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("PARALLEL Sparse Matrix Multiplication")
    print("="*70)
    print(f"\nAvailable CPU cores: {mp.cpu_count()}")
    print("\nUsage:")
    print("  from sparse_multiplication_parallel import multiply_matrices_parallel")
    print("  ")
    print("  multiply_matrices_parallel(")
    print("      file_a='data/output/matrix_a_sorted.csv',")
    print("      file_b='data/output/matrix_b_sorted.csv',")
    print("      output_file='data/output/result_parallel.csv',")
    print("      shape_a=(50000, 50000),")
    print("      shape_b=(50000, 50000),")
    print("      num_workers=8  # Use 8 CPU cores")
    print("  )")
    print("="*70)
