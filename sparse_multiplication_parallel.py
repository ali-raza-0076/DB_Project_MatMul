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
import json
from datetime import datetime

from matrix_formats import COOMatrix, CSRMatrix, CSCMatrix


# ============================================================================
# Helper Functions (since they don't exist in sparse_multiplication.py)
# ============================================================================

@numba.jit(nopython=True, cache=True)
def _multiply_row_col(row_cols, row_vals, col_rows, col_vals):
    """
    Compute dot product of sparse row and sparse column using two-pointer technique.
    """
    result = 0.0
    i, j = 0, 0
    n1, n2 = len(row_cols), len(col_rows)
    
    while i < n1 and j < n2:
        if row_cols[i] < col_rows[j]:
            i += 1
        elif row_cols[i] > col_rows[j]:
            j += 1
        else:  # Match found
            result += row_vals[i] * col_vals[j]
            i += 1
            j += 1
    
    return result


def _numpy_argsort_2d(rows, cols, vals):
    """
    Sort COO matrix by (row, col) using numpy lexsort.
    """
    # Lexsort sorts by last key first, so we reverse the order
    sorted_idx = np.lexsort((cols, rows))
    return rows[sorted_idx], cols[sorted_idx], vals[sorted_idx]


@numba.jit(nopython=True, cache=True)
def _merge_duplicates_numba(rows, cols, vals, count):
    """
    Merge duplicate (row, col) entries by summing their values.
    Assumes input is sorted by (row, col).
    """
    if count == 0:
        return rows, cols, vals, 0
    
    # Allocate output arrays
    out_rows = np.empty(count, dtype=rows.dtype)
    out_cols = np.empty(count, dtype=cols.dtype)
    out_vals = np.empty(count, dtype=vals.dtype)
    
    out_rows[0] = rows[0]
    out_cols[0] = cols[0]
    out_vals[0] = vals[0]
    
    out_idx = 0
    
    for i in range(1, count):
        if rows[i] == out_rows[out_idx] and cols[i] == out_cols[out_idx]:
            # Same position - add values
            out_vals[out_idx] += vals[i]
        else:
            # New position
            out_idx += 1
            out_rows[out_idx] = rows[i]
            out_cols[out_idx] = cols[i]
            out_vals[out_idx] = vals[i]
    
    final_count = out_idx + 1
    return out_rows, out_cols, out_vals, final_count


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
            # Keep 0-based for temporary files (will convert to 1-based at final output)
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
        self.benchmarks = {
            'total_time': 0,
            'load_time': 0,
            'conversion_time': 0,
            'parallel_multiplication_time': 0,
            'merge_time': 0,
            'num_workers': self.num_workers,
            'block_size': block_size,
            'num_blocks': 0
        }
    
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
        start_total = time.time()
        
        if shape_a[1] != shape_b[0]:
            raise ValueError(
                f"Incompatible dimensions: A is {shape_a}, B is {shape_b}"
            )
        
        self.logger.info(f"="*70)
        self.logger.info(f"PARALLEL Sparse Matrix Multiplication - BENCHMARK MODE")
        self.logger.info(f"A{shape_a} × B{shape_b}")
        self.logger.info(f"Workers: {self.num_workers} CPU cores")
        self.logger.info(f"Block size: {self.block_size} rows")
        self.logger.info(f"="*70)
        
        # Load and convert matrices
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"LOADING & CONVERSION PHASE")
        self.logger.info(f"{'='*70}")
        self.logger.info("Loading matrices...")
        start_time = time.time()
        
        coo_a = COOMatrix.from_csv(file_a, shape=shape_a)
        coo_b = COOMatrix.from_csv(file_b, shape=shape_b)
        
        nnz_a = coo_a.count_nnz()
        nnz_b = coo_b.count_nnz()
        
        self.logger.info(f"Matrix A: {nnz_a:,} non-zeros")
        self.logger.info(f"Matrix B: {nnz_b:,} non-zeros")
        
        self.logger.info("Converting A to CSR...")
        start_conv = time.time()
        csr_a = coo_a.to_csr()
        conv_a_time = time.time() - start_conv
        self.logger.info(f"  A → CSR: {conv_a_time:.3f}s")
        
        self.logger.info("Converting B to CSC...")
        start_conv = time.time()
        csc_b = coo_b.to_csc()
        conv_b_time = time.time() - start_conv
        self.logger.info(f"  B → CSC: {conv_b_time:.3f}s")
        
        load_time = time.time() - start_time
        self.benchmarks['load_time'] = load_time
        self.benchmarks['conversion_time'] = conv_a_time + conv_b_time
        self.logger.info(f"\nTotal loading & conversion: {load_time:.3f}s")
        
        # Parallel block multiplication
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"PARALLEL MULTIPLICATION PHASE")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Parallel block multiplication with {self.num_workers} workers...")
        start_mult = time.time()
        result_files = self._multiply_blocked_parallel(csr_a, csc_b)
        self.benchmarks['parallel_multiplication_time'] = time.time() - start_mult
        
        self.logger.info(f"\nParallel multiplication completed in: {self.benchmarks['parallel_multiplication_time']:.3f}s")
        self.logger.info(f"Speedup potential: {self.num_workers}x (with {self.num_workers} workers)")
        
        # Merge results
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"MERGING PHASE")
        self.logger.info(f"{'='*70}")
        self.logger.info("Merging results from all blocks...")
        start_merge = time.time()
        self._merge_block_results(result_files, output_file)
        self.benchmarks['merge_time'] = time.time() - start_merge
        
        self.benchmarks['total_time'] = time.time() - start_total
        
        # Print summary
        self._print_summary()
        
        # Save benchmark report
        self._save_benchmark_report(output_file, shape_a, shape_b, nnz_a, nnz_b)
        
        self.logger.info(f"\n✓ Parallel multiplication complete: {output_file}")
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
        self.benchmarks['num_blocks'] = num_blocks
        work_items = []
        
        for block_idx in range(num_blocks):
            row_start = block_idx * self.block_size
            row_end = min(row_start + self.block_size, m)
            work_items.append((block_idx, row_start, row_end))
        
        self.logger.info(f"Processing {num_blocks} blocks across {self.num_workers} workers...")
        self.logger.info(f"Each block processes ~{self.block_size} rows")
        
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
        
        self.logger.info(f"Parallel computation: {compute_time:.3f}s")
        self.logger.info(f"Generated {len(result_files)} non-empty blocks")
        self.logger.info(f"Average time per block: {compute_time/num_blocks:.4f}s")
        
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
                # Convert from 0-based (internal) to 1-based (external format)
                f.write(f"{rows[idx]+1},{cols[idx]+1},{vals[idx]}\n")
    
    def _print_summary(self):
        """Print benchmark summary."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"PERFORMANCE SUMMARY")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Total execution time:           {self.benchmarks['total_time']:.3f}s")
        self.logger.info(f"  - Loading & Conversion:       {self.benchmarks['load_time']:.3f}s ({self.benchmarks['load_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"  - Parallel Multiplication:    {self.benchmarks['parallel_multiplication_time']:.3f}s ({self.benchmarks['parallel_multiplication_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"  - Merging Results:            {self.benchmarks['merge_time']:.3f}s ({self.benchmarks['merge_time']/self.benchmarks['total_time']*100:.1f}%)")
        self.logger.info(f"\nParallelization configuration:")
        self.logger.info(f"  - Number of CPU cores:        {self.num_workers}")
        self.logger.info(f"  - Block size:                 {self.block_size} rows")
        self.logger.info(f"  - Total blocks processed:     {self.benchmarks['num_blocks']}")
        self.logger.info(f"{'='*70}")
    
    def _save_benchmark_report(self, output_file: str, shape_a: Tuple[int, int], 
                               shape_b: Tuple[int, int], nnz_a: int, nnz_b: int):
        """Save detailed benchmark report to file."""
        # Count result entries
        result_size = 0
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result_size = sum(1 for _ in f)
        
        report = {
            'operation': 'Sparse Matrix Multiplication (Parallel)',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'num_workers': self.num_workers,
                'block_size': self.block_size,
                'num_blocks': self.benchmarks['num_blocks'],
                'available_cores': mp.cpu_count()
            },
            'input': {
                'matrix_a_shape': list(shape_a),
                'matrix_b_shape': list(shape_b),
                'matrix_a_nonzeros': nnz_a,
                'matrix_b_nonzeros': nnz_b,
                'sparsity_a': round(nnz_a / (shape_a[0] * shape_a[1]) * 100, 3),
                'sparsity_b': round(nnz_b / (shape_b[0] * shape_b[1]) * 100, 3)
            },
            'output': {
                'result_shape': [shape_a[0], shape_b[1]],
                'result_nonzeros': result_size,
                'sparsity_result': round(result_size / (shape_a[0] * shape_b[1]) * 100, 3) if shape_a[0] * shape_b[1] > 0 else 0
            },
            'performance': {
                'total_time_seconds': round(self.benchmarks['total_time'], 3),
                'load_conversion_seconds': round(self.benchmarks['load_time'], 3),
                'parallel_multiplication_seconds': round(self.benchmarks['parallel_multiplication_time'], 3),
                'merge_time_seconds': round(self.benchmarks['merge_time'], 3)
            },
            'metrics': {
                'flops': nnz_a * nnz_b,  # Rough estimate
                'throughput_nonzeros_per_sec': round((nnz_a + nnz_b) / self.benchmarks['total_time'], 2) if self.benchmarks['total_time'] > 0 else 0,
                'parallel_efficiency': round(self.benchmarks['parallel_multiplication_time'] / self.benchmarks['total_time'] * 100, 1),
                'theoretical_speedup': self.num_workers,
                'avg_time_per_block': round(self.benchmarks['parallel_multiplication_time'] / self.benchmarks['num_blocks'], 4) if self.benchmarks['num_blocks'] > 0 else 0
            }
        }
        
        # Save as JSON
        report_file = output_file.replace('.csv', '_benchmark.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save as readable text
        report_txt = output_file.replace('.csv', '_benchmark.txt')
        with open(report_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SPARSE MATRIX MULTIPLICATION - PARALLEL BENCHMARK REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write(f"  CPU Cores Used:        {report['configuration']['num_workers']}\n")
            f.write(f"  Available Cores:       {report['configuration']['available_cores']}\n")
            f.write(f"  Block Size:            {report['configuration']['block_size']} rows\n")
            f.write(f"  Total Blocks:          {report['configuration']['num_blocks']}\n\n")
            
            f.write("INPUT MATRICES:\n")
            f.write(f"  Matrix A Shape:        {report['input']['matrix_a_shape'][0]} × {report['input']['matrix_a_shape'][1]}\n")
            f.write(f"  Matrix A Non-zeros:    {report['input']['matrix_a_nonzeros']:,}\n")
            f.write(f"  Matrix A Sparsity:     {report['input']['sparsity_a']:.3f}%\n\n")
            
            f.write(f"  Matrix B Shape:        {report['input']['matrix_b_shape'][0]} × {report['input']['matrix_b_shape'][1]}\n")
            f.write(f"  Matrix B Non-zeros:    {report['input']['matrix_b_nonzeros']:,}\n")
            f.write(f"  Matrix B Sparsity:     {report['input']['sparsity_b']:.3f}%\n\n")
            
            f.write("OUTPUT MATRIX:\n")
            f.write(f"  Result Shape:          {report['output']['result_shape'][0]} × {report['output']['result_shape'][1]}\n")
            f.write(f"  Result Non-zeros:      {report['output']['result_nonzeros']:,}\n")
            f.write(f"  Result Sparsity:       {report['output']['sparsity_result']:.3f}%\n\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write(f"  Total Time:            {report['performance']['total_time_seconds']:.3f}s\n")
            f.write(f"    - Load & Convert:    {report['performance']['load_conversion_seconds']:.3f}s\n")
            f.write(f"    - Parallel Multiply: {report['performance']['parallel_multiplication_seconds']:.3f}s\n")
            f.write(f"    - Merging:           {report['performance']['merge_time_seconds']:.3f}s\n\n")
            
            f.write("EFFICIENCY:\n")
            f.write(f"  Throughput:            {report['metrics']['throughput_nonzeros_per_sec']:,.0f} nonzeros/sec\n")
            f.write(f"  Parallel Efficiency:   {report['metrics']['parallel_efficiency']:.1f}%\n")
            f.write(f"  Theoretical Speedup:   {report['metrics']['theoretical_speedup']}x\n")
            f.write(f"  Avg Time per Block:    {report['metrics']['avg_time_per_block']:.4f}s\n\n")
            
            f.write("="*70 + "\n")
        
        self.logger.info(f"\n📊 Benchmark reports saved:")
        self.logger.info(f"   JSON: {report_file}")
        self.logger.info(f"   TXT:  {report_txt}")


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
    
    # File paths
    file_a = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_a_sorted.csv"
    file_b = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_b_sorted.csv"
    output_file = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_product_parallel.csv"
    
    # Matrix dimensions (change to match your data)
    shape_a = (50000, 50000)
    shape_b = (50000, 50000)
    
    print(f"\nRunning parallel multiplication...")
    print(f"Input A: {file_a}")
    print(f"Input B: {file_b}")
    print(f"Output: {output_file}")
    print(f"Shape A: {shape_a}")
    print(f"Shape B: {shape_b}")
    
    # Run parallel multiplication
    multiply_matrices_parallel(
        file_a=file_a,
        file_b=file_b,
        output_file=output_file,
        shape_a=shape_a,
        shape_b=shape_b,
        num_workers=mp.cpu_count()
    )
    
    print("="*70)
