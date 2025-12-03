"""
Simple test script for block-based GPU multiplication.
Uses small test matrices to verify the implementation works.

Run this locally or on Colab to test before processing large matrices.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path if running locally
sys.path.append(str(Path(__file__).parent))

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("⚠️ CuPy not installed. Install with: pip install cupy-cuda11x")
    GPU_AVAILABLE = False
    sys.exit(1)

from scipy import sparse as sp
import time


def generate_test_matrices(size=1000, sparsity=0.95, seed=42):
    """Generate small test matrices for verification."""
    print(f"Generating test matrices: {size}×{size}, {sparsity*100:.0f}% sparse")
    
    np.random.seed(seed)
    
    # Generate sparse matrices
    density = 1 - sparsity
    nnz = int(size * size * density)
    
    # Matrix A
    rows_a = np.random.randint(0, size, nnz)
    cols_a = np.random.randint(0, size, nnz)
    vals_a = np.random.randint(1, 11, nnz).astype(float)
    A_csr = sp.csr_matrix((vals_a, (rows_a, cols_a)), shape=(size, size))
    
    # Matrix B
    rows_b = np.random.randint(0, size, nnz)
    cols_b = np.random.randint(0, size, nnz)
    vals_b = np.random.randint(1, 11, nnz).astype(float)
    B_csr = sp.csr_matrix((vals_b, (rows_b, cols_b)), shape=(size, size))
    
    print(f"  Matrix A: {A_csr.nnz:,} non-zeros")
    print(f"  Matrix B: {B_csr.nnz:,} non-zeros")
    
    return A_csr, B_csr


def block_multiply_simple(A_csr, B_csr, block_size):
    """
    Simplified block multiplication for testing.
    Returns full result matrix directly (no disk I/O).
    """
    matrix_size = A_csr.shape[0]
    num_blocks = int(np.ceil(matrix_size / block_size))
    
    print(f"\nBlock multiplication:")
    print(f"  Matrix size: {matrix_size}×{matrix_size}")
    print(f"  Block size: {block_size}×{block_size}")
    print(f"  Number of blocks: {num_blocks}×{num_blocks} = {num_blocks**2}")
    
    # Collect all result entries
    result_rows, result_cols, result_vals = [], [], []
    
    total_time = 0
    
    for i in range(num_blocks):
        row_start = i * block_size
        row_end = min(row_start + block_size, matrix_size)
        
        for j in range(num_blocks):
            col_start = j * block_size
            col_end = min(col_start + block_size, matrix_size)
            
            # Extract blocks
            A_block = A_csr[row_start:row_end, :].toarray()
            B_block = B_csr[:, col_start:col_end].toarray()
            
            # Transfer to GPU
            A_gpu = cp.asarray(A_block, dtype=cp.float32)
            B_gpu = cp.asarray(B_block, dtype=cp.float32)
            
            # Multiply on GPU
            start = time.perf_counter()
            C_gpu = cp.matmul(A_gpu, B_gpu)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - start
            total_time += elapsed
            
            # Transfer back
            C_block = cp.asnumpy(C_gpu)
            
            # Clean up
            del A_gpu, B_gpu, C_gpu
            cp.get_default_memory_pool().free_all_blocks()
            
            # Extract non-zeros from result block
            block_rows, block_cols = np.nonzero(C_block)
            for r, c in zip(block_rows, block_cols):
                result_rows.append(row_start + r)
                result_cols.append(col_start + c)
                result_vals.append(C_block[r, c])
            
            print(f"  Block ({i},{j}): {len(block_rows)} nnz, {elapsed*1000:.2f}ms")
    
    # Build result matrix
    result_csr = sp.csr_matrix(
        (result_vals, (result_rows, result_cols)),
        shape=(matrix_size, matrix_size)
    )
    
    print(f"\nTotal time: {total_time:.4f}s")
    print(f"Result: {result_csr.nnz:,} non-zeros")
    
    return result_csr, total_time


def verify_correctness(A_csr, B_csr, result_block, sample_size=100):
    """Verify block multiplication matches CPU sparse multiplication."""
    print(f"\nVerifying correctness (sample: {sample_size}×{sample_size})...")
    
    # Compute expected result on CPU (full sparse)
    expected_full = A_csr @ B_csr
    
    # Compare sample
    expected_sample = expected_full[:sample_size, :sample_size].toarray()
    result_sample = result_block[:sample_size, :sample_size].toarray()
    
    max_diff = np.abs(expected_sample - result_sample).max()
    avg_diff = np.abs(expected_sample - result_sample).mean()
    
    print(f"  Max difference: {max_diff}")
    print(f"  Avg difference: {avg_diff}")
    
    if max_diff < 1e-3:
        print("  ✅ PASS: Results match!")
        return True
    else:
        print("  ❌ FAIL: Results differ!")
        return False


def benchmark_comparison(A_csr, B_csr, block_size):
    """Compare block GPU vs CPU sparse multiplication."""
    print(f"\n{'='*70}")
    print("BENCHMARK COMPARISON")
    print(f"{'='*70}")
    
    # 1. Block GPU
    print("\n[1/3] Block GPU multiplication...")
    result_gpu, time_gpu = block_multiply_simple(A_csr, B_csr, block_size)
    
    # 2. CPU Sparse
    print("\n[2/3] CPU sparse multiplication...")
    start = time.perf_counter()
    result_cpu = A_csr @ B_csr
    time_cpu = time.perf_counter() - start
    print(f"  Time: {time_cpu:.4f}s")
    print(f"  Result: {result_cpu.nnz:,} non-zeros")
    
    # 3. Verification
    print("\n[3/3] Verification...")
    matches = verify_correctness(A_csr, B_csr, result_gpu)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Block GPU: {time_gpu:.4f}s")
    print(f"CPU Sparse: {time_cpu:.4f}s")
    print(f"Speedup: {time_cpu/time_gpu:.2f}x")
    print(f"Verification: {'✅ PASS' if matches else '❌ FAIL'}")
    print(f"{'='*70}")


def main():
    """Run tests."""
    print("="*70)
    print("GPU Block Multiplication - Test Script")
    print("="*70)
    
    # Check GPU
    if not GPU_AVAILABLE:
        return
    
    print(f"\n✅ GPU Available: {cp.cuda.Device()}")
    device = cp.cuda.Device()
    free_mem, total_mem = device.mem_info
    print(f"   GPU Memory: {total_mem/1e9:.2f}GB total, {free_mem/1e9:.2f}GB free")
    
    # Test configurations
    tests = [
        {"size": 1000, "sparsity": 0.95, "block_size": 250},
        {"size": 2000, "sparsity": 0.99, "block_size": 500},
    ]
    
    for i, config in enumerate(tests, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}/{len(tests)}: {config['size']}×{config['size']}, "
              f"{config['sparsity']*100:.0f}% sparse")
        print(f"{'='*70}")
        
        # Generate matrices
        A_csr, B_csr = generate_test_matrices(
            size=config['size'],
            sparsity=config['sparsity']
        )
        
        # Benchmark
        benchmark_comparison(A_csr, B_csr, config['block_size'])
    
    print("\n\n✅ All tests complete!")
    print("\nNext steps:")
    print("  1. If tests pass, you're ready for large matrices!")
    print("  2. Upload gpu_block_multiplication.ipynb to Google Colab")
    print("  3. Upload your matrix CSV files")
    print("  4. Run the notebook")


if __name__ == "__main__":
    main()
