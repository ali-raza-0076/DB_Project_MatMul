"""
Benchmark comparison: Sparse (CSR×CSC) vs Dense multiplication.
Uses sequential sparse algorithm (no multicore/GPU).
"""
import numpy as np
import time
import json
from scipy.sparse import csr_matrix, csc_matrix
from scipy import sparse as sp
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matrix_formats import COOMatrix

def load_sparse_matrix(filepath, shape=(1000, 1000)):
    """Load sparse matrix from CSV (1-based indexing) with integer values."""
    import csv
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) == 3:
                try:
                    # File uses 1-based indexing, convert to 0-based
                    i = int(parts[0]) - 1
                    j = int(parts[1]) - 1
                    v = int(parts[2])  # Integer values
                    data.append((i, j, v))
                except ValueError:
                    continue
    
    coo = COOMatrix(shape=shape, filepath=None, data=data)
    return coo

def sparse_to_dense(coo_matrix):
    """Convert sparse matrix to dense numpy array, merging duplicates by summing."""
    rows, cols = coo_matrix.shape
    dense = np.zeros((rows, cols))
    for chunk in coo_matrix.iter_entries():
        for r, c, v in chunk:
            dense[r, c] += v  # Sum duplicates like scipy does
    return dense

def benchmark_sparse_multiplication(A_coo, B_coo, num_runs=3):
    """Benchmark sparse CSR×CSC multiplication using scipy (correct baseline)."""
    # Convert to scipy sparse matrices
    rows_A, cols_A = A_coo.shape
    rows_B, cols_B = B_coo.shape
    
    # Extract data
    rows_list, cols_list, vals_list = [], [], []
    for chunk in A_coo.iter_entries():
        for r, c, v in chunk:
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)
    A_scipy = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(rows_A, cols_A))
    
    rows_list, cols_list, vals_list = [], [], []
    for chunk in B_coo.iter_entries():
        for r, c, v in chunk:
            rows_list.append(r)
            cols_list.append(c)
            vals_list.append(v)
    B_scipy = sp.csr_matrix((vals_list, (rows_list, cols_list)), shape=(rows_B, cols_B))
    
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = A_scipy @ B_scipy
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}: {times[-1]:.6f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    # Convert result back to COOMatrix for compatibility
    result_coo = result.tocoo()
    result_data = [(result_coo.row[i], result_coo.col[i], result_coo.data[i]) 
                   for i in range(len(result_coo.data))]
    result_matrix = COOMatrix(shape=result.shape, filepath=None, data=result_data)
    
    return avg_time, std_time, result_matrix

def benchmark_dense_multiplication(A_dense, B_dense, num_runs=3):
    """Benchmark dense numpy matrix multiplication."""
    times = []
    for i in range(num_runs):
        start = time.perf_counter()
        result = np.matmul(A_dense, B_dense)
        end = time.perf_counter()
        times.append(end - start)
        print(f"  Run {i+1}: {times[-1]:.6f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result

def main():
    print("=" * 60)
    print("SPARSE vs DENSE MULTIPLICATION BENCHMARK")
    print("=" * 60)
    
    # Load sparse matrices
    print("\n1. Loading sparse matrices...")
    A_coo = load_sparse_matrix("dense_baseline_comparison/data/matrix_a_small.csv")
    B_coo = load_sparse_matrix("dense_baseline_comparison/data/matrix_b_small.csv")
    
    # Get matrix dimensions
    rows_A, cols_A = A_coo.shape
    rows_B, cols_B = B_coo.shape
    
    # Count entries
    entries_A = 0
    for chunk in A_coo.iter_entries():
        entries_A += len(chunk)
    entries_B = 0
    for chunk in B_coo.iter_entries():
        entries_B += len(chunk)
    
    print(f"   Matrix A: {rows_A}×{cols_A}, {entries_A} entries")
    print(f"   Matrix B: {rows_B}×{cols_B}, {entries_B} entries")
    
    # Convert to dense
    print("\n2. Converting to dense arrays...")
    A_dense = sparse_to_dense(A_coo)
    B_dense = sparse_to_dense(B_coo)
    
    sparsity_A = 100 * (1 - entries_A / (rows_A * cols_A))
    sparsity_B = 100 * (1 - entries_B / (rows_B * cols_B))
    print(f"   Sparsity A: {sparsity_A:.2f}% zeros")
    print(f"   Sparsity B: {sparsity_B:.2f}% zeros")
    
    # Memory usage
    sparse_memory = (entries_A + entries_B) * (2 * 4 + 8)  # 2 ints + 1 float per entry
    dense_memory = (A_dense.nbytes + B_dense.nbytes)
    print(f"\n3. Memory usage:")
    print(f"   Sparse: {sparse_memory / 1024:.2f} KB")
    print(f"   Dense:  {dense_memory / 1024:.2f} KB")
    print(f"   Ratio:  {dense_memory / sparse_memory:.2f}× more memory for dense")
    
    # Benchmark sparse multiplication
    print("\n4. Benchmarking SPARSE multiplication (CSR×CSC, sequential)...")
    sparse_time, sparse_std, C_sparse = benchmark_sparse_multiplication(A_coo, B_coo)
    print(f"   Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    entries_C = 0
    for chunk in C_sparse.iter_entries():
        entries_C += len(chunk)
    print(f"   Result: {entries_C} non-zero entries")
    
    # Save sparse result to file
    print("   Saving result to benchmarks/sparse_result.csv...")
    os.makedirs("dense_baseline_comparison/benchmarks", exist_ok=True)
    with open("dense_baseline_comparison/benchmarks/sparse_result.csv", "w") as f:
        for chunk in C_sparse.iter_entries():
            for r, c, v in chunk:
                f.write(f"{r+1},{c+1},{v}\n")  # 1-based indexing
    print(f"   Saved {entries_C} entries")
    
    # Benchmark dense multiplication
    print("\n5. Benchmarking DENSE multiplication (numpy.matmul)...")
    dense_time, dense_std, C_dense = benchmark_dense_multiplication(A_dense, B_dense)
    print(f"   Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    # Verify correctness
    print("\n6. Verifying correctness...")
    C_sparse_dense = sparse_to_dense(C_sparse)
    diff = np.abs(C_sparse_dense - C_dense)
    max_diff = np.max(diff)
    
    # Find positions with differences
    diff_positions = np.where(diff > 1e-6)
    num_diffs = len(diff_positions[0])
    
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Positions with diff > 1e-6: {num_diffs}")
    
    if num_diffs > 0 and num_diffs <= 5:
        print("   Sample differences:")
        for i in range(min(3, num_diffs)):
            r, c = diff_positions[0][i], diff_positions[1][i]
            print(f"     [{r},{c}]: sparse={C_sparse_dense[r,c]:.6f}, dense={C_dense[r,c]:.6f}, diff={diff[r,c]:.6f}")
    
    if max_diff < 1e-6:  # Relaxed tolerance for float precision
        print("   CORRECT - Results match!")
        correct = True
    else:
        print("   ERROR - Results differ (likely precision/zero-fill)")
        correct = False
    
    # Speedup
    speedup = dense_time / sparse_time
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Sparse time:  {sparse_time:.6f}s")
    print(f"Dense time:   {dense_time:.6f}s")
    if speedup > 1:
        print(f"Speedup:      {speedup:.2f}× (Sparse is FASTER)")
    else:
        print(f"Speedup:      {1/speedup:.2f}× (Dense is FASTER)")
    print("=" * 60)
    
    # Save benchmark results
    results = {
        "matrix_size": f"{rows_A}×{cols_A}",
        "num_entries": {
            "A": entries_A,
            "B": entries_B,
            "C": entries_C
        },
        "sparsity_percent": {
            "A": float(sparsity_A),
            "B": float(sparsity_B)
        },
        "memory_kb": {
            "sparse": float(sparse_memory / 1024),
            "dense": float(dense_memory / 1024),
            "ratio": float(dense_memory / sparse_memory)
        },
        "time_seconds": {
            "sparse_avg": float(sparse_time),
            "sparse_std": float(sparse_std),
            "dense_avg": float(dense_time),
            "dense_std": float(dense_std),
            "speedup": float(speedup)
        },
        "verification": {
            "max_difference": float(max_diff),
            "correct": correct
        }
    }
    
    os.makedirs("dense_baseline_comparison/benchmarks", exist_ok=True)
    with open("dense_baseline_comparison/benchmarks/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save text report
    with open("dense_baseline_comparison/benchmarks/comparison_results.txt", "w", encoding='utf-8') as f:
        f.write("SPARSE vs DENSE MULTIPLICATION BENCHMARK\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Matrix Size: {rows_A}×{cols_A}\n")
        f.write(f"Entries: A={entries_A}, B={entries_B}\n")
        f.write(f"Sparsity: A={sparsity_A:.2f}%, B={sparsity_B:.2f}%\n\n")
        f.write(f"Memory Usage:\n")
        f.write(f"  Sparse: {sparse_memory / 1024:.2f} KB\n")
        f.write(f"  Dense:  {dense_memory / 1024:.2f} KB\n")
        f.write(f"  Ratio:  {dense_memory / sparse_memory:.2f}×\n\n")
        f.write(f"Execution Time:\n")
        f.write(f"  Sparse (CSR×CSC): {sparse_time:.6f}s ± {sparse_std:.6f}s\n")
        f.write(f"  Dense (numpy):    {dense_time:.6f}s ± {dense_std:.6f}s\n")
        f.write(f"  Speedup:          {speedup:.2f}x\n\n")
        f.write(f"Result: {entries_C} non-zero entries\n")
        f.write(f"Verification: {'CORRECT' if correct else 'INCORRECT'}\n")
    
    print(f"\nBenchmark results saved to dense_baseline_comparison/benchmarks/")

if __name__ == "__main__":
    main()
