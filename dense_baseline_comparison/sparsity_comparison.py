"""
Compare Dense CPU vs Sparse CSR×CSC at different sparsity levels.
Tests super sparse matrices: 90%, 99%, 99.9% sparsity (≤10% density)
Outputs performance table showing when each method wins.
"""
import numpy as np
import time
from scipy import sparse as sp
import csv
import os
from tabulate import tabulate
import json
from tqdm import tqdm

def generate_sparse_matrix(size, sparsity_percent, seed):
    """
    Generate sparse matrix with given sparsity.
    
    Args:
        size: Matrix dimension (size × size)
        sparsity_percent: Percentage of zeros (50, 90, 95, 99)
        seed: Random seed
    
    Returns:
        rows, cols, values lists for COO format
    """
    np.random.seed(seed)
    
    # Calculate number of non-zero entries
    total_elements = size * size
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(total_elements * density)
    
    # Generate random positions (may have duplicates)
    rows = np.random.randint(0, size, size=num_entries)
    cols = np.random.randint(0, size, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)
    
    return rows, cols, values

def benchmark_sparse_multiplication(A_sparse, B_sparse, num_runs=3):
    """Benchmark sparse CSR × CSC multiplication."""
    times = []
    for _ in tqdm(range(num_runs), desc="  Sparse CPU", leave=False):
        start = time.perf_counter()
        result = A_sparse @ B_sparse
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result

def benchmark_dense_multiplication(A_dense, B_dense, num_runs=3):
    """Benchmark dense numpy multiplication."""
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense CPU", leave=False):
        start = time.perf_counter()
        result = np.matmul(A_dense, B_dense)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result

def run_comparison(size, sparsity_percent, num_runs=3):
    """
    Run comparison for one sparsity level.
    
    Returns:
        dict with results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {size}×{size} matrix, {sparsity_percent}% sparsity")
    print(f"{'='*70}")
    
    # Generate matrices
    print("Generating matrices...")
    rows_A, cols_A, vals_A = generate_sparse_matrix(size, sparsity_percent, seed=42)
    rows_B, cols_B, vals_B = generate_sparse_matrix(size, sparsity_percent, seed=123)
    
    # Create sparse matrices (scipy)
    A_sparse = sp.csr_matrix((vals_A, (rows_A, cols_A)), shape=(size, size))
    B_sparse = sp.csr_matrix((vals_B, (rows_B, cols_B)), shape=(size, size))
    
    actual_nnz_A = A_sparse.nnz
    actual_nnz_B = B_sparse.nnz
    actual_sparsity_A = 100 * (1 - actual_nnz_A / (size * size))
    actual_sparsity_B = 100 * (1 - actual_nnz_B / (size * size))
    
    print(f"Matrix A: {actual_nnz_A:,} non-zeros ({actual_sparsity_A:.2f}% sparse)")
    print(f"Matrix B: {actual_nnz_B:,} non-zeros ({actual_sparsity_B:.2f}% sparse)")
    
    # Convert to dense
    A_dense = A_sparse.toarray()
    B_dense = B_sparse.toarray()
    
    # Memory usage
    sparse_memory = (actual_nnz_A + actual_nnz_B) * (2 * 4 + 8)  # 2 ints + 1 float
    dense_memory = A_dense.nbytes + B_dense.nbytes
    
    print(f"\nMemory:")
    print(f"  Sparse: {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"  Dense:  {dense_memory / 1024 / 1024:.2f} MB")
    print(f"  Ratio:  {dense_memory / sparse_memory:.2f}×")
    
    # Benchmark sparse
    print(f"\nBenchmarking SPARSE (CSR×CSC)...")
    sparse_time, sparse_std, C_sparse = benchmark_sparse_multiplication(A_sparse, B_sparse, num_runs)
    print(f"  Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    
    # Benchmark dense
    print(f"\nBenchmarking DENSE (numpy.matmul)...")
    dense_time, dense_std, C_dense = benchmark_dense_multiplication(A_dense, B_dense, num_runs)
    print(f"  Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    # Calculate speedup
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    print(f"\n{'='*70}")
    print(f"RESULT: {winner} wins with {abs(speedup):.2f}× speedup")
    print(f"{'='*70}")
    
    return {
        "sparsity_percent": sparsity_percent,
        "actual_sparsity_A": actual_sparsity_A,
        "actual_sparsity_B": actual_sparsity_B,
        "matrix_size": size,
        "nnz_A": actual_nnz_A,
        "nnz_B": actual_nnz_B,
        "sparse_time": sparse_time,
        "sparse_std": sparse_std,
        "dense_time": dense_time,
        "dense_std": dense_std,
        "speedup": speedup,
        "winner": winner,
        "memory_sparse_mb": sparse_memory / 1024 / 1024,
        "memory_dense_mb": dense_memory / 1024 / 1024,
        "memory_ratio": dense_memory / sparse_memory
    }

def main():
    print("="*70)
    print("SPARSITY COMPARISON: Dense CPU vs Sparse CSR×CSC")
    print("="*70)
    
    # Configuration
    matrix_size = 1000
    sparsity_levels = [90, 99, 99.9]  # Super sparse: ≤10% density
    num_runs = 3
    
    # Run comparisons
    results = []
    for sparsity in tqdm(sparsity_levels, desc="Running Benchmarks", unit="test"):
        result = run_comparison(matrix_size, sparsity, num_runs)
        results.append(result)
    
    # Create performance table
    print("\n\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    for r in results:
        table_data.append([
            f"{r['sparsity_percent']}%",
            f"{r['nnz_A']:,}",
            f"{r['sparse_time']:.6f}s",
            f"{r['dense_time']:.6f}s",
            f"{r['speedup']:.2f}×",
            r['winner'],
            f"{r['memory_sparse_mb']:.2f} MB",
            f"{r['memory_dense_mb']:.2f} MB"
        ])
    
    headers = ["Sparsity", "Non-Zeros", "Sparse Time", "Dense Time", "Speedup", "Winner", "Sparse Mem", "Dense Mem"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    sparse_wins = sum(1 for r in results if r['winner'] == 'Sparse')
    dense_wins = sum(1 for r in results if r['winner'] == 'Dense')
    
    print(f"Matrix Size: {matrix_size}×{matrix_size}")
    print(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels")
    print(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels")
    print()
    
    # When does each method win?
    sparse_winning_sparsities = [r['sparsity_percent'] for r in results if r['winner'] == 'Sparse']
    dense_winning_sparsities = [r['sparsity_percent'] for r in results if r['winner'] == 'Dense']
    
    if sparse_winning_sparsities:
        print(f"Sparse wins at: {', '.join(map(str, sparse_winning_sparsities))}% sparsity")
    if dense_winning_sparsities:
        print(f"Dense wins at:  {', '.join(map(str, dense_winning_sparsities))}% sparsity")
    
    # Save results
    os.makedirs("dense_baseline_comparison/benchmarks", exist_ok=True)
    
    # Save JSON
    with open("dense_baseline_comparison/benchmarks/sparsity_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save table as text
    with open("dense_baseline_comparison/benchmarks/sparsity_comparison.txt", "w", encoding='utf-8') as f:
        f.write("SPARSITY COMPARISON: Dense CPU vs Sparse CSR×CSC\n")
        f.write("="*70 + "\n\n")
        f.write(f"Matrix Size: {matrix_size}×{matrix_size}\n")
        f.write(f"Runs per test: {num_runs}\n\n")
        f.write(table + "\n\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Sparse wins: {sparse_wins}/{len(results)} sparsity levels\n")
        f.write(f"Dense wins:  {dense_wins}/{len(results)} sparsity levels\n\n")
        if sparse_winning_sparsities:
            f.write(f"Sparse wins at: {', '.join(map(str, sparse_winning_sparsities))}% sparsity\n")
        if dense_winning_sparsities:
            f.write(f"Dense wins at:  {', '.join(map(str, dense_winning_sparsities))}% sparsity\n")
    
    # Save CSV for easy analysis
    with open("dense_baseline_comparison/benchmarks/sparsity_comparison.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sparsity%", "NonZeros", "SparseTime(s)", "DenseTime(s)", "Speedup", "Winner", "SparseMem(MB)", "DenseMem(MB)"])
        for r in results:
            writer.writerow([
                r['sparsity_percent'],
                r['nnz_A'],
                f"{r['sparse_time']:.6f}",
                f"{r['dense_time']:.6f}",
                f"{r['speedup']:.2f}",
                r['winner'],
                f"{r['memory_sparse_mb']:.2f}",
                f"{r['memory_dense_mb']:.2f}"
            ])
    
    print(f"\nResults saved to dense_baseline_comparison/benchmarks/")
    print("  - sparsity_comparison.json")
    print("  - sparsity_comparison.txt")
    print("  - sparsity_comparison.csv")

if __name__ == "__main__":
    main()
