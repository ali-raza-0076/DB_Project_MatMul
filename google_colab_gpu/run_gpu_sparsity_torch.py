"""
GPU Sparsity Benchmark: Dense GPU (PyTorch) multiplication at super sparse levels.
Tests 90%, 99%, 99.9% sparsity (≤10% density). Compares with CPU sparse results.
"""
import torch
import time
import json
import os
import numpy as np
from tqdm import tqdm

def generate_sparse_matrix(size, sparsity_percent, seed):
    """Generate sparse matrix for GPU testing."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    total_elements = size * size
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(total_elements * density)
    
    rows = np.random.randint(0, size, size=num_entries)
    cols = np.random.randint(0, size, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)
    
    # Create dense matrix from sparse coordinates
    matrix = np.zeros((size, size))
    for r, c, v in zip(rows, cols, values):
        matrix[r, c] += v
    
    return matrix, rows, cols, values

def benchmark_gpu_multiplication(A_gpu, B_gpu, num_runs=3):
    """Benchmark dense GPU multiplication with progress bar."""
    # Warm up
    _ = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense GPU", leave=False):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time

def run_sparsity_tests():
    """Run GPU benchmarks at different sparsity levels."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print("CUDA Available: True")
    device = torch.device("cuda:0")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*70)
    print("GPU SPARSITY COMPARISON - Dense GPU Multiplication (RTX 5070 Ti)")
    print("Super Sparse Matrices: 90%, 99%, 99.9% sparsity (≤10% density)")
    print("="*70)
    
    matrix_size = 1000
    sparsity_levels = [90, 99, 99.9]  # Super sparse only
    num_runs = 3
    
    results = []
    
    for sparsity in tqdm(sparsity_levels, desc="Running GPU Tests", unit="test"):
        print(f"\n{'='*70}")
        print(f"Testing: {matrix_size}×{matrix_size} matrix, {sparsity}% sparsity")
        print(f"{'='*70}")
        
        # Generate matrices
        print("Generating matrices...")
        A_np, rows_A, cols_A, vals_A = generate_sparse_matrix(matrix_size, sparsity, 42)
        B_np, rows_B, cols_B, vals_B = generate_sparse_matrix(matrix_size, sparsity, 123)
        
        actual_nnz_A = len(vals_A)
        actual_sparsity = 100 * (1 - actual_nnz_A / (matrix_size * matrix_size))
        print(f"Matrix A: {actual_nnz_A:,} non-zeros ({actual_sparsity:.2f}% sparse)")
        print(f"Matrix B: {len(vals_B):,} non-zeros")
        
        # Transfer to GPU
        A_gpu = torch.from_numpy(A_np).float().to(device)
        B_gpu = torch.from_numpy(B_np).float().to(device)
        
        # Benchmark
        print(f"\nBenchmarking DENSE GPU (torch.matmul)...")
        avg_time, std_time = benchmark_gpu_multiplication(A_gpu, B_gpu, num_runs)
        print(f"Average: {avg_time:.6f}s ± {std_time:.6f}s")
        print("="*70)
        
        results.append({
            "sparsity_percent": sparsity,
            "matrix_size": matrix_size,
            "gpu_time": avg_time,
            "gpu_std": std_time
        })
        
        # Clean up
        del A_gpu, B_gpu
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - GPU Results")
    print("="*70)
    for r in results:
        print(f"{r['sparsity_percent']}% sparse: {r['gpu_time']:.6f}s")
    
    # Save results
    os.makedirs("google_colab_gpu/results", exist_ok=True)
    output_path = os.path.abspath("google_colab_gpu/results/gpu_sparsity_results.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print comparison format
    print("\n" + "="*70)
    print("GPU TIMES (for comparison with CPU):")
    print("="*70)
    for r in results:
        print(f"Sparsity {r['sparsity_percent']}%: {r['gpu_time']:.6f}s ± {r['gpu_std']:.6f}s")

if __name__ == "__main__":
    run_sparsity_tests()
