#!/usr/bin/env python
"""
GPU Sparsity Comparison - RTX 5070 Ti (Using PyTorch)
Dense GPU multiplication at different sparsity levels
Must match CPU test parameters exactly!
"""

import numpy as np
import torch
import time
import json
from scipy import sparse as sp
from pathlib import Path

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
print()

# ============================================================================
# Data Generation
# ============================================================================

def generate_sparse_matrix(size, sparsity_percent, seed):
    """Generate sparse matrix matching CPU test parameters."""
    np.random.seed(seed)
    
    total_elements = size * size
    density = (100 - sparsity_percent) / 100.0
    num_entries = int(total_elements * density)
    
    rows = np.random.randint(0, size, size=num_entries)
    cols = np.random.randint(0, size, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)  # Integer 1-10
    
    # Create sparse then convert to dense
    sparse_mat = sp.csr_matrix((values, (rows, cols)), shape=(size, size))
    dense_mat = sparse_mat.toarray().astype(np.float32)  # Use float32 for GPU
    
    return dense_mat

# ============================================================================
# GPU Benchmark Function
# ============================================================================

def benchmark_gpu_multiplication(A_cpu, B_cpu, num_runs=3):
    """
    Benchmark dense GPU matrix multiplication using PyTorch.
    
    Args:
        A_cpu: numpy array (CPU)
        B_cpu: numpy array (CPU)
        num_runs: number of runs for averaging
    
    Returns:
        avg_time, std_time
    """
    # Transfer to GPU
    device = torch.device('cuda')
    A_gpu = torch.from_numpy(A_cpu).to(device)
    B_gpu = torch.from_numpy(B_cpu).to(device)
    
    times = []
    
    # Warmup run
    _ = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    
    for i in range(num_runs):
        torch.cuda.synchronize()  # Ensure GPU ready
        
        start = time.perf_counter()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()  # Wait for completion
        end = time.perf_counter()
        
        times.append(end - start)
        print(f"  Run {i+1}: {times[-1]:.6f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

# ============================================================================
# Run Sparsity Tests
# ============================================================================

def run_sparsity_tests():
    """Run GPU tests matching CPU benchmark parameters."""
    
    print("="*70)
    print("GPU SPARSITY COMPARISON - Dense GPU Multiplication (RTX 5070 Ti)")
    print("="*70)
    print()
    
    size = 1000
    sparsity_levels = [50, 90, 95, 99]  # Match CPU tests
    num_runs = 3
    
    results = []
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*70}")
        print(f"Testing: {size}×{size} matrix, {sparsity}% sparsity")
        print(f"{'='*70}")
        
        # Generate matrices (same seeds as CPU tests)
        print("Generating matrices...")
        A_cpu = generate_sparse_matrix(size, sparsity, seed=42)
        B_cpu = generate_sparse_matrix(size, sparsity, seed=123)
        
        nnz_A = np.count_nonzero(A_cpu)
        nnz_B = np.count_nonzero(B_cpu)
        actual_sparsity = 100 * (1 - nnz_A / (size * size))
        
        print(f"Matrix A: {nnz_A:,} non-zeros ({actual_sparsity:.2f}% sparse)")
        print(f"Matrix B: {nnz_B:,} non-zeros")
        
        # Benchmark GPU
        print(f"\nBenchmarking DENSE GPU (torch.matmul)...")
        gpu_time, gpu_std = benchmark_gpu_multiplication(A_cpu, B_cpu, num_runs)
        print(f"Average: {gpu_time:.6f}s ± {gpu_std:.6f}s")
        
        results.append({
            "sparsity_percent": sparsity,
            "matrix_size": size,
            "nnz_A": int(nnz_A),
            "nnz_B": int(nnz_B),
            "actual_sparsity": float(actual_sparsity),
            "gpu_time": float(gpu_time),
            "gpu_std": float(gpu_std),
            "method": "Dense GPU (PyTorch)"
        })
        
        print(f"{'='*70}")
    
    return results

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Please install PyTorch with CUDA support.")
        print("Visit: https://pytorch.org/get-started/locally/")
        exit(1)
    
    # Run tests
    results = run_sparsity_tests()
    
    # Print summary
    print("\n\n" + "="*70)
    print("SUMMARY - GPU Results")
    print("="*70)
    
    for r in results:
        print(f"{r['sparsity_percent']}% sparse: {r['gpu_time']:.6f}s")
    
    # Save results
    output = {
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
        "test_type": "sparsity_comparison",
        "matrix_size": 1000,
        "results": results
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "gpu_sparsity_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Display results for comparison
    print("\n" + "="*70)
    print("GPU TIMES (for comparison with CPU):")
    print("="*70)
    for r in results:
        print(f"Sparsity {r['sparsity_percent']}%: {r['gpu_time']:.6f}s ± {r['gpu_std']:.6f}s")
