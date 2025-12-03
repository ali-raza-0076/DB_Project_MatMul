#!/usr/bin/env python
"""
GPU GNN Benchmark - RTX 5070 Ti (Using PyTorch)
Dense GPU multiplication on graph adjacency matrices
Must match CPU test parameters exactly!
"""

import numpy as np
import torch
import time
import json
from scipy import sparse as sp
from pathlib import Path
import csv

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
print()

# ============================================================================
# Data Loading
# ============================================================================

def load_graph_from_csv(filepath, num_nodes):
    """Load graph adjacency matrix from CSV (1-based indexing)."""
    rows, cols, vals = [], [], []
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for parts in reader:
            if len(parts) == 3:
                try:
                    r = int(parts[0]) - 1  # Convert to 0-based
                    c = int(parts[1]) - 1
                    v = int(parts[2])
                    rows.append(r)
                    cols.append(c)
                    vals.append(v)
                except ValueError:
                    continue
    
    # Create scipy sparse, then convert to dense
    sparse_mat = sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    dense_mat = sparse_mat.toarray().astype(np.float32)
    
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
    device = torch.device('cuda')
    A_gpu = torch.from_numpy(A_cpu).to(device)
    B_gpu = torch.from_numpy(B_cpu).to(device)
    
    times = []
    
    # Warmup
    _ = torch.matmul(A_gpu, B_gpu)
    torch.cuda.synchronize()
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        C_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append(end - start)
        print(f"  Run {i+1}: {times[-1]:.6f}s")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

# ============================================================================
# Run GNN Tests
# ============================================================================

def run_gnn_tests():
    """Run GPU tests matching CPU GNN benchmark parameters."""
    
    print("="*70)
    print("GPU GNN BENCHMARK - Dense GPU Multiplication (RTX 5070 Ti)")
    print("="*70)
    print()
    
    # Test configurations matching CPU tests
    test_cases = [
        {"name": "Small", "nodes": 500, "edges_per_node": 20},
        {"name": "Medium", "nodes": 1000, "edges_per_node": 20},
        {"name": "Large", "nodes": 1500, "edges_per_node": 30}
    ]
    
    num_runs = 3
    results = []
    
    # Look for data files
    data_dir = Path("../gnn_benchmark_comparison/data")
    if not data_dir.exists():
        data_dir = Path("../../gnn_benchmark_comparison/data")
    if not data_dir.exists():
        data_dir = Path("gnn_benchmark_comparison/data")
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {test['name']} Graph ({test['nodes']} nodes)")
        print(f"{'='*70}")
        
        # Try to find the graph file
        graph_file = data_dir / f"graph_{test['name'].lower()}.csv"
        
        if not graph_file.exists():
            print(f"⚠ Warning: Data file not found: {graph_file}")
            print(f"  Generating synthetic graph instead...")
            
            # Generate synthetic graph
            num_nodes = test['nodes']
            edges_per_node = test['edges_per_node']
            total_edges = num_nodes * edges_per_node
            
            np.random.seed(42)
            rows = np.random.randint(0, num_nodes, size=total_edges)
            cols = np.random.randint(0, num_nodes, size=total_edges)
            vals = np.ones(total_edges, dtype=int)
            
            sparse_mat = sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
            A_cpu = sparse_mat.toarray().astype(np.float32)
            B_cpu = A_cpu.copy()
            
            nnz = np.count_nonzero(A_cpu)
        else:
            print(f"Loading graph from {graph_file.name}...")
            A_cpu = load_graph_from_csv(graph_file, test['nodes'])
            B_cpu = A_cpu.copy()  # GNN typically multiplies adjacency with itself
            nnz = np.count_nonzero(A_cpu)
        
        sparsity = 100 * (1 - nnz / (test['nodes'] ** 2))
        print(f"Graph: {test['nodes']}×{test['nodes']}, {nnz:,} edges ({sparsity:.2f}% sparse)")
        
        # Benchmark GPU
        print(f"\nBenchmarking DENSE GPU (torch.matmul)...")
        gpu_time, gpu_std = benchmark_gpu_multiplication(A_cpu, B_cpu, num_runs)
        print(f"Average: {gpu_time:.6f}s ± {gpu_std:.6f}s")
        
        results.append({
            "graph_name": test['name'],
            "num_nodes": test['nodes'],
            "num_edges": int(nnz),
            "sparsity": float(sparsity),
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
        exit(1)
    
    # Run tests
    results = run_gnn_tests()
    
    # Print summary
    print("\n\n" + "="*70)
    print("SUMMARY - GPU GNN Results")
    print("="*70)
    
    for r in results:
        print(f"{r['graph_name']}: {r['gpu_time']:.6f}s")
    
    # Save results
    output = {
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
        "test_type": "gnn_benchmark",
        "results": results
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "gpu_gnn_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Display results for comparison
    print("\n" + "="*70)
    print("GPU TIMES (for comparison with CPU):")
    print("="*70)
    for r in results:
        print(f"{r['graph_name']} Graph: {r['gpu_time']:.6f}s ± {r['gpu_std']:.6f}s")
