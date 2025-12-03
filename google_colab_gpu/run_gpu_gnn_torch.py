"""
GPU GNN Benchmark: Dense GPU (PyTorch) multiplication for graph adjacency matrices.
Compares with CPU sparse results for GNN workload analysis.
"""
import torch
import time
import json
import os
import numpy as np
import csv
from tqdm import tqdm

def load_graph_from_csv(filepath, num_nodes):
    """Load graph adjacency matrix from CSV file."""
    rows, cols, vals = [], [], []
    
    if not os.path.exists(filepath):
        # Generate synthetic graph if file doesn't exist
        print(f"⚠ Warning: Data file not found: {filepath}")
        print(f"  Generating synthetic graph instead...")
        np.random.seed(42)
        edges_per_node = 20 if num_nodes <= 1000 else 30
        num_edges = num_nodes * edges_per_node
        
        rows = np.random.randint(0, num_nodes, size=num_edges)
        cols = np.random.randint(0, num_nodes, size=num_edges)
        vals = np.ones(num_edges, dtype=int)
    else:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for parts in reader:
                if len(parts) == 3:
                    try:
                        i = int(parts[0]) - 1  # Convert to 0-based
                        j = int(parts[1]) - 1
                        v = int(parts[2])
                        rows.append(i)
                        cols.append(j)
                        vals.append(v)
                    except ValueError:
                        continue
    
    # Create dense matrix
    matrix = np.zeros((num_nodes, num_nodes))
    for r, c, v in zip(rows, cols, vals):
        if 0 <= r < num_nodes and 0 <= c < num_nodes:
            matrix[r, c] += v
    
    nnz = len([v for v in vals if v != 0])
    sparsity = 100 * (1 - nnz / (num_nodes * num_nodes))
    
    return matrix, nnz, sparsity

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

def run_gnn_tests():
    """Run GPU benchmarks on GNN graph structures."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    print("CUDA Available: True")
    device = torch.device("cuda:0")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*70)
    print("GPU GNN BENCHMARK - Dense GPU Multiplication (RTX 5070 Ti)")
    print("="*70)
    
    graphs = [
        {"name": "Small", "nodes": 500, "file": "gnn_benchmark_comparison/data/graph_small.csv"},
        {"name": "Medium", "nodes": 1000, "file": "gnn_benchmark_comparison/data/graph_medium.csv"},
        {"name": "Large", "nodes": 1500, "file": "gnn_benchmark_comparison/data/graph_large.csv"}
    ]
    
    num_runs = 3
    results = []
    
    for graph in tqdm(graphs, desc="Running GPU GNN Tests", unit="graph"):
        print(f"\n{'='*70}")
        print(f"Testing: {graph['name']} Graph ({graph['nodes']} nodes)")
        print(f"{'='*70}")
        
        # Load graph adjacency matrix
        A_np, nnz, sparsity = load_graph_from_csv(graph['file'], graph['nodes'])
        print(f"Graph: {graph['nodes']}×{graph['nodes']}, {nnz:,} edges ({sparsity:.2f}% sparse)")
        
        # Use same matrix for both (simulating A @ A for GNN operations)
        A_gpu = torch.from_numpy(A_np).float().to(device)
        
        # Benchmark
        print(f"\nBenchmarking DENSE GPU (torch.matmul)...")
        avg_time, std_time = benchmark_gpu_multiplication(A_gpu, A_gpu, num_runs)
        print(f"Average: {avg_time:.6f}s ± {std_time:.6f}s")
        print("="*70)
        
        results.append({
            "graph_name": graph['name'],
            "num_nodes": graph['nodes'],
            "num_edges": nnz,
            "sparsity_percent": sparsity,
            "gpu_time": avg_time,
            "gpu_std": std_time
        })
        
        # Clean up
        del A_gpu
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - GPU GNN Results")
    print("="*70)
    for r in results:
        print(f"{r['graph_name']}: {r['gpu_time']:.6f}s")
    
    # Save results
    os.makedirs("google_colab_gpu/results", exist_ok=True)
    output_path = os.path.abspath("google_colab_gpu/results/gpu_gnn_results.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print comparison format
    print("\n" + "="*70)
    print("GPU TIMES (for comparison with CPU):")
    print("="*70)
    for r in results:
        print(f"{r['graph_name']} Graph: {r['gpu_time']:.6f}s ± {r['gpu_std']:.6f}s")

if __name__ == "__main__":
    run_gnn_tests()
