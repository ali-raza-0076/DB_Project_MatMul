"""
GNN Benchmark: GPU Dense vs CPU Sparse Comparison
Tests graph neural network operations on GPU vs CPU using actual graph data files.
Uses same graph sizes as CPU benchmark (500, 1000, 1500 nodes) for direct comparison.
"""
import numpy as np
import torch
import time
import os
import json
import csv
from scipy import sparse as sp
from tabulate import tabulate
from tqdm import tqdm

def check_gpu():
    """Check GPU availability."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Results will be CPU-only.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    return True

def load_graph_matrix(filepath, num_nodes):
    """
    Load graph adjacency matrix from CSV (1-based indexing).
    
    Args:
        filepath: Path to CSV file
        num_nodes: Number of nodes (matrix will be num_nodes × num_nodes)
    
    Returns:
        scipy sparse CSR matrix
    """
    rows, cols, vals = [], [], []
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
    
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    return matrix

def benchmark_cpu_sparse(A_sparse, B_sparse, num_runs=3):
    """Benchmark CPU sparse matrix multiplication."""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = A_sparse @ B_sparse
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def benchmark_gpu_dense(A_dense, B_dense, num_runs=3):
    """Benchmark GPU dense matrix multiplication."""
    device = torch.device("cuda")
    A_torch = torch.from_numpy(A_dense).float().to(device)
    B_torch = torch.from_numpy(B_dense).float().to(device)
    
    # Warmup
    _ = torch.mm(A_torch, B_torch)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = torch.mm(A_torch, B_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def run_gnn_gpu_benchmark(graph_name, num_nodes, file_a, file_b, num_runs=3):
    """
    Run GPU vs CPU benchmark for one graph size.
    
    Args:
        graph_name: Name of the graph (Small, Medium, Large)
        num_nodes: Number of nodes in the graph
        file_a, file_b: Paths to adjacency matrix CSV files
        num_runs: Number of benchmark runs
    
    Returns:
        dict with results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {graph_name} Graph ({num_nodes:,} nodes)")
    print(f"{'='*80}")
    
    # Load graph adjacency matrices
    print("Loading adjacency matrices from CSV...")
    A_sparse = load_graph_matrix(file_a, num_nodes)
    B_sparse = load_graph_matrix(file_b, num_nodes)
    
    nnz_A = A_sparse.nnz
    nnz_B = B_sparse.nnz
    total_elements = num_nodes * num_nodes
    sparsity_A = 100 * (1 - nnz_A / total_elements)
    sparsity_B = 100 * (1 - nnz_B / total_elements)
    
    print(f"Matrix A: {nnz_A:,} edges ({sparsity_A:.4f}% sparse)")
    print(f"Matrix B: {nnz_B:,} edges ({sparsity_B:.4f}% sparse)")
    
    # Convert to dense for GPU
    print("Converting to dense for GPU...")
    A_dense = A_sparse.toarray().astype(np.float32)
    B_dense = B_sparse.toarray().astype(np.float32)
    
    # Benchmark CPU sparse
    print(f"\nBenchmarking CPU SPARSE (CSR×CSR) - {num_runs} runs...")
    cpu_time, cpu_std = benchmark_cpu_sparse(A_sparse, B_sparse, num_runs)
    print(f"  Average: {cpu_time:.6f}s ± {cpu_std:.6f}s")
    
    # Benchmark GPU dense
    print(f"\nBenchmarking GPU DENSE (torch.mm) - {num_runs} runs...")
    gpu_time, gpu_std = benchmark_gpu_dense(A_dense, B_dense, num_runs)
    print(f"  Average: {gpu_time:.6f}s ± {gpu_std:.6f}s")
    
    # Calculate speedup
    speedup = cpu_time / gpu_time
    winner = "GPU" if speedup > 1 else "CPU"
    
    print(f"\n{'='*80}")
    print(f"RESULT: {winner} wins with {abs(speedup):.2f}× speedup")
    print(f"{'='*80}")
    
    return {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "nnz_A": int(nnz_A),
        "nnz_B": int(nnz_B),
        "sparsity": sparsity_A,
        "cpu_sparse_time": cpu_time,
        "cpu_sparse_std": cpu_std,
        "gpu_dense_time": gpu_time,
        "gpu_dense_std": gpu_std,
        "speedup": speedup,
        "winner": winner
    }

def main():
    print("="*80)
    print("GNN BENCHMARK: GPU DENSE vs CPU SPARSE")
    print("Using Actual Graph Data Files (Same as CPU Benchmark)")
    print("="*80)
    
    has_gpu = check_gpu()
    if not has_gpu:
        print("\nSkipping GPU benchmarks (no CUDA device)")
        return
    
    # Use SAME graph configurations as CPU benchmark
    graphs = [
        {
            "name": "Small",
            "nodes": 500,
            "file_a": "gnn_benchmark_comparison/data/graph_small_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_small_b.csv"
        },
        {
            "name": "Medium",
            "nodes": 1000,
            "file_a": "gnn_benchmark_comparison/data/graph_medium_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_medium_b.csv"
        },
        {
            "name": "Large",
            "nodes": 1500,
            "file_a": "gnn_benchmark_comparison/data/graph_large_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_large_b.csv"
        }
    ]
    
    all_results = []
    num_runs = 3
    
    for graph in tqdm(graphs, desc="Running GPU Benchmarks", unit="graph"):
        result = run_gnn_gpu_benchmark(
            graph["name"],
            graph["nodes"],
            graph["file_a"],
            graph["file_b"],
            num_runs
        )
        all_results.append(result)
    
    # Generate summary table
    print("\n" + "="*80)
    print("SUMMARY: GPU vs CPU Performance (Actual Graph Data)")
    print("="*80)
    
    table_data = []
    for res in all_results:
        table_data.append([
            res["graph_name"],
            res["num_nodes"],
            f"{res['sparsity']:.2f}%",
            f"{res['nnz_A']:,}",
            f"{res['cpu_sparse_time']:.6f}s",
            f"{res['gpu_dense_time']:.6f}s",
            f"{res['speedup']:.2f}×",
            res['winner']
        ])
    
    headers = ["Graph", "Nodes", "Sparsity", "Edges", "CPU Sparse", "GPU Dense", "Speedup", "Winner"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results
    os.makedirs("gnn_benchmark_comparison/benchmarks", exist_ok=True)
    
    with open("gnn_benchmark_comparison/benchmarks/gnn_gpu_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save text summary
    with open("gnn_benchmark_comparison/benchmarks/gnn_gpu_results.txt", "w", encoding='utf-8') as f:
        f.write("GNN BENCHMARK: GPU DENSE vs CPU SPARSE\n")
        f.write("Using Actual Graph Data Files (Same as CPU Benchmark)\n")
        f.write("="*80 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid") + "\n\n")
        f.write("Note: Uses same graph files as CPU benchmark for direct comparison\n")
        f.write("Graph sparsity: ~98% (typical for social networks)\n")
    
    print(f"\n✓ Results saved to gnn_benchmark_comparison/benchmarks/gnn_gpu_results.*")

if __name__ == "__main__":
    main()
