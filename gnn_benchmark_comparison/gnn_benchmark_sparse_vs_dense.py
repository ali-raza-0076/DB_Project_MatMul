"""
GNN Benchmark: CPU vs GPU Matrix Multiplication for Graph Adjacency
Compares:
1. CPU: Sparse CSR×CSR vs Dense numpy
2. GPU: PyTorch dense matrix multiplication
3. Overall winner comparison
"""

import numpy as np
import time
from scipy import sparse as sp
import csv
import os
import json
from tabulate import tabulate
import torch

# Check GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
if GPU_AVAILABLE:
    DEVICE = torch.device('cuda')
    torch.cuda.synchronize()
else:
    DEVICE = torch.device('cpu')

def load_graph_csr(csv_file):
    """Load graph from CSV and convert to CSR format."""
    rows_list = []
    cols_list = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                src, dst = int(row[0]), int(row[1])
                rows_list.append(src)
                cols_list.append(dst)
    
    num_nodes = max(max(rows_list), max(cols_list)) + 1
    data = np.ones(len(rows_list), dtype=np.float32)
    A_csr = sp.csr_matrix((data, (rows_list, cols_list)), shape=(num_nodes, num_nodes))
    
    return A_csr, num_nodes, len(rows_list)

def benchmark_sparse_multiplication(A_csr, B_csr, runs=3):
    """Benchmark sparse CSR×CSR multiplication."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = A_csr @ B_csr
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)

def benchmark_dense_multiplication(A_dense, B_dense, runs=3):
    """Benchmark dense numpy matrix multiplication."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = A_dense @ B_dense
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)

def benchmark_gpu_multiplication(A_gpu, B_gpu, runs=3):
    """Benchmark GPU dense matrix multiplication."""
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = A_gpu @ B_gpu
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)

def benchmark_gpu_sparse_multiplication(A_sparse_gpu, B_sparse_gpu, runs=3):
    """Benchmark GPU sparse matrix multiplication."""
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.sparse.mm(A_sparse_gpu, B_sparse_gpu.to_dense())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return np.mean(times)

def main():
    print("\n" + "=" * 80)
    print("GNN BENCHMARK: Graph Adjacency Matrix Multiplication")
    print("CPU (Sparse CSR vs Dense) | GPU (Sparse vs Dense)")
    print("=" * 80)
    
    runs = 1
    print(f"\nRuns per test: {runs}")
    
    if GPU_AVAILABLE:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: GPU not available, GPU benchmarks will be skipped")
    print()
    
    # Define test graphs - only 99% sparsity for efficiency
    test_graphs = [
        ("../data/graph_4000nodes_99pct_sparsity.csv", "4K-99%", "4,000 nodes, 99% sparsity"),
        ("../data/graph_8000nodes_99pct_sparsity.csv", "8K-99%", "8,000 nodes, 99% sparsity"),
        ("../data/graph_10000nodes_99pct_sparsity.csv", "10K-99%", "10,000 nodes, 99% sparsity"),
    ]
    
    print("Graph Configurations:")
    for _, name, desc in test_graphs:
        print(f"  {name:8}: {desc}")
    print()
    
    cpu_results = []
    gpu_results = []
    cpu_table_data = []
    gpu_table_data = []
    comparison_table_data = []
    
    for csv_file, graph_name, description in test_graphs:
        full_path = os.path.join(os.path.dirname(__file__), csv_file)
        
        if not os.path.exists(full_path):
            print(f"Skipping {graph_name}: File not found")
            continue
        
        print(f"Testing {graph_name}...")
        
        # Load graph
        A_csr, num_nodes, num_edges = load_graph_csr(full_path)
        B_csr = A_csr  # Self-multiplication (typical GNN operation)
        
        # Convert to dense
        A_dense = A_csr.toarray()
        B_dense = B_csr.toarray()
        
        # Calculate sparsity
        total_elements = num_nodes * num_nodes
        sparsity_pct = (1 - num_edges / total_elements) * 100
        
        # CPU Benchmark
        cpu_sparse_time = benchmark_sparse_multiplication(A_csr, B_csr, runs)
        cpu_dense_time = benchmark_dense_multiplication(A_dense, B_dense, runs)
        cpu_speedup = cpu_dense_time / cpu_sparse_time
        cpu_winner = "Sparse" if cpu_speedup > 1 else "Dense"
        cpu_best_time = min(cpu_sparse_time, cpu_dense_time)
        
        # Store CPU results
        cpu_result = {
            "graph": graph_name,
            "nodes": num_nodes,
            "edges": num_edges,
            "sparsity_pct": sparsity_pct,
            "sparse_time_s": cpu_sparse_time,
            "dense_time_s": cpu_dense_time,
            "speedup": cpu_speedup,
            "winner": cpu_winner,
            "best_time": cpu_best_time
        }
        cpu_results.append(cpu_result)
        
        # CPU table
        cpu_table_data.append([
            graph_name,
            f"{num_nodes:,}",
            f"{num_edges:,}",
            f"{sparsity_pct:.4f}%",
            f"{cpu_sparse_time:.6f}s",
            f"{cpu_dense_time:.6f}s",
            f"{cpu_speedup:.2f}×",
            cpu_winner
        ])
        
        # GPU Benchmark
        if GPU_AVAILABLE:
            # Dense GPU
            A_gpu_dense = torch.from_numpy(A_dense).float().to(DEVICE)
            B_gpu_dense = torch.from_numpy(B_dense).float().to(DEVICE)
            gpu_dense_time = benchmark_gpu_multiplication(A_gpu_dense, B_gpu_dense, runs)
            
            # Sparse GPU (convert CSR to PyTorch sparse COO)
            A_coo = A_csr.tocoo()
            indices = torch.LongTensor([A_coo.row, A_coo.col]).to(DEVICE)
            values = torch.FloatTensor(A_coo.data).to(DEVICE)
            A_gpu_sparse = torch.sparse_coo_tensor(indices, values, A_coo.shape).to(DEVICE)
            
            B_coo = B_csr.tocoo()
            indices = torch.LongTensor([B_coo.row, B_coo.col]).to(DEVICE)
            values = torch.FloatTensor(B_coo.data).to(DEVICE)
            B_gpu_sparse = torch.sparse_coo_tensor(indices, values, B_coo.shape).to(DEVICE)
            
            gpu_sparse_time = benchmark_gpu_sparse_multiplication(A_gpu_sparse, B_gpu_sparse, runs)
            gpu_speedup = gpu_dense_time / gpu_sparse_time
            gpu_winner = "Sparse" if gpu_speedup > 1 else "Dense"
            gpu_best_time = min(gpu_sparse_time, gpu_dense_time)
            
            gpu_result = {
                "graph": graph_name,
                "nodes": num_nodes,
                "edges": num_edges,
                "sparsity_pct": sparsity_pct,
                "sparse_time_s": gpu_sparse_time,
                "dense_time_s": gpu_dense_time,
                "speedup": gpu_speedup,
                "winner": gpu_winner,
                "best_time": gpu_best_time
            }
            gpu_results.append(gpu_result)
            
            gpu_table_data.append([
                graph_name,
                f"{num_nodes:,}",
                f"{num_edges:,}",
                f"{sparsity_pct:.4f}%",
                f"{gpu_sparse_time:.6f}s",
                f"{gpu_dense_time:.6f}s",
                f"{gpu_speedup:.2f}×",
                gpu_winner
            ])
            
            # Comparison
            overall_speedup = cpu_best_time / gpu_best_time
            overall_winner = "GPU" if overall_speedup > 1 else "CPU"
            
            comparison_table_data.append([
                graph_name,
                f"{num_nodes:,}",
                f"{cpu_winner} ({cpu_best_time:.6f}s)",
                f"{gpu_winner} ({gpu_best_time:.6f}s)",
                f"{overall_speedup:.2f}×",
                overall_winner
            ])
    
    # Print CPU table
    print("\n" + "=" * 80)
    print("TABLE 1: CPU Multi-Core Performance (Sparse CSR vs Dense)")
    print("=" * 80)
    cpu_headers = ["Graph", "Nodes", "Edges", "Sparsity", "Sparse Time", "Dense Time", "Speedup", "Winner"]
    print(tabulate(cpu_table_data, headers=cpu_headers, tablefmt="grid"))
    
    # CPU Summary
    print("\nCPU SUMMARY:")
    cpu_sparse_wins = sum(1 for r in cpu_results if r["winner"] == "Sparse")
    cpu_dense_wins = sum(1 for r in cpu_results if r["winner"] == "Dense")
    print(f"Sparse CSR wins: {cpu_sparse_wins}/{len(cpu_results)} configurations")
    print(f"Dense wins: {cpu_dense_wins}/{len(cpu_results)} configurations")
    
    # Print GPU table
    if GPU_AVAILABLE and gpu_table_data:
        print("\n" + "=" * 80)
        print("TABLE 2: GPU Multi-Core Performance (Sparse vs Dense)")
        print("=" * 80)
        gpu_headers = ["Graph", "Nodes", "Edges", "Sparsity", "Sparse Time", "Dense Time", "Speedup", "Winner"]
        print(tabulate(gpu_table_data, headers=gpu_headers, tablefmt="grid"))
        
        # GPU Summary
        print("\nGPU SUMMARY:")
        gpu_sparse_wins = sum(1 for r in gpu_results if r["winner"] == "Sparse")
        gpu_dense_wins = sum(1 for r in gpu_results if r["winner"] == "Dense")
        print(f"Sparse wins: {gpu_sparse_wins}/{len(gpu_results)} configurations")
        print(f"Dense wins: {gpu_dense_wins}/{len(gpu_results)} configurations")
        
        # Print comparison table
        print("\n" + "=" * 80)
        print("TABLE 3: CPU vs GPU Winner Comparison")
        print("=" * 80)
        comp_headers = ["Graph", "Nodes", "Best CPU", "Best GPU", "Speedup", "Winner"]
        print(tabulate(comparison_table_data, headers=comp_headers, tablefmt="grid"))
        
        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        gpu_wins = sum(1 for row in comparison_table_data if row[-1] == "GPU")
        cpu_wins_overall = len(comparison_table_data) - gpu_wins
        print(f"GPU wins: {gpu_wins}/{len(comparison_table_data)} configurations")
        print(f"CPU wins: {cpu_wins_overall}/{len(comparison_table_data)} configurations")
        print()
        
        for i, cpu_res in enumerate(cpu_results):
            if i < len(gpu_results):
                gpu_res = gpu_results[i]
                speedup = cpu_res['best_time'] / gpu_res['best_time']
                winner = "GPU" if speedup > 1 else "CPU"
                print(f"{cpu_res['graph']:8} ({cpu_res['nodes']:6,} nodes): {winner:3} - {speedup:.2f}x speedup")
    
    # Save results
    output_dir = "benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    results_json = {
        "cpu_results": cpu_results,
        "gpu_results": gpu_results if GPU_AVAILABLE else []
    }
    with open(os.path.join(output_dir, "gnn_cpu_gpu_comparison.json"), "w") as f:
        json.dump(results_json, f, indent=2)
    
    # Text summary
    with open(os.path.join(output_dir, "gnn_cpu_gpu_comparison.txt"), "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GNN BENCHMARK: Graph Adjacency Matrix Multiplication\n")
        f.write("CPU (Sparse CSR vs Dense) | GPU (Sparse vs Dense)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Runs per test: {runs}\n")
        if GPU_AVAILABLE:
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\nGraph Configurations:\n")
        for _, name, desc in test_graphs:
            f.write(f"  {name:8}: {desc}\n")
        f.write("\n")
        
        # CPU table
        f.write("=" * 80 + "\n")
        f.write("TABLE 1: CPU Multi-Core Performance (Sparse CSR vs Dense)\n")
        f.write("=" * 80 + "\n")
        f.write(tabulate(cpu_table_data, headers=cpu_headers, tablefmt="grid"))
        f.write(f"\n\nCPU SUMMARY:\n")
        f.write(f"Sparse CSR wins: {cpu_sparse_wins}/{len(cpu_results)} configurations\n")
        f.write(f"Dense wins: {cpu_dense_wins}/{len(cpu_results)} configurations\n")
        
        if GPU_AVAILABLE and gpu_table_data:
            # GPU table
            f.write("\n" + "=" * 80 + "\n")
            f.write("TABLE 2: GPU Multi-Core Performance (Sparse vs Dense)\n")
            f.write("=" * 80 + "\n")
            f.write(tabulate(gpu_table_data, headers=gpu_headers, tablefmt="grid"))
            f.write(f"\n\nGPU SUMMARY:\n")
            gpu_sparse_wins = sum(1 for r in gpu_results if r["winner"] == "Sparse")
            gpu_dense_wins = sum(1 for r in gpu_results if r["winner"] == "Dense")
            f.write(f"Sparse wins: {gpu_sparse_wins}/{len(gpu_results)} configurations\n")
            f.write(f"Dense wins: {gpu_dense_wins}/{len(gpu_results)} configurations\n")
            
            # Comparison table
            f.write("\n" + "=" * 80 + "\n")
            f.write("TABLE 3: CPU vs GPU Winner Comparison\n")
            f.write("=" * 80 + "\n")
            f.write(tabulate(comparison_table_data, headers=comp_headers, tablefmt="grid"))
            
            # Overall summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"GPU wins: {gpu_wins}/{len(comparison_table_data)} configurations\n")
            f.write(f"CPU wins: {cpu_wins_overall}/{len(comparison_table_data)} configurations\n\n")
            for i, cpu_res in enumerate(cpu_results):
                if i < len(gpu_results):
                    gpu_res = gpu_results[i]
                    speedup = cpu_res['best_time'] / gpu_res['best_time']
                    winner = "GPU" if speedup > 1 else "CPU"
                    f.write(f"{cpu_res['graph']:8} ({cpu_res['nodes']:6,} nodes): {winner:3} - {speedup:.2f}x speedup\n")
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to:")
    print(f"  - {output_dir}/gnn_cpu_gpu_comparison.json")
    print(f"  - {output_dir}/gnn_cpu_gpu_comparison.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
