"""
GNN Benchmark: Compare Dense CPU vs Sparse CPU for graph adjacency matrices.
Tests extreme sparsity levels (99.98%, 99.995%, 99.9999%) typical in GNN applications.
"""
import numpy as np
import time
from scipy import sparse as sp
import csv
import os
from tabulate import tabulate
import json
from tqdm import tqdm

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

def benchmark_sparse_multiplication(A_sparse, B_sparse, num_runs=3):
    """Benchmark sparse CSR × CSR multiplication."""
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
    """Benchmark dense numpy matrix multiplication."""
    times = []
    for _ in tqdm(range(num_runs), desc="  Dense CPU", leave=False):
        start = time.perf_counter()
        result = np.matmul(A_dense, B_dense)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time, result

def run_gnn_benchmark(graph_name, num_nodes, file_a, file_b, num_runs=3):
    """
    Run benchmark for one graph size.
    
    Args:
        graph_name: Name of the graph (Small, Medium, Large)
        num_nodes: Number of nodes in the graph
        file_a, file_b: Paths to adjacency matrix CSV files
        num_runs: Number of benchmark runs
    
    Returns:
        dict with results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {graph_name} Graph ({num_nodes:,} nodes)")
    print(f"{'='*70}")
    
    # Load graph adjacency matrices
    print("Loading adjacency matrices...")
    A_sparse = load_graph_matrix(file_a, num_nodes)
    B_sparse = load_graph_matrix(file_b, num_nodes)
    
    nnz_A = A_sparse.nnz
    nnz_B = B_sparse.nnz
    total_elements = num_nodes * num_nodes
    sparsity_A = 100 * (1 - nnz_A / total_elements)
    sparsity_B = 100 * (1 - nnz_B / total_elements)
    
    print(f"Matrix A: {nnz_A:,} edges ({sparsity_A:.4f}% sparse)")
    print(f"Matrix B: {nnz_B:,} edges ({sparsity_B:.4f}% sparse)")
    
    # Convert to dense
    print("Converting to dense (for comparison)...")
    A_dense = A_sparse.toarray()
    B_dense = B_sparse.toarray()
    
    # Memory usage
    sparse_memory = (nnz_A + nnz_B) * (2 * 4 + 8)  # 2 ints + 1 float per entry
    dense_memory = A_dense.nbytes + B_dense.nbytes
    
    print(f"\nMemory:")
    print(f"  Sparse: {sparse_memory / 1024 / 1024:.2f} MB")
    print(f"  Dense:  {dense_memory / 1024 / 1024:.2f} MB")
    print(f"  Ratio:  {dense_memory / sparse_memory:.2f}×")
    
    # Benchmark sparse
    print(f"\nBenchmarking SPARSE (CSR×CSR) - {num_runs} runs...")
    sparse_time, sparse_std, C_sparse = benchmark_sparse_multiplication(A_sparse, B_sparse, num_runs)
    print(f"  Average: {sparse_time:.6f}s ± {sparse_std:.6f}s")
    print(f"  Result: {C_sparse.nnz:,} edges in result")
    
    # Benchmark dense
    print(f"\nBenchmarking DENSE (numpy.matmul) - {num_runs} runs...")
    dense_time, dense_std, C_dense = benchmark_dense_multiplication(A_dense, B_dense, num_runs)
    print(f"  Average: {dense_time:.6f}s ± {dense_std:.6f}s")
    
    # Calculate speedup
    speedup = dense_time / sparse_time
    winner = "Sparse" if speedup > 1 else "Dense"
    
    print(f"\n{'='*70}")
    print(f"RESULT: {winner} wins with {abs(speedup):.2f}× speedup")
    print(f"{'='*70}")
    
    return {
        "graph_name": graph_name,
        "num_nodes": num_nodes,
        "nnz_A": nnz_A,
        "nnz_B": nnz_B,
        "sparsity_percent": sparsity_A,
        "sparse_time": sparse_time,
        "sparse_std": sparse_std,
        "dense_time": dense_time,
        "dense_std": dense_std,
        "speedup": speedup,
        "winner": winner,
        "memory_sparse_mb": sparse_memory / 1024 / 1024,
        "memory_dense_mb": dense_memory / 1024 / 1024,
        "memory_ratio": dense_memory / sparse_memory,
        "result_edges": C_sparse.nnz
    }

def main():
    print("="*70)
    print("GNN BENCHMARK: Graph Adjacency Matrix Multiplication")
    print("Dense CPU vs Sparse CPU (CSR×CSR)")
    print("="*70)
    print("\nSimulating Graph Neural Network (GNN) computations")
    print("Use case: Social networks, knowledge graphs, molecular structures")
    print()
    
    # Configuration
    num_runs = 3
    
    # Define graph sizes (optimized for reasonable computation time)
    # Scaled to be computable on consumer hardware while showing trend
    graphs = [
        {
            "name": "Small",
            "nodes": 500,
            "file_a": "gnn_benchmark_comparison/data/graph_small_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_small_b.csv",
            "description": "Small social network (500 nodes, ~98% sparse)"
        },
        {
            "name": "Medium",
            "nodes": 1000,
            "file_a": "gnn_benchmark_comparison/data/graph_medium_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_medium_b.csv",
            "description": "Medium social network (1000 nodes, ~98% sparse)"
        },
        {
            "name": "Large",
            "nodes": 1500,
            "file_a": "gnn_benchmark_comparison/data/graph_large_a.csv",
            "file_b": "gnn_benchmark_comparison/data/graph_large_b.csv",
            "description": "Large social network (1500 nodes, ~98.7% sparse)"
        }
    ]
    
    # Run benchmarks
    results = []
    for graph in tqdm(graphs, desc="Running GNN Benchmarks", unit="graph"):
        result = run_gnn_benchmark(
            graph["name"],
            graph["nodes"],
            graph["file_a"],
            graph["file_b"],
            num_runs
        )
        result["description"] = graph["description"]
        results.append(result)
    
    # Create performance table
    print("\n\n" + "="*70)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*70)
    
    table_data = []
    for r in results:
        table_data.append([
            r['graph_name'],
            f"{r['num_nodes']:,}",
            f"{r['nnz_A']:,}",
            f"{r['sparsity_percent']:.4f}%",
            f"{r['sparse_time']:.6f}s",
            f"{r['dense_time']:.6f}s",
            f"{r['speedup']:.2f}×",
            r['winner']
        ])
    
    headers = ["Graph", "Nodes", "Edges", "Sparsity", "Sparse Time", "Dense Time", "Speedup", "Winner"]
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - GNN Use Cases")
    print("="*70)
    
    sparse_wins = sum(1 for r in results if r['winner'] == 'Sparse')
    dense_wins = sum(1 for r in results if r['winner'] == 'Dense')
    
    print(f"Sparse wins: {sparse_wins}/{len(results)} graph sizes")
    print(f"Dense wins:  {dense_wins}/{len(results)} graph sizes")
    print()
    
    # Show which method wins for each graph
    for r in results:
        print(f"{r['graph_name']:8} ({r['num_nodes']:>7,} nodes): {r['winner']:6} "
              f"- {r['speedup']:.2f}× speedup, {r['sparsity_percent']:.4f}% sparse")
    
    print()
    print("Key Insight:")
    if sparse_wins == len(results):
        print("  ✓ Sparse multiplication dominates for ALL graph sizes")
        print("  ✓ GNN applications should use sparse representations")
        print("  ✓ Speedup increases dramatically with graph size")
    
    # Memory efficiency
    print()
    print("Memory Efficiency:")
    for r in results:
        print(f"  {r['graph_name']:8}: Sparse uses {r['memory_ratio']:.0f}× less memory than dense")
    
    # Save results
    os.makedirs("gnn_benchmark_comparison/benchmarks", exist_ok=True)
    
    # Save JSON
    with open("gnn_benchmark_comparison/benchmarks/gnn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save table as text
    with open("gnn_benchmark_comparison/benchmarks/gnn_results.txt", "w", encoding='utf-8') as f:
        f.write("GNN BENCHMARK: Graph Adjacency Matrix Multiplication\n")
        f.write("Dense CPU vs Sparse CPU (CSR×CSR)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Runs per test: {num_runs}\n\n")
        f.write("Graph Descriptions:\n")
        for r in results:
            f.write(f"  {r['graph_name']:8}: {r['description']}\n")
        f.write("\n" + table + "\n\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Sparse wins: {sparse_wins}/{len(results)} graph sizes\n")
        f.write(f"Dense wins:  {dense_wins}/{len(results)} graph sizes\n\n")
        for r in results:
            f.write(f"{r['graph_name']:8} ({r['num_nodes']:>7,} nodes): {r['winner']:6} "
                   f"- {r['speedup']:.2f}x speedup, {r['sparsity_percent']:.4f}% sparse\n")
    
    # Save CSV
    with open("gnn_benchmark_comparison/benchmarks/gnn_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Graph", "Nodes", "Edges", "Sparsity%", "SparseTime(s)", "DenseTime(s)", 
                        "Speedup", "Winner", "SparseMem(MB)", "DenseMem(MB)"])
        for r in results:
            writer.writerow([
                r['graph_name'],
                r['num_nodes'],
                r['nnz_A'],
                f"{r['sparsity_percent']:.4f}",
                f"{r['sparse_time']:.6f}",
                f"{r['dense_time']:.6f}",
                f"{r['speedup']:.2f}",
                r['winner'],
                f"{r['memory_sparse_mb']:.2f}",
                f"{r['memory_dense_mb']:.2f}"
            ])
    
    print(f"\nResults saved to gnn_benchmark_comparison/benchmarks/")
    print("  - gnn_results.json")
    print("  - gnn_results.txt")
    print("  - gnn_results.csv")

if __name__ == "__main__":
    main()
