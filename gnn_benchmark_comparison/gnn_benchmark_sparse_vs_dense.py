"""
GNN Benchmark: Sparse CSR vs Dense CPU Matrix Multiplication
Compares sparse CSR×CSR vs dense numpy matrix multiplication for graph adjacency matrices.
"""

import numpy as np
import time
from scipy import sparse as sp
import csv
import os
import json
from tabulate import tabulate

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

def main():
    print("\nGNN BENCHMARK: Graph Adjacency Matrix Multiplication")
    print("Dense CPU vs Sparse CPU (CSR×CSR)")
    print("=" * 70)
    
    runs = 3
    print(f"\nRuns per test: {runs}\n")
    
    # Define test graphs
    test_graphs = [
        ("../data/graph_4000nodes_90pct_sparsity.csv", "4K-90%", "4,000 nodes, 90% sparsity"),
        ("../data/graph_4000nodes_95pct_sparsity.csv", "4K-95%", "4,000 nodes, 95% sparsity"),
        ("../data/graph_4000nodes_99pct_sparsity.csv", "4K-99%", "4,000 nodes, 99% sparsity"),
        ("../data/graph_8000nodes_90pct_sparsity.csv", "8K-90%", "8,000 nodes, 90% sparsity"),
        ("../data/graph_8000nodes_95pct_sparsity.csv", "8K-95%", "8,000 nodes, 95% sparsity"),
        ("../data/graph_8000nodes_99pct_sparsity.csv", "8K-99%", "8,000 nodes, 99% sparsity"),
        ("../data/graph_10000nodes_90pct_sparsity.csv", "10K-90%", "10,000 nodes, 90% sparsity"),
        ("../data/graph_10000nodes_95pct_sparsity.csv", "10K-95%", "10,000 nodes, 95% sparsity"),
        ("../data/graph_10000nodes_99pct_sparsity.csv", "10K-99%", "10,000 nodes, 99% sparsity"),
    ]
    
    print("Graph Descriptions:")
    for _, name, desc in test_graphs:
        print(f"  {name:8}: {desc}")
    print()
    
    results = []
    table_data = []
    
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
        
        # Benchmark
        sparse_time = benchmark_sparse_multiplication(A_csr, B_csr, runs)
        dense_time = benchmark_dense_multiplication(A_dense, B_dense, runs)
        
        speedup = dense_time / sparse_time
        winner = "Sparse" if speedup > 1 else "Dense"
        
        # Store results
        result = {
            "graph": graph_name,
            "nodes": num_nodes,
            "edges": num_edges,
            "sparsity_pct": sparsity_pct,
            "sparse_time_s": sparse_time,
            "dense_time_s": dense_time,
            "speedup": speedup,
            "winner": winner
        }
        results.append(result)
        
        # Format for table
        table_data.append([
            graph_name,
            f"{num_nodes:,}",
            f"{num_edges:,}",
            f"{sparsity_pct:.4f}%",
            f"{sparse_time:.6f}s",
            f"{dense_time:.6f}s",
            f"{speedup:.2f}×",
            winner
        ])
    
    # Print table
    headers = ["Graph", "Nodes", "Edges", "Sparsity", "Sparse Time", "Dense Time", "Speedup", "Winner"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Summary
    print("\nSUMMARY")
    print("=" * 70)
    sparse_wins = sum(1 for r in results if r["winner"] == "Sparse")
    dense_wins = sum(1 for r in results if r["winner"] == "Dense")
    print(f"Sparse wins: {sparse_wins}/{len(results)} configurations")
    print(f"Dense wins:  {dense_wins}/{len(results)} configurations")
    print()
    
    for r in results:
        print(f"{r['graph']:8} ({r['nodes']:6,} nodes): {r['winner']:6} - {r['speedup']:.2f}x speedup, {r['sparsity_pct']:.4f}% sparse")
    
    # Save results
    output_dir = "benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    with open(os.path.join(output_dir, "gnn_sparse_dense_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Text summary
    with open(os.path.join(output_dir, "gnn_sparse_dense_results.txt"), "w") as f:
        f.write("GNN BENCHMARK: Graph Adjacency Matrix Multiplication\n")
        f.write("Dense CPU vs Sparse CPU (CSR×CSR)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Runs per test: {runs}\n\n")
        f.write("Graph Descriptions:\n")
        for _, name, desc in test_graphs:
            f.write(f"  {name:8}: {desc}\n")
        f.write("\n")
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
        f.write("\n\nSUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Sparse wins: {sparse_wins}/{len(results)} configurations\n")
        f.write(f"Dense wins:  {dense_wins}/{len(results)} configurations\n\n")
        for r in results:
            f.write(f"{r['graph']:8} ({r['nodes']:6,} nodes): {r['winner']:6} - {r['speedup']:.2f}x speedup, {r['sparsity_pct']:.4f}% sparse\n")
    
    print(f"\nResults saved to:")
    print(f"  - {output_dir}/gnn_sparse_dense_results.json")
    print(f"  - {output_dir}/gnn_sparse_dense_results.txt")

if __name__ == "__main__":
    main()
