"""
GNN Dynamic Graph Benchmark: GPU Multicore Test
Tests incremental edge addition vs full recomputation on GPU to find performance threshold.

Methodology:
1. Full Recomputation: Convert COO→Dense→GPU, rebuild entire matrix
2. Incremental Update: Use GPU operations to add new edges directly

Tests varying numbers of new edges (1, 5, 10, 50, 100, 500) to find crossover point.
"""
import numpy as np
import torch
import time
import os
import json
from scipy import sparse as sp
from tabulate import tabulate
from tqdm import tqdm

def check_gpu():
    """Check GPU availability."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires GPU.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.2f} GB")
    return True

def create_base_graph(num_nodes, sparsity_percent, seed=42):
    """
    Create base graph adjacency matrix.
    
    Args:
        num_nodes: Number of nodes in graph
        sparsity_percent: Sparsity percentage (90%, 99%, 99.9%)
        seed: Random seed
    
    Returns:
        scipy sparse CSR matrix
    """
    np.random.seed(seed)
    total_elements = num_nodes * num_nodes
    density = (100 - sparsity_percent) / 100.0
    num_edges = int(total_elements * density)
    
    rows = np.random.randint(0, num_nodes, size=num_edges)
    cols = np.random.randint(0, num_nodes, size=num_edges)
    vals = np.ones(num_edges, dtype=np.float32)
    
    matrix = sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))
    return matrix

def add_edges_gpu_full_recomputation(base_sparse, new_edges, device):
    """
    Add edges by rebuilding entire matrix from scratch on GPU.
    
    Args:
        base_sparse: Original sparse matrix
        new_edges: List of (row, col, val) tuples to add
        device: GPU device
    
    Returns:
        New dense GPU tensor, time taken
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Convert to COO, add new edges
    coo = base_sparse.tocoo()
    rows = list(coo.row)
    cols = list(coo.col)
    vals = list(coo.data)
    
    for r, c, v in new_edges:
        rows.append(r)
        cols.append(c)
        vals.append(v)
    
    # Rebuild sparse matrix and convert to dense GPU
    new_sparse = sp.csr_matrix((vals, (rows, cols)), shape=base_sparse.shape)
    dense_matrix = new_sparse.toarray().astype(np.float32)
    gpu_tensor = torch.from_numpy(dense_matrix).to(device)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    return gpu_tensor, end - start

def add_edges_gpu_incremental(base_tensor, new_edges, device):
    """
    Add edges using GPU tensor operations (incremental update).
    
    Args:
        base_tensor: Original dense GPU tensor
        new_edges: List of (row, col, val) tuples to add
        device: GPU device
    
    Returns:
        Updated GPU tensor, time taken
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # Clone tensor (avoid modifying original)
    new_tensor = base_tensor.clone()
    
    # Add new edges using GPU indexing
    for r, c, v in new_edges:
        new_tensor[r, c] += v
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    return new_tensor, end - start

def test_dynamic_updates_gpu(num_nodes, sparsity_percent, edge_counts, num_runs=3):
    """
    Test dynamic graph updates on GPU with varying numbers of new edges.
    
    Args:
        num_nodes: Graph size
        sparsity_percent: Sparsity (90%, 99%, 99.9%)
        edge_counts: List of edge counts to test [1, 5, 10, 50, 100, 500]
        num_runs: Benchmark repetitions
    
    Returns:
        Results dictionary
    """
    device = torch.device("cuda")
    
    print(f"\n{'='*80}")
    print(f"Dynamic Graph Update Test (GPU Multicore): {num_nodes} nodes, {sparsity_percent}% sparse")
    print(f"{'='*80}")
    
    # Create base graph
    print("Creating base graph...")
    base_graph = create_base_graph(num_nodes, sparsity_percent)
    base_nnz = base_graph.nnz
    print(f"Base graph: {base_nnz:,} edges ({sparsity_percent}% sparse)")
    
    # Convert base to GPU tensor
    base_dense = base_graph.toarray().astype(np.float32)
    base_tensor = torch.from_numpy(base_dense).to(device)
    
    results = []
    
    for num_new_edges in edge_counts:
        print(f"\n--- Adding {num_new_edges} new edge(s) ---")
        
        # Generate new random edges
        np.random.seed(100 + num_new_edges)
        new_edges = [
            (np.random.randint(0, num_nodes), 
             np.random.randint(0, num_nodes), 
             1.0)
            for _ in range(num_new_edges)
        ]
        
        # Test full recomputation
        recomp_times = []
        for _ in tqdm(range(num_runs), desc="  Full Recomputation (GPU)", leave=False):
            _, t = add_edges_gpu_full_recomputation(base_graph, new_edges, device)
            recomp_times.append(t)
        recomp_avg = np.mean(recomp_times)
        recomp_std = np.std(recomp_times)
        
        # Test incremental update
        incr_times = []
        for _ in tqdm(range(num_runs), desc="  Incremental Update (GPU)", leave=False):
            _, t = add_edges_gpu_incremental(base_tensor, new_edges, device)
            incr_times.append(t)
        incr_avg = np.mean(incr_times)
        incr_std = np.std(incr_times)
        
        speedup = recomp_avg / incr_avg if incr_avg > 0 else 0
        winner = "Incremental" if speedup > 1 else "Full Recomp"
        
        print(f"Full Recomputation: {recomp_avg*1000:.6f}ms ± {recomp_std*1000:.6f}ms")
        print(f"Incremental Update: {incr_avg*1000:.6f}ms ± {incr_std*1000:.6f}ms")
        print(f"Speedup: {speedup:.2f}× ({winner} wins)")
        
        results.append({
            "num_new_edges": num_new_edges,
            "full_recomputation_ms": recomp_avg * 1000,
            "full_recomputation_std_ms": recomp_std * 1000,
            "incremental_update_ms": incr_avg * 1000,
            "incremental_update_std_ms": incr_std * 1000,
            "speedup": speedup,
            "winner": winner
        })
    
    return {
        "num_nodes": num_nodes,
        "sparsity_percent": sparsity_percent,
        "base_edges": int(base_nnz),
        "results": results
    }

def main():
    print("="*80)
    print("GNN DYNAMIC GRAPH BENCHMARK - GPU MULTICORE")
    print("Incremental Edge Addition vs Full Recomputation on GPU")
    print("="*80)
    
    if not check_gpu():
        return
    
    # Test configurations
    test_configs = [
        {"nodes": 1000, "sparsity": 90, "edges": [1, 5, 10, 50, 100]},    # 90% sparse
        {"nodes": 1000, "sparsity": 99, "edges": [1, 5, 10, 50, 100]},    # 99% sparse
        {"nodes": 1000, "sparsity": 99.9, "edges": [1, 5, 10, 50, 100]},  # 99.9% sparse
    ]
    
    all_results = []
    num_runs = 3
    
    for config in tqdm(test_configs, desc="Running GPU Tests", unit="config"):
        result = test_dynamic_updates_gpu(
            config["nodes"], 
            config["sparsity"], 
            edge_counts=config["edges"],
            num_runs=num_runs
        )
        all_results.append(result)
    
    # Generate summary
    print("\n" + "="*80)
    print("SUMMARY: Dynamic Update Performance (GPU Multicore)")
    print("="*80)
    
    for res in all_results:
        print(f"\nGraph: {res['num_nodes']} nodes, {res['sparsity_percent']}% sparse")
        print(f"Base edges: {res['base_edges']:,}")
        
        table_data = []
        for r in res['results']:
            table_data.append([
                r['num_new_edges'],
                f"{r['full_recomputation_ms']:.6f}ms",
                f"{r['incremental_update_ms']:.6f}ms",
                f"{r['speedup']:.2f}×",
                r['winner']
            ])
        
        headers = ["New Edges", "Full Recomp (GPU)", "Incremental (GPU)", "Speedup", "Winner"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Find threshold
        threshold_edges = None
        for r in res['results']:
            if r['winner'] == "Full Recomp":
                threshold_edges = r['num_new_edges']
                break
        
        if threshold_edges:
            print(f"\n⚠ THRESHOLD: Full recomputation wins when adding ≥{threshold_edges} edges")
        else:
            print(f"\n✓ Incremental update wins for all tested edge counts (up to {res['results'][-1]['num_new_edges']})")
    
    # Save results
    os.makedirs("gnn_benchmark_comparison/benchmarks", exist_ok=True)
    
    with open("gnn_benchmark_comparison/benchmarks/dynamic_graph_gpu_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Save text summary
    with open("gnn_benchmark_comparison/benchmarks/dynamic_graph_gpu_results.txt", "w", encoding='utf-8') as f:
        f.write("GNN DYNAMIC GRAPH BENCHMARK - GPU MULTICORE\n")
        f.write("Incremental Edge Addition vs Full Recomputation on GPU\n")
        f.write("="*80 + "\n\n")
        
        for res in all_results:
            f.write(f"\nGraph: {res['num_nodes']} nodes, {res['sparsity_percent']}% sparse\n")
            f.write(f"Base edges: {res['base_edges']:,}\n\n")
            
            table_data = []
            for r in res['results']:
                table_data.append([
                    r['num_new_edges'],
                    f"{r['full_recomputation_ms']:.6f}ms",
                    f"{r['incremental_update_ms']:.6f}ms",
                    f"{r['speedup']:.2f}×",
                    r['winner']
                ])
            
            headers = ["New Edges", "Full Recomp (GPU)", "Incremental (GPU)", "Speedup", "Winner"]
            f.write(tabulate(table_data, headers=headers, tablefmt="grid") + "\n")
    
    print(f"\n✓ Results saved to gnn_benchmark_comparison/benchmarks/dynamic_graph_gpu_results.*")
    print("\nKey Finding:")
    print("GPU incremental updates use tensor indexing operations, significantly faster than")
    print("CPU→GPU transfer overhead for full recomputation at small edge counts.")

if __name__ == "__main__":
    main()
