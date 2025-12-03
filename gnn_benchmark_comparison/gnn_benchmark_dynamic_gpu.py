"""
Benchmark dynamic graph updates on GPU using PyTorch.
Tests incremental updates vs full recomputation for GCN-style operations.
"""

import time
import json
import os
import sys
import numpy as np

# Try to import PyTorch for GPU operations
try:
    import torch
    # Test if CUDA is available
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        DEVICE = torch.device('cuda')
        print(f"GPU acceleration: ENABLED")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Warning: CUDA not available, falling back to CPU")
        GPU_AVAILABLE = False
        DEVICE = torch.device('cpu')
except ImportError as e:
    print(f"Warning: PyTorch not available: {e}")
    print("Install PyTorch with: pip install torch")
    GPU_AVAILABLE = False
    DEVICE = None

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def to_numpy(tensor):
    """Convert PyTorch tensor to NumPy."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


def sync_gpu():
    """Synchronize GPU if available."""
    if GPU_AVAILABLE:
        torch.cuda.synchronize()


def load_graph_data_gpu(csv_file):
    """Load graph from CSV and convert to GPU tensors."""
    data = np.loadtxt(csv_file, delimiter=',', dtype=int)
    rows = torch.from_numpy(data[:, 0]).long().to(DEVICE)
    cols = torch.from_numpy(data[:, 1]).long().to(DEVICE)
    num_nodes = int(max(data[:, 0].max(), data[:, 1].max()) + 1)
    return rows, cols, num_nodes


def gcn_forward_gpu(rows, cols, num_nodes, features, weight):
    """
    Simple GCN forward pass on GPU.
    Aggregates neighbor features and applies weight matrix.
    """
    # Aggregate: sum neighbor features for each node (vectorized)
    aggregated = torch.zeros(num_nodes, features.shape[1], dtype=torch.float32, device=DEVICE)
    aggregated.index_add_(0, rows, features[cols])
    
    # Apply weight matrix
    output = torch.matmul(aggregated, weight)
    
    # Synchronize GPU
    sync_gpu()
    
    return output


def save_results_incremental(results, output_file):
    """Save results immediately after each configuration completes."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def add_edges_gpu_full_recomputation(rows, cols, num_nodes, new_edges, features, weight):
    """Add edges and do full recomputation."""
    # Concatenate new edges
    new_rows_np = np.concatenate([to_numpy(rows), np.array([e[0] for e in new_edges], dtype=np.int64)])
    new_cols_np = np.concatenate([to_numpy(cols), np.array([e[1] for e in new_edges], dtype=np.int64)])
    new_rows = torch.from_numpy(new_rows_np).long().to(DEVICE)
    new_cols = torch.from_numpy(new_cols_np).long().to(DEVICE)
    
    # Full forward pass
    start = time.perf_counter()
    output = gcn_forward_gpu(new_rows, new_cols, num_nodes, features, weight)
    sync_gpu()
    elapsed = time.perf_counter() - start
    
    return output, new_rows, new_cols, elapsed


def add_edges_gpu_incremental(rows, cols, num_nodes, new_edges, features, weight, previous_aggregated):
    """Add edges with incremental update."""
    # Copy previous aggregation
    aggregated = previous_aggregated.clone()
    
    # Update only affected nodes
    start = time.perf_counter()
    for src, dst in new_edges:
        aggregated[src] += features[dst]
    
    # Apply weight matrix
    output = torch.matmul(aggregated, weight)
    
    sync_gpu()
    elapsed = time.perf_counter() - start
    
    # Update edges
    new_rows_np = np.concatenate([to_numpy(rows), np.array([e[0] for e in new_edges], dtype=np.int64)])
    new_cols_np = np.concatenate([to_numpy(cols), np.array([e[1] for e in new_edges], dtype=np.int64)])
    new_rows = torch.from_numpy(new_rows_np).long().to(DEVICE)
    new_cols = torch.from_numpy(new_cols_np).long().to(DEVICE)
    
    return output, new_rows, new_cols, aggregated, elapsed


def test_dynamic_updates_gpu(csv_file, sparsity_pct, edges_to_add=3, num_runs=5):
    """
    Test dynamic graph updates: incremental vs full recomputation.
    """
    print(f"\n{'='*60}")
    print(f"Testing: {os.path.basename(csv_file)} (Sparsity: {sparsity_pct}%)")
    print(f"{'='*60}")
    
    # Load graph
    rows, cols, num_nodes = load_graph_data_gpu(csv_file)
    print(f"Nodes: {num_nodes}, Edges: {len(rows)}")
    
    # Initialize features and weight matrix
    feature_dim = 128
    features = torch.randn(num_nodes, feature_dim, dtype=torch.float32, device=DEVICE)
    weight = torch.randn(feature_dim, feature_dim, dtype=torch.float32, device=DEVICE)
    
    # Precompute initial aggregation for incremental method (optimized)
    print("  Precomputing aggregation...")
    aggregated = torch.zeros(num_nodes, feature_dim, dtype=torch.float32, device=DEVICE)
    
    # Vectorized aggregation (much faster)
    aggregated.index_add_(0, rows, features[cols])
    sync_gpu()
    
    # Generate random edges to add (ensure they don't already exist)
    existing_edges = set(zip(to_numpy(rows), to_numpy(cols)))
    new_edges_list = []
    attempts = 0
    max_attempts = edges_to_add * 100
    while len(new_edges_list) < edges_to_add and attempts < max_attempts:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in existing_edges:
            new_edges_list.append((src, dst))
            existing_edges.add((src, dst))
        attempts += 1
    
    if len(new_edges_list) < edges_to_add:
        print(f"Warning: Could only generate {len(new_edges_list)} new edges")
    
    # Test 1: Full recomputation
    print(f"\nFull Recomputation Method (adding {len(new_edges_list)} edges):")
    full_recomp_times = []
    
    for run in range(num_runs):
        # Reset graph
        curr_rows, curr_cols = rows.clone(), cols.clone()
        
        # Add edges with full recomputation
        _, curr_rows, curr_cols, elapsed = add_edges_gpu_full_recomputation(
            curr_rows, curr_cols, num_nodes, new_edges_list, features, weight
        )
        full_recomp_times.append(elapsed)
        
        if (run + 1) % 5 == 0:
            print(f"  Run {run+1}/{num_runs}: {elapsed*1000:.4f} ms")
    
    avg_full_recomp = np.mean(full_recomp_times)
    
    # Test 2: Incremental update
    print(f"\nIncremental Update Method (adding {len(new_edges_list)} edges):")
    incremental_times = []
    
    for run in range(num_runs):
        # Reset graph
        curr_rows, curr_cols = rows.clone(), cols.clone()
        curr_aggregated = aggregated.clone()
        
        # Add edges incrementally
        _, curr_rows, curr_cols, curr_aggregated, elapsed = add_edges_gpu_incremental(
            curr_rows, curr_cols, num_nodes, new_edges_list, features, weight, curr_aggregated
        )
        incremental_times.append(elapsed)
        
        if (run + 1) % 5 == 0:
            print(f"  Run {run+1}/{num_runs}: {elapsed*1000:.4f} ms")
    
    avg_incremental = np.mean(incremental_times)
    
    # Results
    speedup = avg_full_recomp / avg_incremental if avg_incremental > 0 else float('inf')
    
    print(f"\n{'='*60}")
    print(f"Results for {os.path.basename(csv_file)} (Sparsity: {sparsity_pct}%)")
    print(f"{'='*60}")
    print(f"Full Recomputation:  {avg_full_recomp:.6f} s")
    print(f"Incremental Update:  {avg_incremental:.6f} s")
    print(f"Speedup:            {speedup:.2f}x {'(Incremental faster)' if speedup > 1 else '(Full recomp faster)'}")
    
    return {
        'csv_file': os.path.basename(csv_file),
        'num_nodes': int(num_nodes),
        'num_edges': int(len(rows)),
        'sparsity_pct': sparsity_pct,
        'edges_added': len(new_edges_list),
        'full_recomputation_s': float(avg_full_recomp),
        'incremental_update_s': float(avg_incremental),
        'speedup': float(speedup),
        'winner': 'incremental' if speedup > 1 else 'full_recomputation'
    }


def clear_gpu_cache():
    """Clear GPU cache to prevent memory fragmentation."""
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()


def main():
    if not GPU_AVAILABLE:
        print("\n" + "="*60)
        print("ERROR: GPU not available!")
        print("This benchmark requires CUDA-enabled GPU")
        print("Please check your PyTorch installation")
        print("="*60 + "\n")
        return
    
    print("\n" + "="*60)
    print("GPU DYNAMIC GRAPH BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("Node sizes: 4k, 8k, 10k")
    print("Sparsity levels: 90%, 95%, 99%, 99.9%")
    print("="*60 + "\n")
    
    # Test graphs: 4k, 8k, 10k nodes with multiple sparsity levels
    # Resume from where we stopped: 4000 nodes, 95% sparsity onwards
    initial_sizes = [4000, 8000, 10000]
    sparsity_levels = [90, 95, 99, 99.9]
    
    # Skip completed: 4000 nodes @ 90% sparsity
    completed_tests = {(4000, 90)}
    
    edges_to_add = 3
    num_runs = 1000  # 1000 iterations per test to prevent CUDA memory issues
    
    results = []
    max_successful_nodes = 0
    
    print(f"Testing {len(initial_sizes)} graph sizes with {num_runs} iterations each")
    print(f"Graph sizes: {initial_sizes} nodes\n")
    
    for num_nodes in initial_sizes:
        print(f"\n{'#'*60}")
        print(f"Testing graph size: {num_nodes} nodes")
        print(f"{'#'*60}")
        
        size_results = []
        
        for sparsity in sparsity_levels:
            csv_file = f"../data/graph_{num_nodes}nodes_{int(sparsity)}pct_sparsity.csv"
            
            if not os.path.exists(csv_file):
                print(f"⚠️  Graph file not found: {csv_file}")
                print(f"    Generating graph with {num_nodes} nodes, {sparsity}% sparsity...")
                print(f"    Skipping this graph size")
                continue
            
            result = test_dynamic_updates_gpu(csv_file, sparsity, edges_to_add, num_runs)
            size_results.append(result)
            
            # Save results incrementally after each configuration
            results.extend(size_results)
            save_results_incremental(results, "dynamic_gpu_results.json")
            results = results[:-len(size_results)]  # Remove for proper accumulation
        
        # All sparsity levels completed for this graph size
        results.extend(size_results)
        save_results_incremental(results, "dynamic_gpu_results.json")
        clear_gpu_cache()  # Clear GPU memory between graph sizes
        if num_nodes > max_successful_nodes:
            max_successful_nodes = num_nodes
        print(f"\n✅ Successfully completed {num_nodes} nodes (all sparsity levels)")
    
    if len(results) == 0:
        print("\n⚠️  No results collected. Graph files may not exist.")
        print("    Generate graph data first using generate_graph_data.py")
        return
    
    # Save results
    output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "dynamic_gpu_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Dynamic Graph Updates on GPU")
    print(f"{'='*80}")
    print(f"{'Graph':<25} {'Nodes':<8} {'Edges':<8} {'Sparsity':<10} {'Full Recomp (s)':<16} {'Incremental (s)':<16} {'Speedup':<10} {'Winner'}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['csv_file']:<25} {r['num_nodes']:<8} {r['num_edges']:<8} {r['sparsity_pct']:<10.1f} "
              f"{r['full_recomputation_s']:<16.6f} {r['incremental_update_s']:<16.6f} {r['speedup']:<10.2f} {r['winner']}")
    
    print(f"{'-'*80}")
    print(f"Maximum tested graph size: {max_successful_nodes} nodes")
    print(f"Edges added per update: {edges_to_add}")
    print(f"Runs per test: {num_runs}")
    
    # Save summary to text file
    with open(os.path.join(output_dir, "dynamic_gpu_summary.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("SUMMARY: Dynamic Graph Updates on GPU\n")
        f.write("="*80 + "\n")
        f.write(f"{'Graph':<25} {'Nodes':<8} {'Edges':<8} {'Sparsity':<10} {'Full Recomp (s)':<16} {'Incremental (s)':<16} {'Speedup':<10} {'Winner'}\n")
        f.write("-"*80 + "\n")
        
        for r in results:
            f.write(f"{r['csv_file']:<25} {r['num_nodes']:<8} {r['num_edges']:<8} {r['sparsity_pct']:<10.1f} "
                   f"{r['full_recomputation_s']:<16.6f} {r['incremental_update_s']:<16.6f} {r['speedup']:<10.2f} {r['winner']}\n")
        
        f.write("-"*80 + "\n")
        f.write(f"Maximum tested graph size: {max_successful_nodes} nodes\n")
        f.write(f"Edges added per update: {edges_to_add}\n")
        f.write(f"Runs per test: {num_runs}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {os.path.join(output_dir, 'dynamic_gpu_results.json')}")
    print(f"  - {os.path.join(output_dir, 'dynamic_gpu_summary.txt')}")


if __name__ == "__main__":
    main()
