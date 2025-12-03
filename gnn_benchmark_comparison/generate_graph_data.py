"""
Generate graph adjacency matrices for GNN benchmarking.
Simulates social network graphs with extreme sparsity.
"""
import numpy as np
import csv
import os

def generate_graph_adjacency(num_nodes, edges_per_node, seed, output_file):
    """
    Generate graph adjacency matrix in COO format.
    
    Args:
        num_nodes: Number of nodes in graph
        edges_per_node: Average edges per node
        seed: Random seed
        output_file: Output CSV file path
    
    Returns:
        Number of edges generated
    """
    np.random.seed(seed)
    
    edges = []
    total_edges = num_nodes * edges_per_node
    
    # Generate random edges (directed graph)
    for _ in range(total_edges):
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        weight = np.random.randint(1, 11)  # Edge weight 1-10
        edges.append((src, dst, weight))
    
    # Write to file (1-based indexing)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for src, dst, weight in edges:
            writer.writerow([src + 1, dst + 1, weight])
    
    # Calculate sparsity
    total_possible = num_nodes * num_nodes
    sparsity = 100 * (1 - len(edges) / total_possible)
    
    print(f"Generated {output_file}")
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Edges: {len(edges):,}")
    print(f"  Sparsity: {sparsity:.4f}%")
    print()
    
    return len(edges)

def generate_sparse_graph_fast(num_nodes, sparsity_pct, seed, output_file):
    """
    Generate sparse graph ULTRA FAST - accept ~1% duplicates for massive speedup.
    For large graphs (20k+ nodes), tracking every edge in memory is too slow.
    
    Args:
        num_nodes: Number of nodes
        sparsity_pct: Target sparsity percentage (99)
        seed: Random seed
        output_file: Output CSV file
    """
    np.random.seed(seed)
    
    # Calculate number of edges for target sparsity
    total_possible = num_nodes * num_nodes
    density = (100 - sparsity_pct) / 100
    num_edges = int(total_possible * density)
    
    print(f"Generating {num_nodes:,} nodes with {num_edges:,} edges ({sparsity_pct}% sparsity)...")
    
    # For SPEED: generate slightly more edges to account for duplicates (~5% extra)
    # but DON'T track all edges in a set (too memory intensive for 4M+ edges)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    target_with_buffer = int(num_edges * 1.05)  # 5% extra for duplicates
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Generate all edges at once (vectorized - MUCH faster)
        print(f"  Generating {target_with_buffer:,} random edges...")
        src_all = np.random.randint(0, num_nodes, size=target_with_buffer)
        dst_all = np.random.randint(0, num_nodes, size=target_with_buffer)
        
        print(f"  Writing to file...")
        written = 0
        for src, dst in zip(src_all, dst_all):
            if src != dst:  # Only filter self-loops
                writer.writerow([int(src), int(dst)])
                written += 1
                
                if written % 1000000 == 0:
                    print(f"    Written: {written:,} edges")
    
    actual_sparsity = 100 * (1 - written / total_possible)
    
    print(f"✅ Generated {os.path.basename(output_file)}")
    print(f"   Nodes: {num_nodes:,}, Edges: {written:,}")
    print(f"   Sparsity: {actual_sparsity:.2f}%\\n")
    
    return written


if __name__ == "__main__":
    # Generate graphs for GPU dynamic benchmark: 4k, 8k, 10k nodes
    # Multiple sparsity levels: 90%, 95%, 99%, 99.9%
    # Sparsity represents percentage of zero entries in adjacency matrix
    node_sizes = [4000, 8000, 10000]
    sparsity_levels = [90, 95, 99, 99.9]
    
    print("="*70)
    print("GENERATING SPARSE GRAPHS FOR GPU DYNAMIC BENCHMARKING")
    print("Node sizes: 4,000 | 8,000 | 10,000")
    print("Sparsity levels: 90% | 95% | 99% | 99.9%")
    print("="*70)
    print()
    
    for nodes in node_sizes:
        for sparsity in sparsity_levels:
            output_file = f"../data/graph_{nodes}nodes_{int(sparsity)}pct_sparsity.csv"
            seed = nodes * 100 + int(sparsity * 10)
            
            generate_sparse_graph_fast(nodes, sparsity, seed, output_file)
    
    print("="*70)
    print("GRAPH DATA GENERATION COMPLETE")
    print("="*70)
