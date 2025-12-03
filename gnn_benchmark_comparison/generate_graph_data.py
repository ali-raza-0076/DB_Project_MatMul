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

def generate_sparse_graph(num_nodes, sparsity_pct, seed, output_file):
    """
    Generate sparse graph with specific sparsity level.
    
    Args:
        num_nodes: Number of nodes
        sparsity_pct: Target sparsity percentage (90, 99, 99.9)
        seed: Random seed
        output_file: Output CSV file
    """
    np.random.seed(seed)
    
    # Calculate number of edges for target sparsity
    total_possible = num_nodes * num_nodes
    density = (100 - sparsity_pct) / 100
    num_edges = int(total_possible * density)
    
    # Generate unique random edges
    edges = set()
    attempts = 0
    max_attempts = num_edges * 10
    
    while len(edges) < num_edges and attempts < max_attempts:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst:  # No self-loops
            edges.add((src, dst))
        attempts += 1
    
    # Write to file (0-indexed for direct use)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for src, dst in edges:
            writer.writerow([src, dst])
    
    actual_sparsity = 100 * (1 - len(edges) / total_possible)
    
    print(f"Generated {os.path.basename(output_file)}")
    print(f"  Nodes: {num_nodes:,}, Edges: {len(edges):,}")
    print(f"  Target Sparsity: {sparsity_pct}%, Actual: {actual_sparsity:.4f}%")
    print()
    
    return len(edges)


if __name__ == "__main__":
    # Generate MASSIVE graphs to achieve 30-60 second execution times
    # Up to 50,000 nodes with millions of edges at 90% sparsity
    node_sizes = [5000, 6000, 8000, 10000, 15000, 20000, 30000, 40000, 50000]
    sparsity_levels = [90, 99, 99.9]
    
    print("="*70)
    print("GENERATING MASSIVE SPARSE GRAPHS FOR GNN DYNAMIC BENCHMARKING")
    print("Up to 50,000 nodes to achieve 30-60 second execution times")
    print("="*70)
    print()
    
    for nodes in node_sizes:
        for sparsity in sparsity_levels:
            output_file = f"../data/graph_{nodes}nodes_{int(sparsity)}pct_sparsity.csv"
            seed = nodes * 100 + int(sparsity * 10)
            
            generate_sparse_graph(nodes, sparsity, seed, output_file)
    
    print("="*70)
    print("GRAPH DATA GENERATION COMPLETE")
    print("="*70)
