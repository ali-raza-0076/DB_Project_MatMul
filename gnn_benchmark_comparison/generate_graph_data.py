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

if __name__ == "__main__":
    # Graph configurations (optimized for reasonable computation on consumer hardware)
    # Small incremental sizes to show scaling trend without overwhelming system
    graphs = [
        {"name": "small", "nodes": 500, "edges_per_node": 20, "seed": 42},
        {"name": "medium", "nodes": 1000, "edges_per_node": 20, "seed": 123},
        {"name": "large", "nodes": 1500, "edges_per_node": 30, "seed": 456}
    ]
    
    print("="*70)
    print("GENERATING GRAPH ADJACENCY MATRICES FOR GNN BENCHMARKING")
    print("="*70)
    print()
    
    for graph in graphs:
        # Generate two adjacency matrices (A and B)
        print(f"Graph: {graph['name'].upper()}")
        
        file_a = f"gnn_benchmark_comparison/data/graph_{graph['name']}_a.csv"
        file_b = f"gnn_benchmark_comparison/data/graph_{graph['name']}_b.csv"
        
        generate_graph_adjacency(
            graph['nodes'],
            graph['edges_per_node'],
            graph['seed'],
            file_a
        )
        
        generate_graph_adjacency(
            graph['nodes'],
            graph['edges_per_node'],
            graph['seed'] + 1000,
            file_b
        )
    
    print("="*70)
    print("GRAPH DATA GENERATION COMPLETE")
    print("="*70)
