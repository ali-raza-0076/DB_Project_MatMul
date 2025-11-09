# GNN Benchmark - Graph Neural Network Use Cases

## Purpose

Benchmark sparse vs dense matrix multiplication on graph adjacency matrices, simulating real-world Graph Neural Network (GNN) applications like social networks.

## Graph Sizes (Optimized for Safe Computation)

| Graph | Nodes | Edges/Node | Total Edges | Sparsity | Description |
|-------|-------|------------|-------------|----------|-------------|
| **Small** | 500 | ~20 | ~10,000 | 96% | Small social network |
| **Medium** | 1,000 | ~20 | ~20,000 | 98% | Medium social network |
| **Large** | 1,500 | ~30 | ~45,000 | 98% | Large social network |

## Quick Start

```bash
cd gnn_benchmark_comparison

# 1. Generate graph adjacency matrices
python generate_graph_data.py

# 2. Run benchmark comparison
python gnn_benchmark.py
```

## What Gets Measured

- **Execution time**: Sparse (CSR×CSR) vs Dense (numpy) multiplication
- **Memory usage**: Sparse storage vs dense arrays
- **Speedup**: How much faster sparse is for graph operations

## Results

For graph adjacency matrices (96-98% sparse):
- **Small**: Sparse 86× faster
- **Medium**: Sparse 403× faster  
- **Large**: Sparse 394× faster

**Key Finding**: Sparse dominates for all graph sizes with 6-13× less memory.

## GNN Context

Graph Neural Networks use matrix multiplication for:
- **Message passing** between nodes
- **Feature aggregation** from neighbors
- **Graph convolutions** (A × X operations)

Real-world graphs are extremely sparse, making sparse representations essential.

## Benchmark Output

Results saved to `benchmarks/`:
- `gnn_results.json` - Detailed metrics
- `gnn_results.txt` - Human-readable table
- `gnn_results.csv` - Spreadsheet format

## Real-World Applications

- Social Networks (Facebook, Twitter)
- Knowledge Graphs (Wikipedia)
- Molecular Structures (Drug discovery)
- Recommendation Systems
- Traffic Networks
