# GNN Graph Benchmarks

## Purpose

Benchmark graph operations for Graph Neural Networks:
1. **Static Graphs**: CPU sparse vs dense
2. **Dynamic Graphs**: Incremental edge updates vs full recomputation
3. **GPU Comparison**: GPU dense vs CPU sparse at super sparse levels (90-99.9%)

## Execution

```bash
# Static graph benchmarks (CPU sparse vs dense)
python gnn_benchmark.py

# Dynamic graph updates (1, 2, 3 edge additions)
python gnn_benchmark_dynamic.py

# GPU vs CPU comparison
python gnn_benchmark_gpu.py
```

## Test Configuration

### Static Graphs
- Graph sizes: 500, 1000, 1500 nodes
- Sparsity: ~98% (typical for social networks)
- Iterations: 3 runs per graph
- Operation: Adjacency matrix multiplication (CSR×CSR vs dense)

### Dynamic Graphs
- Graph size: 1000 nodes
- Sparsity: 90%, 99%, 99.9%
- New edges: 1, 2, 3 (simulating friend additions)
- Comparison: Full recomputation vs incremental update (LIL→CSR)

### GPU Comparison
- Graph sizes: 1000, 2000 nodes
- Sparsity: 90%, 99%, 99.9%
- Comparison: GPU dense (PyTorch) vs CPU sparse (SciPy)

## Results

Results in `benchmarks/`:
- `gnn_results.*` - Static graph benchmarks
- `dynamic_graph_results.*` - Dynamic update benchmarks
- `gnn_gpu_results.*` - GPU vs CPU comparison

### Static Graph Performance

| Graph | Nodes | Edges | Sparse Time | Dense Time | Speedup |
|-------|-------|-------|-------------|------------|---------|
| Small | 500 | 9,799 | 3.94ms | 147.26ms | 37.37× |
| Medium | 1,000 | 19,799 | 5.99ms | 1247.26ms | 208.37× |
| Large | 1,500 | 44,537 | 23.93ms | 7425.09ms | 310.32× |

### Dynamic Graph Updates

**Incremental Edge Addition vs Full Recomputation**

| Density | Sparsity | New Edges | Full Recomp | Incremental | Speedup | Winner |
|---------|----------|-----------|-------------|-------------|---------|--------|
| 10% | 90% | 1 | 70.2ms | 7.7ms | **9.1×** | Incremental |
| 10% | 90% | 2 | 70.9ms | 5.9ms | **12.1×** | Incremental |
| 10% | 90% | 3 | 68.7ms | 5.5ms | **12.5×** | Incremental |
| 1% | 99% | 1 | 6.8ms | 1.7ms | **4.0×** | Incremental |
| 1% | 99% | 2 | 7.9ms | 1.6ms | **4.9×** | Incremental |
| 1% | 99% | 3 | 6.8ms | 2.4ms | **2.8×** | Incremental |
| 0.1% | 99.9% | 1 | 0.98ms | 1.66ms | 0.59× | Full Recomp |
| 0.1% | 99.9% | 2 | 0.91ms | 1.57ms | 0.58× | Full Recomp |
| 0.1% | 99.9% | 3 | 1.22ms | 1.36ms | 0.90× | Full Recomp |

**Key Insight**: Incremental updates (LIL→CSR) win at 90-99% sparsity. At 99.9%, format conversion overhead makes full recomputation faster.

### GPU vs CPU Performance

| Nodes | Density | Sparsity | Non-Zeros | CPU Sparse | GPU Dense | Speedup | Winner |
|-------|---------|----------|-----------|------------|-----------|---------|--------|
| 1,000 | 10% | 90% | 95,178 | 0.058s | 0.002s | **27.6×** | GPU |
| 1,000 | 1% | 99% | 9,954 | 0.002s | 0.002s | **1.0×** | GPU |
| 1,000 | 0.1% | 99.9% | 1,000 | 0.0002s | 0.002s | 0.15× | **CPU** |
| 2,000 | 10% | 90% | 380,453 | 0.317s | 0.010s | **33.3×** | GPU |
| 2,000 | 1% | 99% | 39,802 | 0.013s | 0.010s | **1.3×** | GPU |

**Key Insight**: GPU dominates at 90-99% sparsity. CPU sparse wins at extreme sparsity (99.9%).

## Analysis

### Static Graphs
Sparse representations provide 37-310× speedup for graph operations. Performance advantage increases with graph size.

### Dynamic Graphs
- **Use incremental updates** for social networks (90% sparse): 3-12× faster than rebuilding
- **Use full recomputation** for molecular graphs (99.9% sparse): format conversion overhead dominates

### GPU Acceleration
- **GPU optimal** for dense graphs (90% sparse): 27-33× faster
- **CPU optimal** for ultra-sparse graphs (99.9% sparse): minimal computation negates GPU parallelism benefit
- **Crossover point**: ~1,000 non-zeros

### Practical Recommendations
- **Social networks** (90% sparse, frequent updates): Incremental updates on GPU
- **Citation graphs** (99% sparse, occasional updates): Incremental updates on CPU sparse
- **Molecular structures** (99.9% sparse, static): Full recomputation on CPU sparse
