# Dynamic Graph Neural Network Benchmark Methodology

## 1. Experimental Setup

### 1.1 Hardware Configuration
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop (Blackwell architecture)
  - CUDA Cores: 5888
  - Memory: 12 GB GDDR7
  - Compute Capability: sm_120
- **CUDA Version**: 13.0
- **Framework**: PyTorch 2.6.0+cu130

### 1.2 Software Environment
- Python 3.13.0
- PyTorch with CUDA acceleration enabled
- Graph data stored in CSV format (edge list representation)

## 2. Graph Data Generation

### 2.1 Graph Characteristics
We generated synthetic sparse graphs with the following parameters:
- **Node counts**: 4,000 | 8,000 | 10,000 vertices
- **Sparsity levels**: 90% | 95% | 99% | 99.9%

### 2.2 Sparsity Definition
Sparsity percentage represents the proportion of zero entries in the complete adjacency matrix:
```
Sparsity = (1 - |E| / |V|²) × 100%
```
where |E| is the number of edges and |V| is the number of nodes.

For example:
- 90% sparsity → 10% density → 1.6M edges for 4K nodes
- 99.9% sparsity → 0.1% density → 16K edges for 4K nodes

### 2.3 Edge Generation Algorithm
To generate graphs efficiently, we implemented a **vectorized sampling approach**:

1. **Calculate target edge count**:
   ```
   num_edges = (|V|² × (1 - sparsity/100))
   ```

2. **Batch generation with oversampling**:
   ```python
   buffer_factor = 1.05  # 5% extra to account for duplicates
   samples_needed = num_edges × buffer_factor
   ```

3. **Vectorized random sampling**:
   ```python
   source_vertices = np.random.randint(0, num_nodes, size=samples_needed)
   target_vertices = np.random.randint(0, num_nodes, size=samples_needed)
   ```

4. **Self-loop removal**:
   Filter edges where `source == target`

5. **Direct file writing**:
   Stream edges to CSV without storing full edge set in memory

**Rationale**: This approach achieves O(|E|) time complexity and O(1) space complexity by:
- Avoiding explicit duplicate detection (which requires O(|E|) space for large graphs)
- Accepting ~1% duplicate edges as a trade-off for memory efficiency
- Using vectorized NumPy operations instead of Python loops

## 3. Graph Neural Network Operations

### 3.1 GCN Forward Pass
We implement a simplified Graph Convolutional Network (GCN) aggregation:

```
H' = AH W
```

where:
- **H**: Node feature matrix (N × D)
- **A**: Adjacency matrix (N × N)
- **W**: Weight matrix (D × D)
- **H'**: Output features (N × D)

### 3.2 GPU Implementation Details

#### Sparse Representation
Graphs stored in **COO (Coordinate) format**:
```python
rows = torch.tensor([src_nodes], device='cuda')
cols = torch.tensor([dst_nodes], device='cuda')
```

#### Aggregation Operation
Using PyTorch's `index_add_` for efficient neighbor aggregation:
```python
aggregated = torch.zeros(num_nodes, feature_dim, device='cuda')
aggregated.index_add_(0, rows, features[cols])
```

**Complexity**: O(|E|) where |E| is the number of edges

#### GPU Synchronization
Accurate timing requires explicit GPU synchronization:
```python
torch.cuda.synchronize()  # Wait for all GPU kernels to complete
```

## 4. Dynamic Update Scenarios

### 4.1 Problem Definition
Evaluate two approaches for handling graph structure changes:

1. **Full Recomputation**:
   - Recompute entire aggregation after adding edges
   - Complexity: O(|E_old| + |E_new|)

2. **Incremental Update**:
   - Update only affected aggregations
   - Complexity: O(|E_new|) where |E_new| << |E_old|

### 4.2 Dynamic Update Protocol

For each test configuration:

1. **Initialize graph**: Load adjacency list and node features

2. **Precompute baseline aggregation**:
   ```python
   aggregated = aggregate_neighbors(graph, features)
   ```

3. **Generate dynamic edges**:
   - Sample 3 random edges not in original graph
   - Ensure edges don't create self-loops
   - Verify edges are truly new (not duplicates)

4. **Benchmark full recomputation** (5000 iterations):
   ```python
   for i in range(5000):
       graph_copy = graph.copy()
       graph_copy.add_edges(new_edges)
       result = aggregate_neighbors(graph_copy, features)
   ```

5. **Benchmark incremental update** (5000 iterations):
   ```python
   for i in range(5000):
       aggregated_copy = aggregated.copy()
       for (src, dst) in new_edges:
           aggregated_copy[src] += features[dst]
       result = aggregated_copy
   ```

### 4.3 Iteration Count Justification
**5000 iterations per test** chosen to:
- Achieve statistically significant timing measurements (>10 seconds total)
- Amortize kernel launch overhead
- Demonstrate sustained performance under repeated operations
- Provide meaningful execution times for presentation

## 5. Performance Metrics

### 5.1 Measured Quantities

1. **Per-iteration execution time** (milliseconds):
   ```
   time_per_iteration = total_time / num_iterations
   ```

2. **Speedup ratio**:
   ```
   speedup = time_full_recomp / time_incremental
   ```

### 5.2 Statistical Approach
- Report **mean execution time** over 5000 runs
- Progress updates every 5 iterations during execution
- GPU timing via CUDA events (not wall-clock time)

## 6. Experimental Design Rationale

### 6.1 Why These Graph Sizes?
- **4K nodes**: Baseline representative of small-scale networks
- **8K nodes**: Mid-range to observe scaling behavior
- **10K nodes**: Upper bound for reasonable iteration counts (5000 × 10K = 50M operations)

### 6.2 Why These Sparsity Levels?
- **90%**: Dense case - validates performance with substantial edge counts
- **95%**: Moderate sparsity - typical for citation networks
- **99%**: High sparsity - representative of social networks (Twitter, LinkedIn)
- **99.9%**: Extreme sparsity - demonstrates scalability to very sparse graphs

### 6.3 Why Remove Early Stopping?
Early stopping was **removed** to ensure:
- Complete evaluation across all configurations
- Fair comparison between different sparsity levels
- Comprehensive results for all graph sizes
- No premature termination that could bias results

### 6.4 Why GPU-Only Testing?
Dynamic graph operations are **compute-intensive** and benefit from:
- Massive parallelism (5888 CUDA cores)
- High memory bandwidth (GDDR7)
- Optimized tensor operations (PyTorch CUDA backend)

CPU testing would be orders of magnitude slower and less relevant for production GNN deployments.

## 7. Expected Outcomes

### 7.1 Hypotheses
1. **Incremental updates should be faster than full recomputation** for small edge additions
2. **Speedup should increase with graph density** (more edges = more wasted recomputation)
3. **GPU parallelism should maintain consistent per-iteration times** regardless of iteration count

### 7.2 Key Performance Indicators
- Incremental speedup >5× indicates significant practical benefit
- Execution time scaling with sparsity validates algorithmic complexity
- Sustained performance across 5000 iterations demonstrates production readiness

## 8. Limitations and Assumptions

### 8.1 Simplifications
- Undirected graphs (no edge weights)
- Uniform random edge sampling (not realistic distribution)
- Single GCN layer (no multi-layer composition)
- No batch processing (single graph per test)

### 8.2 Scope
This benchmark focuses on:
- **Graph structure updates** (adding edges)
- **Neighbor aggregation** (core GNN primitive)
- **GPU acceleration** (CUDA-based execution)

Not evaluated:
- Edge deletions or node additions
- Feature updates without structure changes
- Multi-GPU or distributed settings
- Training/backpropagation

## 9. Reproducibility

### 9.1 Data Generation
All graphs use **deterministic seeding**:
```python
seed = num_nodes × 100 + int(sparsity × 10)
np.random.seed(seed)
```

### 9.2 Output Files
- **Graph data**: `../data/graph_{nodes}nodes_{sparsity}pct_sparsity.csv`
- **Results**: `dynamic_gpu_results.json`
- **Summary**: `dynamic_gpu_summary.txt`

### 9.3 Environment Requirements
```
torch>=2.6.0
numpy>=1.24.0
CUDA>=13.0
GPU with compute capability >=8.0
```

---

## Summary for Presentation

**What we did**:
1. Generated sparse graphs (4K-10K nodes, 90-99.9% sparsity) using vectorized sampling
2. Implemented GPU-accelerated GCN aggregation using PyTorch CUDA
3. Benchmarked full recomputation vs. incremental updates over 5000 iterations
4. Measured per-iteration execution time and speedup ratios

**Why this approach**:
- Vectorized generation: O(|E|) time, O(1) space - scales to millions of edges
- GPU acceleration: Leverages 5888 CUDA cores for parallel tensor operations
- 5000 iterations: Provides statistically significant timing (10+ seconds)
- Multiple sparsities: Validates performance across realistic graph densities

**Key technical decisions**:
- COO format for sparse representation (memory efficient)
- `index_add_` for neighbor aggregation (optimized PyTorch primitive)
- CUDA synchronization for accurate timing (prevents CPU-GPU timing skew)
- Removed early stopping for comprehensive evaluation (no biased results)
