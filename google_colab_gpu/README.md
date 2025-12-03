# GPU Benchmarks

## Purpose

Compare GPU (PyTorch/CUDA) vs CPU sparse matrix performance at super sparse levels (90-99.9%, ≤10% density) to identify optimal hardware.

## Execution

```bash
python run_gpu_sparsity_torch.py   # Sparsity tests (90%, 99%, 99.9%)
python run_gpu_gnn_torch.py        # GNN tests
python compare_results.py           # Generate comparison
```

## Test Configuration

- **GPU**: NVIDIA RTX 5070 Ti (5888 CUDA cores, 12GB VRAM)
- **CPU**: AMD Ryzen 9 8940HX (16 cores)
- **Framework**: PyTorch 2.6.0 + CUDA 12.4
- **Sparsity Levels**: 90%, 99%, 99.9% (super sparse: ≤10% density)

## Results

Results in `results/`:
- `gpu_sparsity_results.json` - GPU sparsity data
- `gpu_gnn_results.json` - GPU GNN data  
- Console output from `compare_results.py` shows full analysis

### GPU Sparsity Tests (Dense GPU Only)

| Sparsity | Non-Zeros | GPU Time | Consistency |
|----------|-----------|----------|-------------|
| 90% | 100,000 | 1.98ms ± 0.99ms | ✓ |
| 99% | 10,000 | 2.04ms ± 0.70ms | ✓ |
| 99.9% | 999 | 2.37ms ± 0.48ms | ✓ |

**Result**: GPU performance stable at ~2ms regardless of sparsity. Dense operations don't benefit from sparsity.

### GNN: Dense GPU vs Sparse CPU

| Graph | Nodes | Sparse CPU | Dense GPU | Winner | Speedup |
|-------|-------|------------|-----------|--------|---------|
| Small | 500 | 3.94ms | 0.54ms | GPU | 7.3× |
| Medium | 1,000 | 5.99ms | 1.90ms | GPU | 3.2× |
| Large | 1,500 | 23.93ms | 6.26ms | GPU | 3.8× |

**Result**: GPU wins for traditional GNN graphs (~98% sparse).

### Super Sparse Comparison (from GNN GPU benchmark)

| Nodes | Density | Sparsity | Sparse CPU | Dense GPU | Winner | Speedup |
|-------|---------|----------|------------|-----------|--------|---------|
| 1,000 | 10% | 90% | 0.058s | 0.002s | GPU | **27.6×** |
| 1,000 | 1% | 99% | 0.002s | 0.002s | GPU | **1.0×** |
| 1,000 | 0.1% | 99.9% | 0.0002s | 0.002s | **CPU** | 0.15× |
| 2,000 | 10% | 90% | 0.317s | 0.010s | GPU | **33.3×** |
| 2,000 | 1% | 99% | 0.013s | 0.010s | GPU | **1.3×** |

**Result**: GPU optimal for 90-99% sparsity. CPU sparse wins at ≥99.9% sparsity.

## Analysis

### Performance Characteristics

1. **GPU dense performance constant** (~2ms) regardless of sparsity
2. **CPU sparse performance scales** with number of non-zeros
3. **Crossover point**: ~1,000-10,000 non-zeros (99-99.9% sparsity for 1000×1000 matrices)

### When to Use GPU

- **90% sparse** (100k non-zeros): GPU 27-33× faster
- **99% sparse** (10k non-zeros): GPU marginally faster (1-1.3×)
- **Social networks, knowledge graphs**: GPU provides consistent speedup

### When to Use CPU Sparse

- **99.9% sparse** (<1k non-zeros): CPU 6.7× faster
- **Molecular structures, citation graphs**: Sparse operations dominate
- **Extreme sparsity**: GPU parallelism overhead exceeds benefit

### Hardware Utilization

- All 46 streaming multiprocessors active
- No compute partitioning
- ECC memory: Not available (consumer GPU)
- Fault tolerance: Fail-fast (no graceful degradation)
- **CUDA Compatibility**: sm_120 (Blackwell) exceeds PyTorch support (sm_90 max). Results valid but hardware underutilized.

## Practical Recommendations

- **Dense graphs** (90% sparse): GPU mandatory (27-33× speedup)
- **Moderate sparsity** (99% sparse): GPU marginally better, consider data transfer overhead
- **Ultra-sparse** (99.9% sparse): CPU sparse mandatory (6.7× faster)
- **Production systems**: Upgrade PyTorch for full sm_120 support
