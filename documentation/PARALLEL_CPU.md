# Parallel CPU Operations

## Parallel Addition

**File**: `sparse_addition_parallel.py`

**What it does**: Adds sparse matrices using multiple CPU cores for faster processing.

**Parallelization strategy**:
1. Divides input into chunks (default 100K entries per chunk)
2. Processes chunk pairs in parallel using `multiprocessing.Pool`
3. Each worker merges its chunk pair independently
4. Final step merges all chunk results

**Configuration**:
- Default: Uses all available CPU cores
- Adjustable chunk size for memory control
- Reports timing breakdown (counting, parallel processing, merging)

**Performance**: ~90% time spent in parallel processing phase

---

## Parallel Multiplication

**File**: `sparse_multiplication_parallel.py`

**What it does**: Multiplies sparse matrices using multi-core CPU parallelization.

**Parallelization strategy**:
1. Converts A → CSR, B → CSC (done once)
2. Divides matrix A into row blocks (default 1000 rows per block)
3. Each worker processes multiple blocks independently
4. Workers compute partial results in parallel
5. Final merge combines all block results

**Configuration**:
- Default: Uses all available CPU cores (8 workers)
- Block size: 1000 rows (adjustable)
- Total blocks: 50 for 50,000×50,000 matrix

**Performance**: 
- ~87% time in parallel computation
- ~8% in loading/conversion
- ~5% in merging results

**Speedup**: Near-linear with number of cores for large matrices

---

## Why Parallel is Faster

**Sequential bottleneck**: Processing 50K rows one-by-one is slow

**Parallel benefit**: 
- 8 workers process rows simultaneously
- Each worker independently computes row×column dot products
- Minimal synchronization needed (only final merge)

**Best for**: Large matrices (>10K rows) with many non-zero entries

---

## Benchmark Reports

Both parallel operations generate reports:
- **JSON file**: Machine-readable metrics for analysis
- **TXT file**: Human-readable summary with timing breakdown

**Metrics tracked**:
- Total execution time
- Time per phase (load, compute, merge)
- Throughput (entries/second)
- Parallel efficiency percentage
- Configuration (cores used, chunk/block size)
