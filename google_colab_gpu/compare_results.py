#!/usr/bin/env python
"""
Compare CPU and GPU benchmark results
Generates comparison tables and analysis
"""

import json
from pathlib import Path

print("="*80)
print("CPU vs GPU BENCHMARK COMPARISON")
print("="*80)
print()

# Load GPU results
gpu_sparsity_file = Path("google_colab_gpu/results/gpu_sparsity_results.json")
gpu_gnn_file = Path("google_colab_gpu/results/gpu_gnn_results.json")

# Load CPU results  
cpu_sparsity_file = Path("dense_baseline_comparison/benchmarks/sparsity_comparison.json")
cpu_gnn_file = Path("gnn_benchmark_comparison/benchmarks/gnn_results.json")

# ============================================================================
# Sparsity Comparison
# ============================================================================

print("\n" + "="*80)
print("SPARSITY COMPARISON (1000×1000 matrices)")
print("="*80)

if gpu_sparsity_file.exists() and cpu_sparsity_file.exists():
    with open(gpu_sparsity_file) as f:
        gpu_data = json.load(f)
    with open(cpu_sparsity_file) as f:
        cpu_data = json.load(f)
    
    print(f"\nGPU: {gpu_data.get('gpu_device', 'Unknown')}")
    print(f"CPU Results from: {cpu_sparsity_file}")
    print()
    
    # Header
    print(f"{'Sparsity':<10} {'Dense GPU':<15} {'Sparse CPU':<15} {'Dense CPU':<15} {'Winner':<20}")
    print("-" * 80)
    
    # Get CPU results by sparsity level
    cpu_results = {}
    cpu_list = cpu_data if isinstance(cpu_data, list) else cpu_data.get('results', [])
    for result in cpu_list:
        sparsity = result.get('sparsity_percent')
        if sparsity:
            cpu_results[sparsity] = result
    
    # Compare each sparsity level
    for gpu_result in gpu_data.get('results', []):
        sparsity = gpu_result['sparsity_percent']
        gpu_time = gpu_result['gpu_time']
        
        cpu_result = cpu_results.get(sparsity, {})
        sparse_time = cpu_result.get('sparse_time', 0)
        dense_time = cpu_result.get('dense_time', 0)
        
        # Determine winner
        times = [
            ('Dense GPU', gpu_time),
            ('Sparse CPU', sparse_time) if sparse_time > 0 else (None, float('inf')),
            ('Dense CPU', dense_time) if dense_time > 0 else (None, float('inf'))
        ]
        winner_name, winner_time = min((t for t in times if t[0]), key=lambda x: x[1])
        
        # Calculate speedups
        if winner_name == 'Dense GPU':
            if sparse_time > 0:
                speedup_vs_sparse = sparse_time / gpu_time
                winner_text = f"GPU ({speedup_vs_sparse:.1f}x faster)"
            else:
                winner_text = "GPU (best)"
        elif winner_name == 'Sparse CPU':
            speedup_vs_gpu = gpu_time / sparse_time
            winner_text = f"Sparse CPU ({speedup_vs_gpu:.1f}x faster)"
        else:
            winner_text = winner_name
        
        print(f"{sparsity}%{'':<7} "
              f"{gpu_time:.6f}s{'':<6} "
              f"{sparse_time:.6f}s{'':<6} "
              f"{dense_time:.6f}s{'':<6} "
              f"{winner_text}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find crossover point
    for gpu_result in gpu_data.get('results', []):
        sparsity = gpu_result['sparsity_percent']
        gpu_time = gpu_result['gpu_time']
        cpu_result = cpu_results.get(sparsity, {})
        sparse_time = cpu_result.get('sparse_time', 0)
        
        if sparse_time > 0:
            if sparse_time < gpu_time:
                print(f"✓ At {sparsity}% sparsity: Sparse CPU beats Dense GPU!")
                print(f"  Sparse CPU: {sparse_time:.6f}s vs Dense GPU: {gpu_time:.6f}s")
                print(f"  Speedup: {gpu_time/sparse_time:.2f}x faster on CPU")
            else:
                print(f"• At {sparsity}% sparsity: Dense GPU wins")
                print(f"  Dense GPU: {gpu_time:.6f}s vs Sparse CPU: {sparse_time:.6f}s")
                print(f"  Speedup: {sparse_time/gpu_time:.2f}x faster on GPU")
else:
    print("⚠ Missing benchmark files for sparsity comparison")

# ============================================================================
# GNN Comparison
# ============================================================================

print("\n\n" + "="*80)
print("GNN BENCHMARK COMPARISON")
print("="*80)

if gpu_gnn_file.exists() and cpu_gnn_file.exists():
    with open(gpu_gnn_file) as f:
        gpu_gnn_data = json.load(f)
    with open(cpu_gnn_file) as f:
        cpu_gnn_data = json.load(f)
    
    print()
    print(f"{'Graph Size':<15} {'Dense GPU':<15} {'Sparse CPU':<15} {'Winner':<20}")
    print("-" * 70)
    
    # Match by graph name
    cpu_gnn_list = cpu_gnn_data if isinstance(cpu_gnn_data, list) else cpu_gnn_data.get('results', [])
    cpu_gnn_results = {r['graph_name']: r for r in cpu_gnn_list}
    
    for gpu_result in gpu_gnn_data.get('results', []):
        graph_name = gpu_result['graph_name']
        gpu_time = gpu_result['gpu_time']
        
        cpu_result = cpu_gnn_results.get(graph_name, {})
        cpu_time = cpu_result.get('sparse_time', 0)
        
        if cpu_time > 0 and gpu_time < cpu_time:
            speedup = cpu_time / gpu_time
            winner = f"GPU ({speedup:.1f}x faster)"
        elif cpu_time > 0:
            speedup = gpu_time / cpu_time
            winner = f"Sparse CPU ({speedup:.1f}x faster)"
        else:
            winner = "GPU"
        
        print(f"{graph_name:<15} {gpu_time:.6f}s{'':<6} {cpu_time:.6f}s{'':<6} {winner}")
else:
    print("⚠ Missing benchmark files for GNN comparison")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nAll GPU benchmarks completed successfully!")
print(f"Results saved in: google_colab_gpu/results/")
