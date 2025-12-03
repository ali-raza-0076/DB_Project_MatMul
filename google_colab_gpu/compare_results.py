"""
Compare CPU vs GPU benchmark results and generate comprehensive analysis.
"""
import json
import os

def load_results(filepath):
    """Load benchmark results from JSON file."""
    if not os.path.exists(filepath):
        print(f"⚠ Warning: File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    print("="*80)
    print("CPU vs GPU BENCHMARK COMPARISON")
    print("="*80)
    
    # Load results
    cpu_sparse = load_results("dense_baseline_comparison/benchmarks/sparsity_comparison.json")
    gpu_sparsity = load_results("google_colab_gpu/results/gpu_sparsity_results.json")
    cpu_gnn = load_results("gnn_benchmark_comparison/benchmarks/gnn_results.json")
    gpu_gnn = load_results("google_colab_gpu/results/gpu_gnn_results.json")
    
    # Sparsity Comparison
    if cpu_sparse and gpu_sparsity:
        print("\n" + "="*80)
        print("SPARSITY COMPARISON (1000×1000 matrices)")
        print("="*80)
        print("\nGPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU")
        print(f"CPU Results from: dense_baseline_comparison\\benchmarks\\sparsity_comparison.json")
        
        print(f"\n{'Sparsity':<10} {'Dense GPU':<15} {'Sparse CPU':<15} {'Dense CPU':<15} {'Winner'}")
        print("-"*80)
        
        # Create lookup dictionaries
        cpu_dict = {r['sparsity_percent']: r for r in cpu_sparse}
        gpu_dict = {r['sparsity_percent']: r for r in gpu_sparsity}
        
        for sparsity in [50, 90, 95, 99]:
            if sparsity in cpu_dict and sparsity in gpu_dict:
                cpu = cpu_dict[sparsity]
                gpu = gpu_dict[sparsity]
                
                gpu_time = gpu['gpu_time']
                sparse_cpu_time = cpu['sparse_time']
                dense_cpu_time = cpu['dense_time']
                
                # Determine winner
                if gpu_time < sparse_cpu_time:
                    speedup = sparse_cpu_time / gpu_time
                    winner = f"GPU ({speedup:.1f}x faster)"
                else:
                    speedup = gpu_time / sparse_cpu_time
                    winner = f"Sparse CPU ({speedup:.1f}x faster)"
                
                print(f"{sparsity}%{'':<6} {gpu_time:.6f}s{'':<4} {sparse_cpu_time:.6f}s{'':<4} {dense_cpu_time:.6f}s{'':<4} {winner}")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        for sparsity in [50, 90, 95, 99]:
            if sparsity in cpu_dict and sparsity in gpu_dict:
                cpu = cpu_dict[sparsity]
                gpu = gpu_dict[sparsity]
                gpu_time = gpu['gpu_time']
                sparse_cpu_time = cpu['sparse_time']
                dense_cpu_time = cpu['dense_time']
                
                if gpu_time < sparse_cpu_time:
                    speedup = sparse_cpu_time / gpu_time
                    print(f"• At {sparsity}% sparsity: Dense GPU wins")
                    print(f"  Dense GPU: {gpu_time:.6f}s vs Sparse CPU: {sparse_cpu_time:.6f}s")
                    print(f"  Speedup: {speedup:.2f}x faster on GPU")
                else:
                    speedup = gpu_time / sparse_cpu_time
                    print(f"✓ At {sparsity}% sparsity: Sparse CPU beats Dense GPU!")
                    print(f"  Sparse CPU: {sparse_cpu_time:.6f}s vs Dense GPU: {gpu_time:.6f}s")
                    print(f"  Speedup: {speedup:.2f}x faster on CPU")
    
    # GNN Comparison
    if cpu_gnn and gpu_gnn:
        print("\n\n" + "="*80)
        print("GNN BENCHMARK COMPARISON")
        print("="*80)
        
        print(f"\n{'Graph Size':<15} {'Dense GPU':<15} {'Sparse CPU':<15} {'Winner'}")
        print("-"*70)
        
        graph_names = ["Small", "Medium", "Large"]
        cpu_gnn_dict = {r['graph_name']: r for r in cpu_gnn}
        gpu_gnn_dict = {r['graph_name']: r for r in gpu_gnn}
        
        for graph_name in graph_names:
            if graph_name in cpu_gnn_dict and graph_name in gpu_gnn_dict:
                cpu = cpu_gnn_dict[graph_name]
                gpu = gpu_gnn_dict[graph_name]
                
                gpu_time = gpu['gpu_time']
                cpu_time = cpu['sparse_time']
                
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    winner = f"GPU ({speedup:.1f}x faster)"
                else:
                    speedup = gpu_time / cpu_time
                    winner = f"CPU ({speedup:.1f}x faster)"
                
                print(f"{graph_name:<15} {gpu_time:.6f}s{'':<4} {cpu_time:.6f}s{'':<4} {winner}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll GPU benchmarks completed successfully!")
    print("Results saved in: google_colab_gpu/results/")

if __name__ == "__main__":
    main()
