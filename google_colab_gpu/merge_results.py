"""
Merge CPU and GPU benchmark results into final comparison tables.

Run this AFTER you get GPU results from Google Colab.

Usage:
    python merge_results.py
    
Input files (must exist):
    - google_colab_gpu/results/gpu_sparsity_results.json
    - google_colab_gpu/results/gpu_gnn_results.json
    - dense_baseline_comparison/benchmarks/sparsity_comparison.json
    - gnn_benchmark_comparison/benchmarks/gnn_results.json

Output:
    - final_comparison_sparsity.txt
    - final_comparison_gnn.txt
    - final_comparison_sparsity.csv
    - final_comparison_gnn.csv
"""

import json
import csv
from tabulate import tabulate

def load_json(filepath):
    """Load JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {filepath} not found!")
        return None

def merge_sparsity_results():
    """Merge sparsity comparison results (CPU + GPU)."""
    
    print("="*70)
    print("MERGING SPARSITY COMPARISON RESULTS")
    print("="*70)
    
    # Load CPU results
    cpu_file = "dense_baseline_comparison/benchmarks/sparsity_comparison.json"
    cpu_data = load_json(cpu_file)
    
    # Load GPU results
    gpu_file = "google_colab_gpu/results/gpu_sparsity_results.json"
    gpu_data = load_json(gpu_file)
    
    if not cpu_data or not gpu_data:
        print("ERROR: Missing result files!")
        return
    
    # Merge results
    merged = []
    for cpu_result in cpu_data:
        sparsity = cpu_result['sparsity_percent']
        
        # Find matching GPU result
        gpu_result = next((r for r in gpu_data['results'] if r['sparsity_percent'] == sparsity), None)
        
        if gpu_result:
            sparse_cpu_time = cpu_result['sparse_time']
            dense_cpu_time = cpu_result['dense_time']
            dense_gpu_time = gpu_result['gpu_time']
            
            # Determine winner
            best_time = min(sparse_cpu_time, dense_cpu_time, dense_gpu_time)
            if best_time == sparse_cpu_time:
                winner = "Sparse CPU"
            elif best_time == dense_gpu_time:
                winner = "Dense GPU"
            else:
                winner = "Dense CPU"
            
            merged.append({
                "sparsity": f"{sparsity}%",
                "sparse_cpu": f"{sparse_cpu_time:.6f}s",
                "dense_cpu": f"{dense_cpu_time:.6f}s",
                "dense_gpu": f"{dense_gpu_time:.6f}s",
                "winner": winner,
                "gpu_vs_sparse": f"{dense_gpu_time / sparse_cpu_time:.2f}×"
            })
    
    # Create table
    headers = ["Sparsity", "Sparse CPU", "Dense CPU", "Dense GPU", "Winner", "GPU/Sparse Ratio"]
    table_data = [[m["sparsity"], m["sparse_cpu"], m["dense_cpu"], m["dense_gpu"], m["winner"], m["gpu_vs_sparse"]] for m in merged]
    
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    print("\n" + table)
    
    # Save to file
    with open("google_colab_gpu/results/final_comparison_sparsity.txt", "w", encoding='utf-8') as f:
        f.write("FINAL COMPARISON: Sparse CPU vs Dense CPU vs Dense GPU\n")
        f.write("="*70 + "\n\n")
        f.write(table + "\n\n")
        f.write(f"GPU Device: {gpu_data.get('gpu_device', 'Unknown')}\n")
    
    # Save CSV
    with open("google_colab_gpu/results/final_comparison_sparsity.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["sparsity", "sparse_cpu", "dense_cpu", "dense_gpu", "winner", "gpu_vs_sparse"])
        writer.writeheader()
        writer.writerows(merged)
    
    print("\n✓ Saved: final_comparison_sparsity.txt")
    print("✓ Saved: final_comparison_sparsity.csv")

def merge_gnn_results():
    """Merge GNN benchmark results (CPU + GPU)."""
    
    print("\n\n" + "="*70)
    print("MERGING GNN BENCHMARK RESULTS")
    print("="*70)
    
    # Load CPU results
    cpu_file = "gnn_benchmark_comparison/benchmarks/gnn_results.json"
    cpu_data = load_json(cpu_file)
    
    # Load GPU results
    gpu_file = "google_colab_gpu/results/gpu_gnn_results.json"
    gpu_data = load_json(gpu_file)
    
    if not cpu_data or not gpu_data:
        print("ERROR: Missing result files!")
        return
    
    # Merge results
    merged = []
    for cpu_result in cpu_data:
        graph_name = cpu_result['graph_name']
        
        # Find matching GPU result
        gpu_result = next((r for r in gpu_data['results'] if r['graph_name'] == graph_name), None)
        
        if gpu_result:
            sparse_cpu_time = cpu_result['sparse_time']
            dense_cpu_time = cpu_result['dense_time']
            dense_gpu_time = gpu_result['gpu_time']
            
            # Determine winner
            best_time = min(sparse_cpu_time, dense_cpu_time, dense_gpu_time)
            if best_time == sparse_cpu_time:
                winner = "Sparse CPU"
            elif best_time == dense_gpu_time:
                winner = "Dense GPU"
            else:
                winner = "Dense CPU"
            
            merged.append({
                "graph": f"{graph_name} ({cpu_result['num_nodes']} nodes)",
                "sparse_cpu": f"{sparse_cpu_time:.6f}s",
                "dense_cpu": f"{dense_cpu_time:.6f}s",
                "dense_gpu": f"{dense_gpu_time:.6f}s",
                "winner": winner,
                "gpu_vs_sparse": f"{dense_gpu_time / sparse_cpu_time:.2f}×"
            })
    
    # Create table
    headers = ["Graph", "Sparse CPU", "Dense CPU", "Dense GPU", "Winner", "GPU/Sparse Ratio"]
    table_data = [[m["graph"], m["sparse_cpu"], m["dense_cpu"], m["dense_gpu"], m["winner"], m["gpu_vs_sparse"]] for m in merged]
    
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    
    print("\n" + table)
    
    # Save to file
    with open("google_colab_gpu/results/final_comparison_gnn.txt", "w", encoding='utf-8') as f:
        f.write("FINAL COMPARISON: GNN Graphs - Sparse CPU vs Dense CPU vs Dense GPU\n")
        f.write("="*70 + "\n\n")
        f.write(table + "\n\n")
        f.write(f"GPU Device: {gpu_data.get('gpu_device', 'Unknown')}\n")
    
    # Save CSV
    with open("google_colab_gpu/results/final_comparison_gnn.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["graph", "sparse_cpu", "dense_cpu", "dense_gpu", "winner", "gpu_vs_sparse"])
        writer.writeheader()
        writer.writerows(merged)
    
    print("\n✓ Saved: final_comparison_gnn.txt")
    print("✓ Saved: final_comparison_gnn.csv")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CPU + GPU RESULTS MERGER")
    print("="*70)
    print()
    
    merge_sparsity_results()
    merge_gnn_results()
    
    print("\n" + "="*70)
    print("MERGE COMPLETE!")
    print("="*70)
    print("\nFinal comparison tables saved to google_colab_gpu/results/")
