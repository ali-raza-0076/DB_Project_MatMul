"""
Generate small sparse matrices (1000 entries) for dense baseline comparison.
"""
import numpy as np
import csv
import os

def generate_sparse_matrix(rows, cols, num_entries, seed, output_file):
    """Generate sparse matrix with 1-based indexing."""
    np.random.seed(seed)
    
    # Generate random positions
    row_indices = np.random.randint(0, rows, size=num_entries)
    col_indices = np.random.randint(0, cols, size=num_entries)
    values = np.random.randint(1, 11, size=num_entries)  # Integer values 1-10
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for r, c, v in zip(row_indices, col_indices, values):
            writer.writerow([r + 1, c + 1, v])  # 1-based indexing
    
    print(f"Generated {output_file}: {num_entries} entries")

if __name__ == "__main__":
    size = 1000  # 1000×1000 matrices
    num_entries = 1000
    
    generate_sparse_matrix(size, size, num_entries, seed=42, 
                          output_file="data/matrix_a_small.csv")
    generate_sparse_matrix(size, size, num_entries, seed=123, 
                          output_file="data/matrix_b_small.csv")
    
    print(f"\nMatrices: {size}×{size}, {num_entries} entries each (~0.1% density)")
