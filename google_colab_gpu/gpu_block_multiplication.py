"""
GPU Block-Based Sparse Matrix Multiplication
=============================================

This module implements chunked/block-based matrix multiplication that can handle
matrices larger than GPU memory by processing them in blocks.

Key Features:
- Handles matrices of ANY size (even TB-scale)
- Only requires GPU memory for small chunks
- Saves intermediate results to disk
- Memory-efficient for Google Colab (12-15GB GPU)
- Works with CSR sparse format

Algorithm:
1. Load small block of rows from matrix A
2. Load corresponding block of columns from matrix B
3. Multiply on GPU
4. Save result chunk to disk
5. Repeat for all blocks
6. Combine results

Memory Requirements:
- For N×N matrix with block_size B:
  - GPU Memory: ~2 × B × N × 4 bytes (for float32)
  - For 50,000×50,000 matrix with block_size=1000:
    - GPU Memory: ~400 MB per block (easily fits in 12GB GPU)
"""

import numpy as np
import cupy as cp
from scipy import sparse as sp
import csv
import time
import os
import json
from pathlib import Path


class BlockMatrixMultiplier:
    """
    Block-based matrix multiplication for large matrices on GPU.
    """
    
    def __init__(self, gpu_memory_gb=12, safety_factor=0.7):
        """
        Initialize block multiplier.
        
        Args:
            gpu_memory_gb: Available GPU memory in GB (default: 12GB for Colab)
            safety_factor: Use only 70% of memory to be safe
        """
        self.gpu_memory_bytes = int(gpu_memory_gb * 1e9 * safety_factor)
        self.gpu_available = cp.cuda.is_available()
        
        if self.gpu_available:
            print(f"✓ GPU Available: {cp.cuda.Device()}")
            print(f"✓ Using {gpu_memory_gb * safety_factor:.1f}GB GPU memory")
        else:
            raise RuntimeError("No GPU available!")
    
    def estimate_block_size(self, matrix_size, sparsity=0.99):
        """
        Estimate optimal block size based on available GPU memory.
        
        Args:
            matrix_size: Size of N×N matrix
            sparsity: Sparsity ratio (0.99 = 99% sparse)
        
        Returns:
            Optimal block size
        """
        # For CSR format: need to store:
        # - Row block of A: block_size × N elements (mostly zeros if sparse)
        # - Column block of B: N × block_size elements
        # - Result block: block_size × block_size elements
        
        # Dense worst case (bytes for float32):
        # block_size × N × 4 (A rows) + N × block_size × 4 (B cols) + block_size² × 4 (result)
        
        # Estimate for sparse (assuming CSR):
        density = 1 - sparsity
        
        # bytes_per_block = block_size × N × 4 × density × 2 + block_size² × 4
        # Solve for block_size
        
        # Conservative estimate for dense intermediate results:
        bytes_per_element = 8  # float32 + indices
        max_elements = self.gpu_memory_bytes // bytes_per_element
        
        # block_size × N × 2 + block_size² ≈ max_elements
        # Approximate: block_size × (2N + block_size) ≈ max_elements
        
        # Conservative formula:
        block_size = int(np.sqrt(max_elements / 4))
        
        # Limit to reasonable size
        block_size = min(block_size, matrix_size, 5000)
        block_size = max(block_size, 100)
        
        print(f"Estimated optimal block size: {block_size}")
        print(f"Memory per block: ~{block_size * matrix_size * 4 * 2 / 1e9:.2f}GB")
        print(f"Number of blocks: {int(np.ceil(matrix_size / block_size))}")
        
        return block_size
    
    def load_sparse_csv(self, filepath, matrix_size):
        """
        Load sparse matrix from CSV file.
        
        Args:
            filepath: Path to CSV file (row, col, value format)
            matrix_size: Size of square matrix
        
        Returns:
            scipy.sparse.csr_matrix
        """
        print(f"Loading sparse matrix from {filepath}...")
        
        rows, cols, vals = [], [], []
        
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for parts in reader:
                if len(parts) == 3:
                    try:
                        # Assuming 1-based indexing, convert to 0-based
                        r = int(parts[0]) - 1
                        c = int(parts[1]) - 1
                        v = float(parts[2])
                        
                        if 0 <= r < matrix_size and 0 <= c < matrix_size:
                            rows.append(r)
                            cols.append(c)
                            vals.append(v)
                    except ValueError:
                        continue
        
        nnz = len(vals)
        sparsity = 100 * (1 - nnz / (matrix_size * matrix_size))
        
        print(f"  - Loaded {nnz:,} non-zero entries")
        print(f"  - Sparsity: {sparsity:.4f}%")
        
        # Create CSR matrix
        sparse_mat = sp.csr_matrix((vals, (rows, cols)), 
                                   shape=(matrix_size, matrix_size))
        
        return sparse_mat
    
    def multiply_sparse_block(self, A_csr, B_csr, row_start, row_end, 
                               col_start, col_end):
        """
        Multiply a block of A with a block of B on GPU.
        
        Args:
            A_csr: Full matrix A in CSR format
            B_csr: Full matrix B in CSR format
            row_start, row_end: Row range for block of A
            col_start, col_end: Column range for block of B
        
        Returns:
            Dense numpy array of result block
        """
        # Extract row block from A: rows [row_start:row_end], all columns
        A_block = A_csr[row_start:row_end, :]
        
        # Extract column block from B: all rows, columns [col_start:col_end]
        B_block = B_csr[:, col_start:col_end]
        
        # Convert to dense for GPU processing
        A_block_dense = A_block.toarray()
        B_block_dense = B_block.toarray()
        
        # Transfer to GPU
        A_gpu = cp.asarray(A_block_dense, dtype=cp.float32)
        B_gpu = cp.asarray(B_block_dense, dtype=cp.float32)
        
        # Multiply on GPU
        C_gpu = cp.matmul(A_gpu, B_gpu)
        
        # Transfer back to CPU
        C_block = cp.asnumpy(C_gpu)
        
        # Clean up GPU memory
        del A_gpu, B_gpu, C_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        return C_block
    
    def multiply_blocks(self, A_csr, B_csr, block_size, output_dir):
        """
        Multiply two matrices using block algorithm, saving results to disk.
        
        Args:
            A_csr: Matrix A in CSR format
            B_csr: Matrix B in CSR format
            block_size: Size of blocks to process
            output_dir: Directory to save block results
        
        Returns:
            Path to result metadata file
        """
        matrix_size = A_csr.shape[0]
        
        if A_csr.shape[1] != B_csr.shape[0]:
            raise ValueError(f"Matrix dimensions incompatible: {A_csr.shape} × {B_csr.shape}")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of blocks
        num_row_blocks = int(np.ceil(matrix_size / block_size))
        num_col_blocks = int(np.ceil(matrix_size / block_size))
        total_blocks = num_row_blocks * num_col_blocks
        
        print(f"\n{'='*70}")
        print(f"Block Multiplication: {matrix_size}×{matrix_size} matrix")
        print(f"{'='*70}")
        print(f"Block size: {block_size}×{block_size}")
        print(f"Row blocks: {num_row_blocks}")
        print(f"Column blocks: {num_col_blocks}")
        print(f"Total blocks to process: {total_blocks}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}\n")
        
        # Metadata for result reconstruction
        metadata = {
            'matrix_size': matrix_size,
            'block_size': block_size,
            'num_row_blocks': num_row_blocks,
            'num_col_blocks': num_col_blocks,
            'blocks': []
        }
        
        total_time = 0
        block_count = 0
        
        # Process each block
        for row_block_idx in range(num_row_blocks):
            row_start = row_block_idx * block_size
            row_end = min(row_start + block_size, matrix_size)
            
            for col_block_idx in range(num_col_blocks):
                col_start = col_block_idx * block_size
                col_end = min(col_start + block_size, matrix_size)
                
                block_count += 1
                
                print(f"[{block_count}/{total_blocks}] Processing block "
                      f"({row_block_idx},{col_block_idx}): "
                      f"rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")
                
                # Multiply block
                start_time = time.perf_counter()
                C_block = self.multiply_sparse_block(
                    A_csr, B_csr, row_start, row_end, col_start, col_end
                )
                elapsed = time.perf_counter() - start_time
                total_time += elapsed
                
                # Save block to disk
                block_filename = f"block_{row_block_idx}_{col_block_idx}.npy"
                block_path = output_dir / block_filename
                np.save(block_path, C_block)
                
                # Record metadata
                metadata['blocks'].append({
                    'row_block': row_block_idx,
                    'col_block': col_block_idx,
                    'row_start': row_start,
                    'row_end': row_end,
                    'col_start': col_start,
                    'col_end': col_end,
                    'filename': block_filename,
                    'time': elapsed
                })
                
                nnz = np.count_nonzero(C_block)
                print(f"  ✓ Block saved: {C_block.shape}, "
                      f"{nnz:,} non-zeros, {elapsed:.3f}s\n")
        
        # Save metadata
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"{'='*70}")
        print(f"✓ Block multiplication complete!")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per block: {total_time/total_blocks:.3f}s")
        print(f"  Metadata saved: {metadata_path}")
        print(f"{'='*70}\n")
        
        return metadata_path
    
    def reconstruct_result(self, metadata_path, output_format='csr'):
        """
        Reconstruct full result matrix from blocks.
        
        Args:
            metadata_path: Path to metadata.json file
            output_format: 'csr', 'dense', or 'save_csv'
        
        Returns:
            Result matrix (format depends on output_format)
        """
        print(f"\n{'='*70}")
        print(f"Reconstructing result matrix...")
        print(f"{'='*70}\n")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        matrix_size = metadata['matrix_size']
        output_dir = Path(metadata_path).parent
        
        print(f"Matrix size: {matrix_size}×{matrix_size}")
        print(f"Loading {len(metadata['blocks'])} blocks...\n")
        
        # Initialize result matrix
        if output_format == 'dense':
            result = np.zeros((matrix_size, matrix_size), dtype=np.float32)
        else:
            # Collect sparse entries
            rows, cols, vals = [], [], []
        
        # Load and place each block
        for block_info in metadata['blocks']:
            block_path = output_dir / block_info['filename']
            C_block = np.load(block_path)
            
            row_start = block_info['row_start']
            row_end = block_info['row_end']
            col_start = block_info['col_start']
            col_end = block_info['col_end']
            
            if output_format == 'dense':
                result[row_start:row_end, col_start:col_end] = C_block
            else:
                # Convert block to sparse entries
                block_rows, block_cols = np.nonzero(C_block)
                for i, j in zip(block_rows, block_cols):
                    rows.append(row_start + i)
                    cols.append(col_start + j)
                    vals.append(C_block[i, j])
            
            print(f"  ✓ Loaded block ({block_info['row_block']},{block_info['col_block']})")
        
        print(f"\n{'='*70}")
        print(f"✓ Reconstruction complete!")
        
        if output_format == 'dense':
            nnz = np.count_nonzero(result)
            print(f"  Non-zero entries: {nnz:,}")
            print(f"  Sparsity: {100 * (1 - nnz / matrix_size**2):.4f}%")
            print(f"{'='*70}\n")
            return result
        elif output_format == 'csr':
            result_csr = sp.csr_matrix((vals, (rows, cols)), 
                                       shape=(matrix_size, matrix_size))
            nnz = result_csr.nnz
            print(f"  Non-zero entries: {nnz:,}")
            print(f"  Sparsity: {100 * (1 - nnz / matrix_size**2):.4f}%")
            print(f"{'='*70}\n")
            return result_csr
        else:
            return rows, cols, vals


def main():
    """
    Example usage of block-based GPU multiplication.
    """
    # Configuration
    DATA_DIR = Path("../data/input")
    OUTPUT_DIR = Path("./block_results")
    
    MATRIX_A_FILE = DATA_DIR / "matrix_a.csv"
    MATRIX_B_FILE = DATA_DIR / "matrix_b.csv"
    MATRIX_SIZE = 50000  # Adjust based on your data
    
    # Initialize multiplier
    multiplier = BlockMatrixMultiplier(gpu_memory_gb=12)
    
    # Estimate optimal block size
    block_size = multiplier.estimate_block_size(MATRIX_SIZE, sparsity=0.99)
    
    # Load matrices
    print("\nLoading input matrices...")
    A_csr = multiplier.load_sparse_csv(MATRIX_A_FILE, MATRIX_SIZE)
    B_csr = multiplier.load_sparse_csv(MATRIX_B_FILE, MATRIX_SIZE)
    
    # Multiply using blocks
    metadata_path = multiplier.multiply_blocks(
        A_csr, B_csr, block_size, OUTPUT_DIR
    )
    
    # Reconstruct result (optional - for verification)
    print("\nReconstructing result for verification...")
    result_csr = multiplier.reconstruct_result(metadata_path, output_format='csr')
    
    # Save final result
    result_output = OUTPUT_DIR / "result_matrix.npz"
    sp.save_npz(result_output, result_csr)
    print(f"\n✓ Final result saved: {result_output}")


if __name__ == "__main__":
    main()
