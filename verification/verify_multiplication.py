"""
Verification Script for Sparse Matrix Multiplication
Tests whether the parallel multiplication produces correct results.

This script:
1. Loads matrices A and B from CSV
2. Computes C = A × B using scipy (ground truth)
3. Compares with our parallel implementation result
4. Reports differences if any
"""

import numpy as np
from scipy import sparse as sp
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_coo_from_csv(filepath, shape):
    """Load sparse matrix from CSV in COO format."""
    rows, cols, values = [], [], []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    rows.append(int(parts[0]))
                    cols.append(int(parts[1]))
                    values.append(float(parts[2]))
                except ValueError:
                    continue
    
    return sp.coo_matrix((values, (rows, cols)), shape=shape)


def compare_matrices(matrix1, matrix2, tolerance=1e-9):
    """Compare two sparse matrices."""
    if matrix1.shape != matrix2.shape:
        logger.error(f"Shape mismatch: {matrix1.shape} vs {matrix2.shape}")
        return False
    
    diff = matrix1 - matrix2
    max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
    
    if max_diff > tolerance:
        logger.error(f"Maximum difference: {max_diff}")
        logger.error(f"Number of differing entries: {diff.nnz}")
        return False
    
    logger.info("✓ Matrices match!")
    return True


def verify_simple_example():
    """
    Test with a simple manual example:
    A = [[1, 0, 2],    (row 0: col 0=1, col 2=2)
         [0, 3, 0]]    (row 1: col 1=3)
    
    B = [[4, 0],       (row 0: col 0=4)
         [0, 5],       (row 1: col 1=5)
         [6, 0]]       (row 2: col 0=6)
    
    Expected C = A × B:
    C[0,0] = A[0,:] · B[:,0] = 1*4 + 0*0 + 2*6 = 4 + 12 = 16
    C[0,1] = A[0,:] · B[:,1] = 1*0 + 0*5 + 2*0 = 0
    C[1,0] = A[1,:] · B[:,0] = 0*4 + 3*0 + 0*6 = 0
    C[1,1] = A[1,:] · B[:,1] = 0*0 + 3*5 + 0*0 = 15
    
    So C = [[16, 0],
            [0, 15]]
    """
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Simple Manual Example")
    logger.info("="*70)
    
    # Matrix A (2x3)
    rows_a = [0, 0, 1]
    cols_a = [0, 2, 1]
    vals_a = [1, 2, 3]
    A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(2, 3))
    
    logger.info(f"Matrix A ({A.shape}):")
    logger.info(f"  (0,0) = 1")
    logger.info(f"  (0,2) = 2")
    logger.info(f"  (1,1) = 3")
    
    # Matrix B (3x2)
    rows_b = [0, 1, 2]
    cols_b = [0, 1, 0]
    vals_b = [4, 5, 6]
    B = sp.coo_matrix((vals_b, (rows_b, cols_b)), shape=(3, 2))
    
    logger.info(f"\nMatrix B ({B.shape}):")
    logger.info(f"  (0,0) = 4")
    logger.info(f"  (1,1) = 5")
    logger.info(f"  (2,0) = 6")
    
    # Compute with scipy
    C_scipy = A.tocsr() @ B.tocsr()
    
    logger.info(f"\nExpected C = A × B ({C_scipy.shape}):")
    C_coo = C_scipy.tocoo()
    for i, j, v in zip(C_coo.row, C_coo.col, C_coo.data):
        logger.info(f"  ({i},{j}) = {v}")
    
    logger.info("\nManual verification:")
    logger.info(f"  C[0,0] = A[0,:] · B[:,0] = 1*4 + 0*0 + 2*6 = {1*4 + 2*6}")
    logger.info(f"  C[0,1] = A[0,:] · B[:,1] = 1*0 + 0*5 + 2*0 = {1*0 + 2*0}")
    logger.info(f"  C[1,0] = A[1,:] · B[:,0] = 0*4 + 3*0 + 0*6 = {3*0}")
    logger.info(f"  C[1,1] = A[1,:] · B[:,1] = 0*0 + 3*5 + 0*0 = {3*5}")
    
    return True


def verify_your_matrices():
    """Verify the actual matrices from your files."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Verifying Your Matrices")
    logger.info("="*70)
    
    # File paths
    file_a = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_a_sorted.csv"
    file_b = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_b_sorted.csv"
    result_file = "C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\DB_Project_MatMul\\data\\ouput\\matrix_product_parallel.csv"
    
    shape_a = (50000, 50000)
    shape_b = (50000, 50000)
    
    logger.info("Loading matrices (this may take a moment)...")
    A = load_coo_from_csv(file_a, shape_a)
    B = load_coo_from_csv(file_b, shape_b)
    
    logger.info(f"\nMatrix A: {A.shape}, {A.nnz:,} non-zeros")
    logger.info(f"Matrix B: {B.shape}, {B.nnz:,} non-zeros")
    
    # Sample some entries from A and B
    logger.info("\nSample from Matrix A (first 5 entries):")
    A_coo = A.tocoo()
    for idx in range(min(5, A.nnz)):
        logger.info(f"  A[{A_coo.row[idx]},{A_coo.col[idx]}] = {A_coo.data[idx]}")
    
    logger.info("\nSample from Matrix B (first 5 entries):")
    B_coo = B.tocoo()
    for idx in range(min(5, B.nnz)):
        logger.info(f"  B[{B_coo.row[idx]},{B_coo.col[idx]}] = {B_coo.data[idx]}")
    
    # Compute with scipy
    logger.info("\nComputing C = A × B with scipy (ground truth)...")
    C_scipy = (A.tocsr() @ B.tocsr()).tocoo()
    
    logger.info(f"Result C: {C_scipy.shape}, {C_scipy.nnz:,} non-zeros")
    logger.info(f"Sparsity: {C_scipy.nnz / (C_scipy.shape[0] * C_scipy.shape[1]) * 100:.4f}%")
    
    logger.info("\nSample from scipy result (first 10 entries):")
    for idx in range(min(10, C_scipy.nnz)):
        logger.info(f"  C[{C_scipy.row[idx]},{C_scipy.col[idx]}] = {C_scipy.data[idx]}")
    
    # Load our result
    logger.info("\nLoading your parallel implementation result...")
    try:
        C_ours = load_coo_from_csv(result_file, (shape_a[0], shape_b[1]))
        logger.info(f"Your result: {C_ours.shape}, {C_ours.nnz:,} non-zeros")
        
        logger.info("\nSample from your result (first 10 entries):")
        C_ours_coo = C_ours.tocoo()
        for idx in range(min(10, C_ours.nnz)):
            logger.info(f"  C[{C_ours_coo.row[idx]},{C_ours_coo.col[idx]}] = {C_ours_coo.data[idx]}")
        
        # Compare
        logger.info("\n" + "="*70)
        logger.info("COMPARISON")
        logger.info("="*70)
        logger.info(f"Scipy result:  {C_scipy.nnz:,} non-zeros")
        logger.info(f"Your result:   {C_ours.nnz:,} non-zeros")
        logger.info(f"Difference:    {abs(C_scipy.nnz - C_ours.nnz):,} entries")
        
        if compare_matrices(C_scipy, C_ours, tolerance=1e-6):
            logger.info("\n✓✓✓ SUCCESS! Your implementation is CORRECT! ✓✓✓")
        else:
            logger.error("\n✗✗✗ MISMATCH! Your implementation has errors. ✗✗✗")
            
            # Show specific differences
            logger.info("\nInvestigating differences...")
            
            # Find entries only in scipy result
            scipy_set = set()
            for i, j, v in zip(C_scipy.row, C_scipy.col, C_scipy.data):
                scipy_set.add((i, j, round(v, 6)))
            
            ours_set = set()
            for i, j, v in zip(C_ours_coo.row, C_ours_coo.col, C_ours_coo.data):
                ours_set.add((i, j, round(v, 6)))
            
            missing = scipy_set - ours_set
            extra = ours_set - scipy_set
            
            if missing:
                logger.info(f"\nEntries in scipy but missing in yours: {len(missing)}")
                for i, j, v in list(missing)[:5]:
                    logger.info(f"  Missing: C[{i},{j}] = {v}")
            
            if extra:
                logger.info(f"\nExtra entries in yours not in scipy: {len(extra)}")
                for i, j, v in list(extra)[:5]:
                    logger.info(f"  Extra: C[{i},{j}] = {v}")
    
    except FileNotFoundError:
        logger.warning(f"Result file not found: {result_file}")
        logger.info("Run sparse_multiplication_parallel.py first to generate results.")
        
        # Still show expected result
        logger.info("\nExpected result statistics:")
        logger.info(f"  Shape: {C_scipy.shape}")
        logger.info(f"  Non-zeros: {C_scipy.nnz:,}")
        
        # Analyze the expected result more
        logger.info("\nAnalyzing expected result structure...")
        logger.info(f"  Rows with entries: {len(set(C_scipy.row))}")
        logger.info(f"  Cols with entries: {len(set(C_scipy.col))}")
        logger.info(f"  Min value: {C_scipy.data.min()}")
        logger.info(f"  Max value: {C_scipy.data.max()}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPARSE MATRIX MULTIPLICATION VERIFICATION")
    print("="*70)
    
    # Test 1: Simple example
    verify_simple_example()
    
    # Test 2: Your actual matrices
    verify_your_matrices()
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
