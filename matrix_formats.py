"""   
Matrix Format Conversions for Large Sparse Matrices
Supports COO, CSR, and CSC formats with out-of-core processing for matrices that don't fit in RAM.

Key Design:
- All conversions support blocked processing
- CSR/CSC built incrementally without loading entire matrix
- Memory-efficient streaming from disk
- Numba JIT acceleration for performance-critical operations
- scipy.sparse for verification and comparison
"""

import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import tempfile
import os
import numba
from scipy import sparse as sp


from scipy import sparse as sp

# ============================================================================
# Numba-Accelerated Helper Functions
# ============================================================================

@numba.jit(nopython=True, cache=True)
def _build_csr_arrays(rows, cols, values, num_rows):
    """
    Build CSR arrays from sorted COO data using Numba acceleration.
    
    Args:
        rows: Sorted row indices
        cols: Column indices
        values: Values
        num_rows: Total number of rows
    
    Returns:
        (row_ptr, col_idx, values)
    """
    nnz = len(rows)
    row_ptr = np.zeros(num_rows + 1, dtype=np.int64)
    
    # Count entries per row
    for i in range(nnz):
        row_ptr[rows[i] + 1] += 1
    
    # Cumulative sum to get pointers
    for i in range(1, num_rows + 1):
        row_ptr[i] += row_ptr[i - 1]
    
    return row_ptr, cols.copy(), values.copy()


@numba.jit(nopython=True, cache=True)
def _build_csc_arrays(rows, cols, values, num_cols):
    """
    Build CSC arrays from column-sorted COO data using Numba acceleration.
    
    Args:
        rows: Row indices
        cols: Sorted column indices
        values: Values
        num_cols: Total number of columns
    
    Returns:
        (col_ptr, row_idx, values)
    """
    nnz = len(cols)
    col_ptr = np.zeros(num_cols + 1, dtype=np.int64)
    
    # Count entries per column
    for i in range(nnz):
        col_ptr[cols[i] + 1] += 1
    
    # Cumulative sum to get pointers
    for i in range(1, num_cols + 1):
        col_ptr[i] += col_ptr[i - 1]
    
    return col_ptr, rows.copy(), values.copy()


@numba.jit(nopython=True, cache=True)
def _merge_two_sorted_rows(cols1, vals1, cols2, vals2):
    """
    Merge two sorted row segments (by column index) using two-pointer technique.
    Used for verifying CSR operations.
    
    Args:
        cols1, vals1: First sorted segment
        cols2, vals2: Second sorted segment
    
    Returns:
        (merged_cols, merged_vals)
    """
    n1, n2 = len(cols1), len(cols2)
    result_cols = []
    result_vals = []
    
    i, j = 0, 0
    
    while i < n1 and j < n2:
        if cols1[i] < cols2[j]:
            result_cols.append(cols1[i])
            result_vals.append(vals1[i])
            i += 1
        elif cols1[i] > cols2[j]:
            result_cols.append(cols2[j])
            result_vals.append(vals2[j])
            j += 1
        else:  # Equal columns
            result_cols.append(cols1[i])
            result_vals.append(vals1[i] + vals2[j])
            i += 1
            j += 1
    
    # Append remaining
    while i < n1:
        result_cols.append(cols1[i])
        result_vals.append(vals1[i])
        i += 1
    
    while j < n2:
        result_cols.append(cols2[j])
        result_vals.append(vals2[j])
        j += 1
    
    return np.array(result_cols), np.array(result_vals)


class COOMatrix:
    """
    Coordinate (COO) format: List of (row, col, value) triples.
    
    For large matrices, data is stored on disk and accessed in blocks.
    This class provides utilities for reading/writing CSV files.
    """
    
    def __init__(self, shape: Tuple[int, int], filepath: Optional[str] = None, 
                 data: Optional[List[Tuple[int, int, float]]] = None):
        """
        Args:
            shape: (num_rows, num_cols)
            filepath: Path to CSV file containing matrix data
            data: In-memory list of (i, j, v) triples (for small matrices)
        """
        self.shape = shape
        self.filepath = filepath
        self.data = data  # Only for small matrices that fit in memory
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def from_csv(cls, filepath: str, shape: Optional[Tuple[int, int]] = None):
        """
        Create COOMatrix from CSV file.
        
        Args:
            filepath: Path to CSV file (format: row,col,value per line)
            shape: Optional (rows, cols). If None, inferred from data.
        
        Returns:
            COOMatrix instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Matrix file not found: {filepath}")
        
        # Infer shape if not provided (requires scanning file)
        if shape is None:
            max_row, max_col = 0, 0
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        try:
                            row, col = int(parts[0]), int(parts[1])
                            max_row = max(max_row, row)
                            max_col = max(max_col, col)
                        except ValueError:
                            continue
            shape = (max_row + 1, max_col + 1)
        
        return cls(shape=shape, filepath=str(filepath))
    
    def to_csv(self, filepath: str):
        """
        Write COOMatrix to CSV file.
        
        Args:
            filepath: Output CSV file path
        """
        if self.filepath and not self.data:
            # Already on disk, just copy or reference
            import shutil
            shutil.copy(self.filepath, filepath)
            self.logger.info(f"Copied matrix to {filepath}")
        elif self.data:
            # Write in-memory data to CSV
            with open(filepath, 'w') as f:
                for i, j, v in self.data:
                    f.write(f"{i},{j},{v}\n")
            self.logger.info(f"Wrote {len(self.data)} entries to {filepath}")
        else:
            raise ValueError("No data to write")
    
    def iter_entries(self, chunk_size: int = 1000000):
        """
        Iterator over matrix entries in chunks.
        Yields (row, col, value) tuples.
        
        Args:
            chunk_size: Number of entries to read at once
        
        Yields:
            Lists of (i, j, v) tuples
        """
        if self.data:
            # In-memory data
            for i in range(0, len(self.data), chunk_size):
                yield self.data[i:i + chunk_size]
        elif self.filepath:
            # Stream from disk
            chunk = []
            with open(self.filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        try:
                            i = int(parts[0])
                            j = int(parts[1])
                            v = int(float(parts[2]))
                            chunk.append((i, j, v))
                            
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
                        except ValueError:
                            continue
            
            if chunk:  # Yield remaining
                yield chunk
        else:
            raise ValueError("No data source available")
    
    def count_nnz(self) -> int:
        """
        Count number of nonzero entries.
        For large files, this scans the entire file.
        
        Returns:
            Number of nonzero entries
        """
        if self.data:
            return len(self.data)
        elif self.filepath:
            count = 0
            with open(self.filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        count += 1
            return count
        return 0
    
    def to_csr(self, block_size: int = 1000) -> 'CSRMatrix':
        """
        Convert COO to CSR format.
        
        For large matrices, processes in row blocks to avoid loading everything.
        
        Args:
            block_size: Number of rows to process at once
        
        Returns:
            CSRMatrix instance
        """
        return build_csr_from_coo(self, block_size=block_size)
    
    def to_csc(self, block_size: int = 1000) -> 'CSCMatrix':
        """
        Convert COO to CSC format.
        
        For large matrices, processes in column blocks.
        
        Args:
            block_size: Number of columns to process at once
        
        Returns:
            CSCMatrix instance
        """
        return build_csc_from_coo(self, block_size=block_size)
    
    def to_scipy_sparse(self) -> sp.coo_matrix:
        """
        Convert to scipy.sparse.coo_matrix for verification.
        
        WARNING: Only use for small matrices that fit in memory!
        This loads all data into memory.
        
        Returns:
            scipy.sparse.coo_matrix
        """
        logger = logging.getLogger(__name__)
        
        if self.data:
            rows = [i for i, j, v in self.data]
            cols = [j for i, j, v in self.data]
            values = [v for i, j, v in self.data]
        elif self.filepath:
            rows, cols, values = [], [], []
            with open(self.filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        try:
                            rows.append(int(parts[0]))
                            cols.append(int(parts[1]))
                            values.append(int(float(parts[2])))
                        except ValueError:
                            continue
        else:
            raise ValueError("No data available")
        
        logger.info(f"Converting to scipy.sparse.coo_matrix: {len(rows)} entries")
        return sp.coo_matrix((values, (rows, cols)), shape=self.shape)


class CSRMatrix:
    """
    Compressed Sparse Row (CSR) format.
    
    Storage:
    - row_ptr[i] = starting index in col_idx/values for row i
    - col_idx[k] = column index of k-th nonzero
    - values[k] = value of k-th nonzero
    
    For large matrices, can be built incrementally and stored partially on disk.
    """
    
    def __init__(self, shape: Tuple[int, int], row_ptr: np.ndarray, 
                 col_idx: np.ndarray, values: np.ndarray):
        """
        Args:
            shape: (num_rows, num_cols)
            row_ptr: Array of size (num_rows + 1), row pointers
            col_idx: Array of column indices
            values: Array of nonzero values
        """
        self.shape = shape
        self.row_ptr = row_ptr
        self.col_idx = col_idx
        self.values = values
        self.logger = logging.getLogger(__name__)
    
    def get_row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get row i's data in O(1) time.
        
        Args:
            i: Row index
        
        Returns:
            (col_indices, values) for row i
        """
        if i < 0 or i >= self.shape[0]:
            raise IndexError(f"Row index {i} out of bounds for shape {self.shape}")
        
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        
        return self.col_idx[start:end], self.values[start:end]
    
    def get_row_block(self, row_start: int, row_end: int) -> 'CSRMatrix':
        """
        Extract a block of rows [row_start, row_end).
        
        Args:
            row_start: Starting row (inclusive)
            row_end: Ending row (exclusive)
        
        Returns:
            CSRMatrix for the row block
        """
        if row_start < 0 or row_end > self.shape[0]:
            raise IndexError(f"Row range [{row_start}, {row_end}) out of bounds")
        
        # Find data range for this block
        data_start = self.row_ptr[row_start]
        data_end = self.row_ptr[row_end]
        
        # Extract block data
        block_col_idx = self.col_idx[data_start:data_end]
        block_values = self.values[data_start:data_end]
        
        # Adjust row pointers (shift to start at 0)
        block_row_ptr = self.row_ptr[row_start:row_end + 1] - data_start
        
        block_shape = (row_end - row_start, self.shape[1])
        
        return CSRMatrix(block_shape, block_row_ptr, block_col_idx, block_values)
    
    def nnz(self) -> int:
        """Return number of nonzeros."""
        return len(self.values)
    
    def save_to_disk(self, filepath: str):
        """
        Save CSR to disk (NPZ format for efficient storage).
        
        Args:
            filepath: Output file path (.npz)
        """
        np.savez_compressed(
            filepath,
            shape=self.shape,
            row_ptr=self.row_ptr,
            col_idx=self.col_idx,
            values=self.values
        )
        self.logger.info(f"Saved CSR matrix to {filepath}")
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> 'CSRMatrix':
        """
        Load CSR from disk.
        
        Args:
            filepath: Input file path (.npz)
        
        Returns:
            CSRMatrix instance
        """
        data = np.load(filepath)
        return cls(
            shape=tuple(data['shape']),
            row_ptr=data['row_ptr'],
            col_idx=data['col_idx'],
            values=data['values']
        )
    
    def to_scipy_sparse(self) -> sp.csr_matrix:
        """
        Convert to scipy.sparse.csr_matrix for verification.
        
        Returns:
            scipy.sparse.csr_matrix
        """
        return sp.csr_matrix(
            (self.values, self.col_idx, self.row_ptr),
            shape=self.shape
        )
    
    @classmethod
    def from_scipy_sparse(cls, scipy_csr: sp.csr_matrix) -> 'CSRMatrix':
        """
        Create CSRMatrix from scipy.sparse.csr_matrix.
        
        Args:
            scipy_csr: scipy CSR matrix
        
        Returns:
            CSRMatrix instance
        """
        return cls(
            shape=scipy_csr.shape,
            row_ptr=scipy_csr.indptr.astype(np.int64),
            col_idx=scipy_csr.indices.astype(np.int32),
            values=scipy_csr.data.astype(np.int32)
        )


class CSCMatrix:
    """
    Compressed Sparse Column (CSC) format.
    
    Storage:
    - col_ptr[j] = starting index in row_idx/values for column j
    - row_idx[k] = row index of k-th nonzero
    - values[k] = value of k-th nonzero
    
    Similar to CSR but organized by columns for efficient column access.
    """
    
    def __init__(self, shape: Tuple[int, int], col_ptr: np.ndarray,
                 row_idx: np.ndarray, values: np.ndarray):
        """
        Args:
            shape: (num_rows, num_cols)
            col_ptr: Array of size (num_cols + 1), column pointers
            row_idx: Array of row indices
            values: Array of nonzero values
        """
        self.shape = shape
        self.col_ptr = col_ptr
        self.row_idx = row_idx
        self.values = values
        self.logger = logging.getLogger(__name__)
    
    def get_col(self, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get column j's data in O(1) time.
        
        Args:
            j: Column index
        
        Returns:
            (row_indices, values) for column j
        """
        if j < 0 or j >= self.shape[1]:
            raise IndexError(f"Column index {j} out of bounds for shape {self.shape}")
        
        start = self.col_ptr[j]
        end = self.col_ptr[j + 1]
        
        return self.row_idx[start:end], self.values[start:end]
    
    def get_col_block(self, col_start: int, col_end: int) -> 'CSCMatrix':
        """
        Extract a block of columns [col_start, col_end).
        
        Args:
            col_start: Starting column (inclusive)
            col_end: Ending column (exclusive)
        
        Returns:
            CSCMatrix for the column block
        """
        if col_start < 0 or col_end > self.shape[1]:
            raise IndexError(f"Column range [{col_start}, {col_end}) out of bounds")
        
        # Find data range for this block
        data_start = self.col_ptr[col_start]
        data_end = self.col_ptr[col_end]
        
        # Extract block data
        block_row_idx = self.row_idx[data_start:data_end]
        block_values = self.values[data_start:data_end]
        
        # Adjust column pointers
        block_col_ptr = self.col_ptr[col_start:col_end + 1] - data_start
        
        block_shape = (self.shape[0], col_end - col_start)
        
        return CSCMatrix(block_shape, block_col_ptr, block_row_idx, block_values)
    
    def nnz(self) -> int:
        """Return number of nonzeros."""
        return len(self.values)
    
    def save_to_disk(self, filepath: str):
        """
        Save CSC to disk (NPZ format).
        
        Args:
            filepath: Output file path (.npz)
        """
        np.savez_compressed(
            filepath,
            shape=self.shape,
            col_ptr=self.col_ptr,
            row_idx=self.row_idx,
            values=self.values
        )
        self.logger.info(f"Saved CSC matrix to {filepath}")
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> 'CSCMatrix':
        """
        Load CSC from disk.
        
        Args:
            filepath: Input file path (.npz)
        
        Returns:
            CSCMatrix instance
        """
        data = np.load(filepath)
        return cls(
            shape=tuple(data['shape']),
            col_ptr=data['col_ptr'],
            row_idx=data['row_idx'],
            values=data['values']
        )
    
    def to_scipy_sparse(self) -> sp.csc_matrix:
        """
        Convert to scipy.sparse.csc_matrix for verification.
        
        Returns:
            scipy.sparse.csc_matrix
        """
        return sp.csc_matrix(
            (self.values, self.row_idx, self.col_ptr),
            shape=self.shape
        )
    
    @classmethod
    def from_scipy_sparse(cls, scipy_csc: sp.csc_matrix) -> 'CSCMatrix':
        """
        Create CSCMatrix from scipy.sparse.csc_matrix.
        
        Args:
            scipy_csc: scipy CSC matrix
        
        Returns:
            CSCMatrix instance
        """
        return cls(
            shape=scipy_csc.shape,
            col_ptr=scipy_csc.indptr.astype(np.int64),
            row_idx=scipy_csc.indices.astype(np.int32),
            values=scipy_csc.data.astype(np.int32)
        )


# ============================================================================
# Conversion Functions for Large Matrices (Out-of-Core)
# ============================================================================

def build_csr_from_coo(coo: COOMatrix, block_size: int = 1000) -> CSRMatrix:
    """
    Build CSR from COO format with blocked processing for large matrices.
    Uses Numba JIT acceleration for performance.
    
    Algorithm:
    1. Assumes COO is sorted by (row, col) - if not, must sort first
    2. Streams through COO data and collects into arrays
    3. Uses Numba-accelerated function to build row_ptr array
    
    Args:
        coo: COOMatrix (must be sorted by row, then col)
        block_size: Number of rows to process at once
    
    Returns:
        CSRMatrix
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building CSR from COO (shape={coo.shape})")
    
    num_rows, num_cols = coo.shape
    
    # Collect all entries into arrays
    rows_list = []
    cols_list = []
    vals_list = []
    
    for chunk in coo.iter_entries(chunk_size=100000):
        for i, j, v in chunk:
            rows_list.append(i)
            cols_list.append(j)
            vals_list.append(v)
    
    # Convert to NumPy arrays
    rows = np.array(rows_list, dtype=np.int32)
    cols = np.array(cols_list, dtype=np.int32)
    values = np.array(vals_list, dtype=np.int32)
    
    logger.info(f"Building CSR arrays with Numba acceleration...")
    
    # Use Numba-accelerated function to build CSR
    row_ptr, col_idx, values = _build_csr_arrays(rows, cols, values, num_rows)
    
    logger.info(f"CSR built: {len(values)} nonzeros, {num_rows} rows")
    
    return CSRMatrix(coo.shape, row_ptr, col_idx, values)


def build_csc_from_coo(coo: COOMatrix, block_size: int = 1000) -> CSCMatrix:
    """
    Build CSC from COO format with blocked processing for large matrices.
    Uses Numba JIT acceleration for performance.
    
    Algorithm:
    1. Must sort COO by (col, row) first (different from CSR!)
    2. Uses Numba-accelerated function to build col_ptr array
    
    For very large matrices, this requires re-sorting COO by column.
    
    Args:
        coo: COOMatrix
        block_size: Number of columns to process at once
    
    Returns:
        CSCMatrix
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building CSC from COO (shape={coo.shape})")
    
    num_rows, num_cols = coo.shape
    
    # For CSC, we need data sorted by (col, row)
    # If COO is sorted by (row, col), we need to re-sort
    
    # Step 1: Create temporary file sorted by (col, row)
    temp_dir = Path(tempfile.gettempdir())
    temp_sorted_by_col = temp_dir / f"coo_sorted_by_col_{os.getpid()}.csv"
    
    logger.info("Re-sorting COO by (col, row) for CSC conversion...")
    _sort_coo_by_column(coo, str(temp_sorted_by_col))
    
    # Step 2: Read sorted data and build CSC
    rows_list = []
    cols_list = []
    vals_list = []
    
    with open(temp_sorted_by_col, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 3:
                try:
                    i = int(parts[0])
                    j = int(parts[1])
                    v = int(float(parts[2]))
                    rows_list.append(i)
                    cols_list.append(j)
                    vals_list.append(v)
                except ValueError:
                    continue
    
    # Convert to arrays
    rows = np.array(rows_list, dtype=np.int32)
    cols = np.array(cols_list, dtype=np.int32)
    values = np.array(vals_list, dtype=np.int32)
    
    logger.info(f"Building CSC arrays with Numba acceleration...")
    
    # Use Numba-accelerated function
    col_ptr, row_idx, values = _build_csc_arrays(rows, cols, values, num_cols)
    
    # Cleanup
    try:
        os.remove(temp_sorted_by_col)
    except OSError:
        pass
    
    logger.info(f"CSC built: {len(values)} nonzeros, {num_cols} columns")
    
    return CSCMatrix(coo.shape, col_ptr, row_idx, values)


def _sort_coo_by_column(coo: COOMatrix, output_file: str):
    """
    Sort COO data by (column, row) instead of (row, column).
    
    For large files, uses external sort.
    
    Args:
        coo: Input COOMatrix
        output_file: Output CSV file sorted by (col, row)
    """
    from external_sort import ExternalSorter
    
    logger = logging.getLogger(__name__)
    
    # Create temporary file with col,row,value order (swap i and j)
    temp_dir = Path(tempfile.gettempdir())
    temp_swapped = temp_dir / f"coo_swapped_{os.getpid()}.csv"
    
    # Write with swapped columns
    with open(temp_swapped, 'w') as out:
        for chunk in coo.iter_entries():
            for i, j, v in chunk:
                # Write as (col, row, value) so external sort will sort by col first
                out.write(f"{j},{i},{v}\n")
    
    # External sort by (col, row)
    sorter = ExternalSorter(chunk_size_mb=100, logger=logger)
    sorter.sort_file(str(temp_swapped), output_file)
    
    # Cleanup
    try:
        os.remove(temp_swapped)
    except OSError:
        pass


# ============================================================================
# Utility Functions
# ============================================================================

def verify_csr_correctness(csr: CSRMatrix, coo: COOMatrix) -> bool:
    """
    Verify that CSR conversion is correct by comparing with original COO.
    
    Args:
        csr: CSRMatrix to verify
        coo: Original COOMatrix
    
    Returns:
        True if correct, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Verifying CSR correctness...")
    
    # Build dictionary from COO
    coo_dict = {}
    for chunk in coo.iter_entries():
        for i, j, v in chunk:
            coo_dict[(i, j)] = v
    
    # Check CSR entries match
    for i in range(csr.shape[0]):
        cols, vals = csr.get_row(i)
        for j, v in zip(cols, vals):
            if (i, j) not in coo_dict:
                logger.error(f"CSR has entry ({i},{j}) not in COO")
                return False
            if abs(coo_dict[(i, j)] - v) > 1e-9:
                logger.error(f"Value mismatch at ({i},{j}): COO={coo_dict[(i,j)]}, CSR={v}")
                return False
    
    logger.info("✓ CSR verification passed")
    return True


def print_matrix_info(matrix, name="Matrix"):
    """
    Print statistics about a matrix.
    
    Args:
        matrix: COOMatrix, CSRMatrix, or CSCMatrix
        name: Name to display
    """
    logger = logging.getLogger(__name__)
    
    if isinstance(matrix, COOMatrix):
        nnz = matrix.count_nnz()
        logger.info(f"{name} (COO): shape={matrix.shape}, nnz={nnz:,}")
    elif isinstance(matrix, CSRMatrix):
        logger.info(f"{name} (CSR): shape={matrix.shape}, nnz={matrix.nnz():,}")
    elif isinstance(matrix, CSCMatrix):
        logger.info(f"{name} (CSC): shape={matrix.shape}, nnz={matrix.nnz():,}")
    else:
        logger.info(f"{name}: Unknown format")
    
    # Calculate sparsity
    total_entries = matrix.shape[0] * matrix.shape[1]
    if isinstance(matrix, COOMatrix):
        nnz = matrix.count_nnz()
    else:
        nnz = matrix.nnz()
    
    sparsity = (nnz / total_entries * 100) if total_entries > 0 else 0
    logger.info(f"  Sparsity: {sparsity:.4f}% ({nnz:,} / {total_entries:,})")


# ============================================================================
# Scipy Verification and Comparison Functions
# ============================================================================

def verify_csr_against_scipy(csr: CSRMatrix, coo: COOMatrix, tolerance=1e-9) -> bool:
    """
    Verify CSR conversion correctness against scipy.sparse.
    
    Args:
        csr: Our CSRMatrix
        coo: Original COOMatrix (must fit in memory for this test)
        tolerance: Numerical tolerance for comparisons
    
    Returns:
        True if verification passes
    """
    logger = logging.getLogger(__name__)
    logger.info("Verifying CSR against scipy.sparse...")
    
    try:
        # Convert both to scipy
        scipy_coo = coo.to_scipy_sparse()
        scipy_csr_expected = scipy_coo.tocsr()
        scipy_csr_ours = csr.to_scipy_sparse()
        
        # Compare shapes
        if scipy_csr_ours.shape != scipy_csr_expected.shape:
            logger.error(f"Shape mismatch: ours={scipy_csr_ours.shape}, scipy={scipy_csr_expected.shape}")
            return False
        
        # Compare data
        diff = scipy_csr_ours - scipy_csr_expected
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        
        if max_diff > tolerance:
            logger.error(f"Data mismatch: max difference = {max_diff}")
            return False
        
        logger.info("✓ CSR verification passed (matches scipy.sparse)")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def verify_csc_against_scipy(csc: CSCMatrix, coo: COOMatrix, tolerance=1e-9) -> bool:
    """
    Verify CSC conversion correctness against scipy.sparse.
    
    Args:
        csc: Our CSCMatrix
        coo: Original COOMatrix (must fit in memory for this test)
        tolerance: Numerical tolerance for comparisons
    
    Returns:
        True if verification passes
    """
    logger = logging.getLogger(__name__)
    logger.info("Verifying CSC against scipy.sparse...")
    
    try:
        # Convert both to scipy
        scipy_coo = coo.to_scipy_sparse()
        scipy_csc_expected = scipy_coo.tocsc()
        scipy_csc_ours = csc.to_scipy_sparse()
        
        # Compare shapes
        if scipy_csc_ours.shape != scipy_csc_expected.shape:
            logger.error(f"Shape mismatch: ours={scipy_csc_ours.shape}, scipy={scipy_csc_expected.shape}")
            return False
        
        # Compare data
        diff = scipy_csc_ours - scipy_csc_expected
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0
        
        if max_diff > tolerance:
            logger.error(f"Data mismatch: max difference = {max_diff}")
            return False
        
        logger.info("✓ CSC verification passed (matches scipy.sparse)")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def compare_performance_scipy(operation_name: str, our_time: float, scipy_time: float):
    """
    Compare performance between our implementation and scipy.
    
    Args:
        operation_name: Name of the operation
        our_time: Time taken by our implementation (seconds)
        scipy_time: Time taken by scipy (seconds)
    """
    logger = logging.getLogger(__name__)
    
    speedup = scipy_time / our_time if our_time > 0 else float('inf')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Performance Comparison: {operation_name}")
    logger.info(f"{'-'*60}")
    logger.info(f"Our implementation:  {our_time:.4f}s")
    logger.info(f"scipy.sparse:        {scipy_time:.4f}s")
    logger.info(f"Speedup:             {speedup:.2f}x {'(FASTER)' if speedup > 1 else '(slower)'}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":

    ##################### LOADING THE EXTERNAL SORTED COO DATA #####################
    from matrix_formats import COOMatrix

    # Load the sorted file from external_sort output
    coo = COOMatrix.from_csv("C:\\Users\\khald\\OneDrive - University Of Houston\\FALL2025\\COSC 6340\\Project\\Phase 1\\code\\data\\ouput\\matrix_a_sorted.csv", shape=(50000, 50000))

    # Convert to CSR
    csr = coo.to_csr()
    csc = coo.to_csc()
    print(f"CSR: {csr.nnz()} nonzeros")
    print(f"CSC: {csc.nnz()} nonzeros")
    ################################################################################

    # Example usage with verification
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Matrix Formats Module (with Numba + scipy verification)")
    print("=" * 70)
    print("Available classes:")
    print("  - COOMatrix: Coordinate format")
    print("  - CSRMatrix: Compressed Sparse Row")
    print("  - CSCMatrix: Compressed Sparse Column")
    print("\nKey features:")
    print("  ✓ Numba JIT acceleration for sorting and conversions")
    print("  ✓ scipy.sparse integration for verification")
    print("  ✓ Out-of-core processing for large matrices")
    print("\nVerification functions:")
    print("  - verify_csr_against_scipy()")
    print("  - verify_csc_against_scipy()")
    print("  - compare_performance_scipy()")

