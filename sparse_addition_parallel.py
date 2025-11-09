"""
Parallel Sparse Matrix Addition Module
Multi-core CPU parallelization for faster sparse matrix addition

Key Features:
- Parallel chunk processing using multiprocessing
- Reuses optimized merge algorithm from sparse_addition.py
- Minimal overhead with process pools
- Compatible with existing sparse_addition.py

Usage:
    from sparse_addition_parallel import add_matrices_parallel
    
    add_matrices_parallel(
        file_a='data/output/matrix_a.csv',
        file_b='data/output/matrix_b.csv',
        output_file='data/output/sum.csv',
        num_workers=8  # Use 8 CPU cores
    )
"""

import csv
import heapq
from typing import Iterator, Tuple, List, Optional
import os
import tempfile
import logging
import multiprocessing as mp
from functools import partial
import itertools

# Import existing optimized functions from sparse_addition.py
from sparse_addition import SparseAddition


# ============================================================================
# Parallel Worker Functions
# ============================================================================

def _process_chunk_pair(chunk_id: int,
                       file1: str, start1: int, size1: int,
                       file2: str, start2: int, size2: int,
                       temp_dir: str) -> str:
    """
    Worker function to process a pair of chunks in parallel.
    Uses the existing merge_sorted_streams from sparse_addition.py
    
    Args:
        chunk_id: Chunk identifier
        file1: Path to first input file
        start1, size1: Start line and size for first chunk
        file2: Path to second input file
        start2, size2: Start line and size for second chunk
        temp_dir: Directory for temporary files
    
    Returns:
        Path to temporary file containing merged chunk
    """
    # Create iterators for chunks
    stream1 = _read_chunk_as_iterator(file1, start1, size1)
    stream2 = _read_chunk_as_iterator(file2, start2, size2)
    
    # Use existing merge algorithm from sparse_addition.py
    adder = SparseAddition()
    merged = list(adder.merge_sorted_streams(stream1, stream2))
    
    if not merged:
        return None
    
    # Write to temporary file
    temp_file = os.path.join(temp_dir, f'chunk_{chunk_id:04d}.csv')
    with open(temp_file, 'w', newline='') as f:
        for row, col, val in merged:
            f.write(f"{row},{col},{val}\n")
    
    return temp_file


def _read_chunk_as_iterator(filepath: str, start_line: int, num_lines: int) -> Iterator[Tuple[int, int, float]]:
    """
    Read a chunk of entries from a CSV file as an iterator.
    
    Args:
        filepath: Path to input file
        start_line: Starting line number (0-indexed, skips header)
        num_lines: Number of lines to read
    
    Yields:
        (row, col, value) tuples
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        
        # Skip to start line
        for _ in range(start_line):
            try:
                next(reader)
            except StopIteration:
                return
        
        # Read chunk
        for _ in range(num_lines):
            try:
                row_data = next(reader)
                yield (int(row_data['row']), int(row_data['col']), float(row_data['value']))
            except StopIteration:
                break


# ============================================================================
# Parallel Matrix Addition Class
# ============================================================================

class ParallelSparseAddition:
    """
    Parallel sparse matrix addition using multi-core CPU.
    
    Processes chunks in parallel then merges results.
    """
    
    def __init__(self,
                 chunk_size: int = 100000,
                 num_workers: Optional[int] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            chunk_size: Number of entries per chunk
            num_workers: Number of parallel workers (default: CPU count)
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.num_workers = num_workers or mp.cpu_count()
        self.logger = logger or logging.getLogger(__name__)
    
    def add_matrices(self, file1: str, file2: str, output_file: str) -> str:
        """
        Parallel addition algorithm.
        
        Args:
            file1: Path to first matrix (COO CSV, sorted)
            file2: Path to second matrix (COO CSV, sorted)
            output_file: Path for result matrix
        
        Returns:
            Path to output file
        """
        self.logger.info(f"="*70)
        self.logger.info(f"PARALLEL Sparse Matrix Addition")
        self.logger.info(f"Workers: {self.num_workers} CPU cores")
        self.logger.info(f"="*70)
        
        # Count entries
        size1 = self._count_lines(file1)
        size2 = self._count_lines(file2)
        
        self.logger.info(f"Matrix 1: {size1} nonzeros")
        self.logger.info(f"Matrix 2: {size2} nonzeros")
        
        # If small enough, use simple merge
        if size1 + size2 <= self.chunk_size:
            self.logger.info("Small matrices - using direct merge...")
            self._add_direct(file1, file2, output_file)
            return output_file
        
        # Parallel chunk processing
        self.logger.info(f"Processing chunks in parallel...")
        temp_files = self._process_chunks_parallel(file1, size1, file2, size2)
        
        # Merge all chunks
        self.logger.info(f"Merging {len(temp_files)} chunks...")
        self._merge_chunks(temp_files, output_file)
        
        self.logger.info(f"✓ Parallel addition complete: {output_file}")
        self.logger.info(f"="*70)
        
        return output_file
    
    def _count_lines(self, filepath: str) -> int:
        """Count number of entries in file (excluding header)."""
        # Use existing method from SparseAddition
        adder = SparseAddition()
        return adder._count_entries(filepath)
    
    def _add_direct(self, file1: str, file2: str, output_file: str):
        """Direct merge for small matrices using existing algorithm."""
        # Use the existing optimized in-memory addition
        adder = SparseAddition(memory_limit_mb=self.chunk_size // 25000)  # Rough conversion
        adder._add_in_memory(file1, file2, output_file)
    
    def _process_chunks_parallel(self, file1: str, size1: int, 
                                 file2: str, size2: int) -> List[str]:
        """
        Process chunks in parallel.
        """
        # Determine number of chunks
        max_size = max(size1, size2)
        num_chunks = (max_size + self.chunk_size - 1) // self.chunk_size
        
        # Create work items
        work_items = []
        temp_dir = tempfile.mkdtemp(prefix='sparse_add_parallel_')
        
        for chunk_id in range(num_chunks):
            start1 = chunk_id * self.chunk_size
            chunk_size1 = min(self.chunk_size, size1 - start1) if start1 < size1 else 0
            
            start2 = chunk_id * self.chunk_size
            chunk_size2 = min(self.chunk_size, size2 - start2) if start2 < size2 else 0
            
            if chunk_size1 > 0 or chunk_size2 > 0:
                work_items.append((
                    chunk_id,
                    file1, start1, chunk_size1,
                    file2, start2, chunk_size2,
                    temp_dir
                ))
        
        self.logger.info(f"Processing {len(work_items)} chunk pairs...")
        
        # Process in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            result_files = pool.starmap(_process_chunk_pair, work_items)
        
        # Filter out None results
        result_files = [f for f in result_files if f is not None]
        
        return result_files
    
    def _merge_chunks(self, temp_files: List[str], output_file: str):
        """
        Merge all chunk files using existing merge algorithm.
        """
        if not temp_files:
            # Write empty file with header
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['row', 'col', 'value'])
            return
        
        if len(temp_files) == 1:
            # Only one file - add header and use it
            with open(output_file, 'w', newline='') as out:
                writer = csv.writer(out)
                writer.writerow(['row', 'col', 'value'])
                with open(temp_files[0], 'r') as inf:
                    for line in inf:
                        out.write(line)
            os.remove(temp_files[0])
            return
        
        # Multi-way merge using existing merge algorithm
        # Strategy: iteratively merge pairs until we have one file
        adder = SparseAddition()
        
        current_files = temp_files[:]
        merge_round = 0
        
        while len(current_files) > 1:
            merge_round += 1
            next_files = []
            
            # Merge pairs of files
            for i in range(0, len(current_files), 2):
                if i + 1 < len(current_files):
                    # Merge two files
                    file1 = current_files[i]
                    file2 = current_files[i + 1]
                    merged_file = tempfile.mktemp(
                        prefix=f'merge_r{merge_round}_',
                        suffix='.csv',
                        dir=os.path.dirname(temp_files[0])
                    )
                    
                    # Use existing merge from SparseAddition
                    stream1 = self._read_temp_file_iterator(file1)
                    stream2 = self._read_temp_file_iterator(file2)
                    
                    with open(merged_file, 'w', newline='') as out:
                        for row, col, val in adder.merge_sorted_streams(stream1, stream2):
                            out.write(f"{row},{col},{val}\n")
                    
                    next_files.append(merged_file)
                    
                    # Clean up input files
                    os.remove(file1)
                    os.remove(file2)
                else:
                    # Odd file out - carry forward
                    next_files.append(current_files[i])
            
            current_files = next_files
        
        # Final file - add header
        with open(output_file, 'w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['row', 'col', 'value'])
            with open(current_files[0], 'r') as inf:
                for line in inf:
                    out.write(line)
        
        # Clean up final temp file
        os.remove(current_files[0])
    
    def _read_temp_file_iterator(self, filepath: str) -> Iterator[Tuple[int, int, float]]:
        """Read temporary file (no header) as iterator."""
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    yield (int(parts[0]), int(parts[1]), float(parts[2]))


# ============================================================================
# Convenience Function
# ============================================================================

def add_matrices_parallel(file1: str, file2: str, output_file: str,
                          chunk_size: int = 100000,
                          num_workers: Optional[int] = None) -> str:
    """
    Parallel sparse matrix addition using multi-core CPU.
    
    Args:
        file1: Path to first matrix (COO CSV, sorted)
        file2: Path to second matrix (COO CSV, sorted)
        output_file: Path for result matrix
        chunk_size: Number of entries per chunk (default 100000)
        num_workers: Number of CPU cores to use (default: all available)
    
    Returns:
        Path to output file
    """
    logger = logging.getLogger(__name__)
    
    adder = ParallelSparseAddition(
        chunk_size=chunk_size,
        num_workers=num_workers,
        logger=logger
    )
    
    return adder.add_matrices(file1, file2, output_file)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("PARALLEL Sparse Matrix Addition")
    print("="*70)
    print(f"\nAvailable CPU cores: {mp.cpu_count()}")
    print("\nUsage:")
    print("  from sparse_addition_parallel import add_matrices_parallel")
    print("  ")
    print("  add_matrices_parallel(")
    print("      file1='data/output/matrix_a.csv',")
    print("      file2='data/output/matrix_b.csv',")
    print("      output_file='data/output/sum_parallel.csv',")
    print("      num_workers=8  # Use 8 CPU cores")
    print("  )")
    print("="*70)
