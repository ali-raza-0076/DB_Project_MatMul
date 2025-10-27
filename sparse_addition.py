import csv
import heapq
from typing import Iterator, Tuple, List, Optional
import os

class SparseAddition:
    """
    Sparse matrix addition using sort-merge algorithm.
    Assumes input matrices are already sorted by (row, col).
    """
    
    def __init__(self, memory_limit_mb: int = 100):
        """
        Initialize sparse addition handler.
        
        Args:
            memory_limit_mb: Memory limit in MB for block processing
        """
        self.memory_limit_mb = memory_limit_mb
        self.block_size = self._calculate_block_size()
    
    def _calculate_block_size(self) -> int:
        """Calculate number of entries per block based on memory limit."""
        # Assume each entry takes ~40 bytes (row, col, value + overhead)
        bytes_per_entry = 40
        return (self.memory_limit_mb * 1024 * 1024) // bytes_per_entry
    
    def add_matrices(self, input_file1: str, input_file2: str, output_file: str) -> None:
        """
        Main addition algorithm for sparse matrices.
        
        Args:
            input_file1: Path to first sorted COO CSV file
            input_file2: Path to second sorted COO CSV file
            output_file: Path to output COO CSV file
        """
        # Check if matrices are small enough to fit in memory
        size1 = self._count_entries(input_file1)
        size2 = self._count_entries(input_file2)
        
        if size1 + size2 <= self.block_size:
            # Both matrices fit in memory - use simple merge
            self._add_in_memory(input_file1, input_file2, output_file)
        else:
            # Use block-based addition
            self.add_blocked(input_file1, input_file2, output_file)
    
    def merge_sorted_streams(self, stream1: Iterator[Tuple[int, int, float]], 
                           stream2: Iterator[Tuple[int, int, float]]) -> Iterator[Tuple[int, int, float]]:
        """
        Two-pointer merge of two sorted streams of (row, col, value) tuples.
        
        Args:
            stream1: Iterator of (row, col, value) from first matrix
            stream2: Iterator of (row, col, value) from second matrix
            
        Yields:
            Merged (row, col, value) tuples with combined values for same positions
        """
        try:
            entry1 = next(stream1)
            has_entry1 = True
        except StopIteration:
            has_entry1 = False
        
        try:
            entry2 = next(stream2)
            has_entry2 = True
        except StopIteration:
            has_entry2 = False
        
        while has_entry1 or has_entry2:
            if not has_entry1:
                # Only stream2 has entries left
                yield entry2
                for entry in stream2:
                    yield entry
                break
            elif not has_entry2:
                # Only stream1 has entries left
                yield entry1
                for entry in stream1:
                    yield entry
                break
            else:
                # Both streams have entries - compare positions
                row1, col1, val1 = entry1
                row2, col2, val2 = entry2
                
                if (row1, col1) < (row2, col2):
                    # Entry from stream1 comes first
                    yield entry1
                    try:
                        entry1 = next(stream1)
                    except StopIteration:
                        has_entry1 = False
                elif (row1, col1) > (row2, col2):
                    # Entry from stream2 comes first
                    yield entry2
                    try:
                        entry2 = next(stream2)
                    except StopIteration:
                        has_entry2 = False
                else:
                    # Same position - add values
                    result_val = val1 + val2
                    if abs(result_val) > 1e-10:  # Only output non-zero values
                        yield (row1, col1, result_val)
                    try:
                        entry1 = next(stream1)
                    except StopIteration:
                        has_entry1 = False
                    try:
                        entry2 = next(stream2)
                    except StopIteration:
                        has_entry2 = False
    
    def add_blocked(self, input_file1: str, input_file2: str, output_file: str) -> None:
        """
        Block-based addition for large matrices that don't fit in RAM.
        Processes matrices in blocks and merges incrementally.
        
        Args:
            input_file1: Path to first sorted COO CSV file
            input_file2: Path to second sorted COO CSV file
            output_file: Path to output COO CSV file
        """
        with open(output_file, 'w', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(['row', 'col', 'value'])
            
            # Create iterators for both input files
            stream1 = self._read_coo_csv(input_file1)
            stream2 = self._read_coo_csv(input_file2)
            
            # Merge and write in blocks
            block_count = 0
            for entry in self.merge_sorted_streams(stream1, stream2):
                row, col, value = entry
                writer.writerow([row, col, value])
                block_count += 1
                
                # Flush periodically to avoid memory buildup
                if block_count % self.block_size == 0:
                    outf.flush()
    
    def _add_in_memory(self, input_file1: str, input_file2: str, output_file: str) -> None:
        """
        Simple in-memory addition when both matrices fit in RAM.
        
        Args:
            input_file1: Path to first sorted COO CSV file
            input_file2: Path to second sorted COO CSV file
            output_file: Path to output COO CSV file
        """
        stream1 = self._read_coo_csv(input_file1)
        stream2 = self._read_coo_csv(input_file2)
        
        with open(output_file, 'w', newline='') as outf:
            writer = csv.writer(outf)
            writer.writerow(['row', 'col', 'value'])
            
            for entry in self.merge_sorted_streams(stream1, stream2):
                row, col, value = entry
                writer.writerow([row, col, value])
    
    def _read_coo_csv(self, filename: str) -> Iterator[Tuple[int, int, float]]:
        """
        Read COO format CSV file as iterator.
        
        Args:
            filename: Path to COO CSV file
            
        Yields:
            (row, col, value) tuples
        """
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield (int(row['row']), int(row['col']), float(row['value']))
    
    def _count_entries(self, filename: str) -> int:
        """Count number of entries in a COO CSV file."""
        with open(filename, 'r') as f:
            return sum(1 for _ in f) - 1  # Subtract header row


def main():
    """Example usage of SparseAddition."""
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python sparse_addition.py <input1.csv> <input2.csv> <output.csv>")
        sys.exit(1)
    
    input_file1 = sys.argv[1]
    input_file2 = sys.argv[2]
    output_file = sys.argv[3]
    
    adder = SparseAddition(memory_limit_mb=100)
    adder.add_matrices(input_file1, input_file2, output_file)
    print(f"Addition complete. Result written to {output_file}")


if __name__ == "__main__":
    main()
