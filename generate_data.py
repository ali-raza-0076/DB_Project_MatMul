"""
Sparse Matrix Data Generator
Generates synthetic sparse matrices in COO format (CSV) for testing.

Features:
- Control matrix size and sparsity
- Memory estimation before generation
- Various sparsity patterns
- Progress tracking
- Safe generation (won't crash your computer)
"""

import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional
from tqdm import tqdm
import sys


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class SparseMatrixGenerator:
    """Generate synthetic sparse matrices for testing."""
    
    def __init__(self, output_dir: str = "data/input"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def estimate_memory(self, num_rows: int, num_cols: int, nnz: int) -> dict:
        """
        Estimate memory requirements for generating and storing the matrix.
        
        Args:
            num_rows: Number of rows
            num_cols: Number of columns
            nnz: Number of nonzeros
        
        Returns:
            Dictionary with memory estimates in MB
        """
        # Each entry: row_idx (4 bytes) + col_idx (4 bytes) + value (8 bytes) = 16 bytes
        memory_generation_mb = (nnz * 16) / (1024 * 1024)
        
        # CSV file size: roughly 20-25 bytes per entry (including formatting)
        csv_size_mb = (nnz * 25) / (1024 * 1024)
        
        return {
            'generation_mb': memory_generation_mb,
            'csv_size_mb': csv_size_mb,
            'total_mb': memory_generation_mb + csv_size_mb
        }
    
    def check_safety(self, num_rows: int, num_cols: int, nnz: int, max_memory_mb: float = 500):
        """
        Check if generation is safe (won't crash computer).
        
        Args:
            num_rows, num_cols, nnz: Matrix parameters
            max_memory_mb: Maximum allowed memory (default 500 MB)
        
        Raises:
            ValueError if unsafe
        """
        estimates = self.estimate_memory(num_rows, num_cols, nnz)
        
        if estimates['total_mb'] > max_memory_mb:
            raise ValueError(
                f"Matrix too large! Estimated memory: {estimates['total_mb']:.1f} MB\n"
                f"Maximum allowed: {max_memory_mb} MB\n"
                f"Suggestion: Reduce nnz to {int(nnz * max_memory_mb / estimates['total_mb'])}"
            )
        
        logger.info(f"Memory estimate: {estimates['total_mb']:.1f} MB (SAFE)")
    
    def generate_random(
        self,
        num_rows: int,
        num_cols: int,
        nnz: int,
        filename: str,
        seed: Optional[int] = None,
        check_duplicates: bool = False
    ) -> str:
        """
        Generate random sparse matrix with uniform distribution.
        
        Args:
            num_rows: Number of rows
            num_cols: Number of columns
            nnz: Number of nonzeros
            filename: Output CSV filename
            seed: Random seed for reproducibility
            check_duplicates: If True, ensure no duplicate (i,j) pairs
        
        Returns:
            Path to generated file
        """
        logger.info(f"Generating random matrix: {num_rows}×{num_cols}, {nnz:,} nonzeros")
        
        # Safety check
        self.check_safety(num_rows, num_cols, nnz)
        
        if seed is not None:
            np.random.seed(seed)
        
        filepath = self.output_dir / filename
        
        if check_duplicates:
            # Slower but ensures no duplicates
            logger.info("Ensuring unique (row, col) pairs...")
            total_possible = num_rows * num_cols
            
            if nnz > total_possible:
                raise ValueError(f"Cannot generate {nnz} unique entries in {num_rows}×{num_cols} matrix")
            
            # Sample unique positions
            positions = np.random.choice(total_possible, size=nnz, replace=False)
            rows = positions // num_cols
            cols = positions % num_cols
            values = np.random.randint(-100, 100, nnz)  # Random integers between -100 and 100
        else:
            # Faster, may have duplicates
            rows = np.random.randint(0, num_rows, nnz)
            cols = np.random.randint(0, num_cols, nnz)
            values = np.random.randint(-100, 100, nnz)
        
        # Write to CSV with progress bar
        logger.info(f"Writing to {filepath}...")
        with open(filepath, 'w') as f:
            for i in tqdm(range(nnz), desc="Writing entries", unit=" entries"):
                f.write(f"{rows[i]},{cols[i]},{values[i]}\n")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Generated {filepath} ({file_size_mb:.1f} MB)")
        
        return str(filepath)
    
    def generate_banded(
        self,
        size: int,
        bandwidth: int,
        filename: str,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate banded matrix (nonzeros near diagonal).
        Common in differential equations and physics simulations.
        
        Args:
            size: Matrix size (size × size)
            bandwidth: Number of diagonals on each side of main diagonal
            filename: Output CSV filename
            seed: Random seed
        
        Returns:
            Path to generated file
        """
        logger.info(f"Generating banded matrix: {size}×{size}, bandwidth={bandwidth}")
        
        if seed is not None:
            np.random.seed(seed)
        
        filepath = self.output_dir / filename
        
        entries = []
        for i in range(size):
            for k in range(-bandwidth, bandwidth + 1):
                j = i + k
                if 0 <= j < size:
                    value = np.random.randn()
                    entries.append((i, j, value))
        
        nnz = len(entries)
        self.check_safety(size, size, nnz)
        
        logger.info(f"Writing {nnz:,} entries to {filepath}...")
        with open(filepath, 'w') as f:
            for i, j, v in tqdm(entries, desc="Writing entries"):
                f.write(f"{i},{j},{v}\n")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Generated {filepath} ({file_size_mb:.1f} MB)")
        
        return str(filepath)
    
    def generate_block_sparse(
        self,
        num_blocks: int,
        block_size: int,
        block_density: float,
        filename: str,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate block-sparse matrix (common in graph partitioning, neural networks).
        
        Args:
            num_blocks: Number of blocks along diagonal
            block_size: Size of each block
            block_density: Density within each block (0.0 to 1.0)
            filename: Output CSV filename
            seed: Random seed
        
        Returns:
            Path to generated file
        """
        logger.info(f"Generating block-sparse matrix: {num_blocks} blocks of {block_size}×{block_size}")
        
        if seed is not None:
            np.random.seed(seed)
        
        size = num_blocks * block_size
        filepath = self.output_dir / filename
        
        entries = []
        for block in range(num_blocks):
            row_start = block * block_size
            col_start = block * block_size
            
            # Generate dense entries within block
            block_nnz = int(block_size * block_size * block_density)
            local_rows = np.random.randint(0, block_size, block_nnz)
            local_cols = np.random.randint(0, block_size, block_nnz)
            values = np.random.randn(block_nnz)
            
            for lr, lc, v in zip(local_rows, local_cols, values):
                entries.append((row_start + lr, col_start + lc, v))
        
        nnz = len(entries)
        self.check_safety(size, size, nnz)
        
        logger.info(f"Writing {nnz:,} entries to {filepath}...")
        with open(filepath, 'w') as f:
            for i, j, v in tqdm(entries, desc="Writing entries"):
                f.write(f"{i},{j},{v}\n")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Generated {filepath} ({file_size_mb:.1f} MB)")
        
        return str(filepath)
    
    def generate_power_law(
        self,
        num_rows: int,
        num_cols: int,
        nnz: int,
        alpha: float,
        filename: str,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate matrix with power-law degree distribution.
        Common in social networks, web graphs.
        
        Args:
            num_rows, num_cols: Matrix dimensions
            nnz: Number of nonzeros
            alpha: Power-law exponent (typically 2-3)
            filename: Output CSV filename
            seed: Random seed
        
        Returns:
            Path to generated file
        """
        logger.info(f"Generating power-law matrix: {num_rows}×{num_cols}, alpha={alpha}")
        
        self.check_safety(num_rows, num_cols, nnz)
        
        if seed is not None:
            np.random.seed(seed)
        
        filepath = self.output_dir / filename
        
        # Generate row degrees following power law
        row_degrees = np.random.pareto(alpha, num_rows).astype(int) + 1
        row_degrees = np.clip(row_degrees, 1, num_cols)
        
        # Normalize to get exactly nnz entries
        row_degrees = (row_degrees / row_degrees.sum() * nnz).astype(int)
        
        # Adjust to match exact nnz
        diff = nnz - row_degrees.sum()
        if diff > 0:
            row_degrees[:diff] += 1
        elif diff < 0:
            row_degrees[:abs(diff)] -= 1
        
        logger.info(f"Writing {nnz:,} entries to {filepath}...")
        with open(filepath, 'w') as f:
            for i in tqdm(range(num_rows), desc="Writing entries"):
                degree = row_degrees[i]
                if degree > 0:
                    cols = np.random.choice(num_cols, size=degree, replace=False)
                    values = np.random.randn(degree)
                    
                    for j, v in zip(cols, values):
                        f.write(f"{i},{j},{v}\n")
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Generated {filepath} ({file_size_mb:.1f} MB)")
        
        return str(filepath)


def generate_preset_matrices(output_dir: str = "data/input"):
    """Generate common test matrices for the project."""
    generator = SparseMatrixGenerator(output_dir)
    
    presets = [
        # Small test matrices
        ("small_A.csv", "random", {"num_rows": 100, "num_cols": 100, "nnz": 500}),
        ("small_B.csv", "random", {"num_rows": 100, "num_cols": 100, "nnz": 500}),
        
        # Medium matrices (for testing algorithms)
        ("medium_A.csv", "random", {"num_rows": 1000, "num_cols": 1000, "nnz": 10000}),
        ("medium_B.csv", "random", {"num_rows": 1000, "num_cols": 1000, "nnz": 10000}),
        
        # Large matrices (for performance testing)
        ("large_A.csv", "random", {"num_rows": 10000, "num_cols": 10000, "nnz": 100000}),
        ("large_B.csv", "random", {"num_rows": 10000, "num_cols": 10000, "nnz": 100000}),
        
        # Very large (1 million entries)
        ("xlarge_A.csv", "random", {"num_rows": 50000, "num_cols": 50000, "nnz": 1000000}),
        ("xlarge_B.csv", "random", {"num_rows": 50000, "num_cols": 50000, "nnz": 1000000}),
        
        # Special patterns
        ("banded_1000.csv", "banded", {"size": 1000, "bandwidth": 5}),
        ("block_sparse_1000.csv", "block_sparse", {"num_blocks": 10, "block_size": 100, "block_density": 0.1}),
    ]
    
    logger.info(f"\nGenerating {len(presets)} preset matrices...")
    logger.info("=" * 70)
    
    generated_files = []
    
    for filename, pattern, params in presets:
        try:
            logger.info(f"\nGenerating {filename}...")
            
            if pattern == "random":
                filepath = generator.generate_random(**params, filename=filename, seed=42)
            elif pattern == "banded":
                filepath = generator.generate_banded(**params, filename=filename, seed=42)
            elif pattern == "block_sparse":
                filepath = generator.generate_block_sparse(**params, filename=filename, seed=42)
            
            generated_files.append(filepath)
            
        except ValueError as e:
            logger.warning(f"Skipped {filename}: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✓ Generated {len(generated_files)} matrices in {output_dir}")
    logger.info("=" * 70)
    
    return generated_files


def main():
    """Command-line interface for data generation."""
    parser = argparse.ArgumentParser(
        description="Generate sparse matrices for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all preset matrices
  python generate_data.py --preset
  
  # Generate custom random matrix (1 million entries)
  python generate_data.py --random --rows 50000 --cols 50000 --nnz 1000000 -o my_matrix.csv
  
  # Generate banded matrix
  python generate_data.py --banded --size 5000 --bandwidth 10 -o banded.csv
  
  # Generate block-sparse matrix
  python generate_data.py --block --num-blocks 20 --block-size 100 --density 0.05 -o blocks.csv
        """
    )
    
    parser.add_argument('--output-dir', default='data/input', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-memory', type=float, default=500, help='Max memory in MB (safety limit)')
    
    # Preset matrices
    parser.add_argument('--preset', action='store_true', help='Generate all preset test matrices')
    
    # Custom generation options
    parser.add_argument('--random', action='store_true', help='Generate random matrix')
    parser.add_argument('--banded', action='store_true', help='Generate banded matrix')
    parser.add_argument('--block', action='store_true', help='Generate block-sparse matrix')
    parser.add_argument('--power-law', action='store_true', help='Generate power-law matrix')
    
    # Matrix parameters
    parser.add_argument('--rows', type=int, help='Number of rows')
    parser.add_argument('--cols', type=int, help='Number of columns')
    parser.add_argument('--nnz', type=int, help='Number of nonzeros')
    parser.add_argument('--size', type=int, help='Matrix size (for square matrices)')
    parser.add_argument('--bandwidth', type=int, help='Bandwidth for banded matrices')
    parser.add_argument('--num-blocks', type=int, help='Number of blocks')
    parser.add_argument('--block-size', type=int, help='Block size')
    parser.add_argument('--density', type=float, help='Density within blocks (0.0-1.0)')
    parser.add_argument('--alpha', type=float, default=2.5, help='Power-law exponent')
    
    parser.add_argument('-o', '--output', help='Output filename')
    
    args = parser.parse_args()
    
    generator = SparseMatrixGenerator(args.output_dir)
    
    # Preset matrices
    if args.preset:
        generate_preset_matrices(args.output_dir)
        return
    
    # Custom generation
    if args.random:
        if not all([args.rows, args.cols, args.nnz, args.output]):
            parser.error("--random requires --rows, --cols, --nnz, and -o")
        
        generator.generate_random(
            num_rows=args.rows,
            num_cols=args.cols,
            nnz=args.nnz,
            filename=args.output,
            seed=args.seed
        )
    
    elif args.banded:
        if not all([args.size, args.bandwidth, args.output]):
            parser.error("--banded requires --size, --bandwidth, and -o")
        
        generator.generate_banded(
            size=args.size,
            bandwidth=args.bandwidth,
            filename=args.output,
            seed=args.seed
        )
    
    elif args.block:
        if not all([args.num_blocks, args.block_size, args.density, args.output]):
            parser.error("--block requires --num-blocks, --block-size, --density, and -o")
        
        generator.generate_block_sparse(
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            block_density=args.density,
            filename=args.output,
            seed=args.seed
        )
    
    elif args.power_law:
        if not all([args.rows, args.cols, args.nnz, args.output]):
            parser.error("--power-law requires --rows, --cols, --nnz, and -o")
        
        generator.generate_power_law(
            num_rows=args.rows,
            num_cols=args.cols,
            nnz=args.nnz,
            alpha=args.alpha,
            filename=args.output,
            seed=args.seed
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
