#!/usr/bin/env python3
"""
Single Gene Pair Correlation Analysis Tool

A command-line tool for exploring gene expression data and computing correlations 
between specific gene pairs using CCC (Clustered Correlation Coefficient), 
Spearman, and Pearson correlation methods.

This script provides two main functionalities:
1. Data exploration: Show available genes and their symbols for a tissue
2. Correlation analysis: Compute correlations for a specific gene pair in a tissue

Author: Generated for CCC-GPU project
Version: 1.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import numpy as np

# Import correlation methods
try:
    from ccc.corr import ccc_gpu, pearson, spearman
except ImportError:
    print("Error: CCC library not found. Please install the ccc package.")
    sys.exit(1)


def setup_logging(debug: bool = False, output_dir: Optional[Path] = None) -> Optional[Path]:
    """Configure logging for the script.
    
    Args:
        debug: Enable debug level logging if True
        output_dir: Directory to write log files to (optional)
        
    Returns:
        Path to log file if output_dir provided, None otherwise
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler (if output directory provided)
    log_file = None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"gene_pair_correlation_analysis_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True
    )
    
    if log_file:
        logging.info(f"Log file created: {log_file}")
        
    return log_file


class GeneExpressionAnalyzer:
    """Main class for gene expression analysis and correlation computation."""
    
    def __init__(self, data_dir: str, gene_mapping_file: str):
        """Initialize the analyzer with data directory and gene mapping file.
        
        Args:
            data_dir: Directory containing tissue expression data files
            gene_mapping_file: Path to gene ID to symbol mapping file
        """
        self.data_dir = Path(data_dir)
        self.gene_mapping_file = Path(gene_mapping_file)
        self._gene_mapping = None
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate that input paths exist and are accessible."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        if not self.gene_mapping_file.exists():
            raise FileNotFoundError(f"Gene mapping file not found: {self.gene_mapping_file}")
    
    @property
    def gene_mapping(self) -> pd.DataFrame:
        """Load and cache gene mapping data."""
        if self._gene_mapping is None:
            logging.info(f"Loading gene mapping from: {self.gene_mapping_file}")
            self._gene_mapping = pd.read_pickle(self.gene_mapping_file)
            logging.info(f"Loaded {len(self._gene_mapping)} gene mappings")
        return self._gene_mapping
    
    def list_available_tissues(self) -> list:
        """Get list of available tissue files.
        
        Returns:
            List of tissue names (without file extensions)
        """
        tissue_files = list(self.data_dir.glob("gtex_v8_data_*.pkl"))
        tissues = [f.stem.replace("gtex_v8_data_", "") for f in tissue_files]
        return sorted(tissues)
    
    def _find_tissue_file(self, tissue: str) -> Path:
        """Find the tissue file for a given tissue name.
        
        Args:
            tissue: Tissue name
            
        Returns:
            Path to tissue file
            
        Raises:
            FileNotFoundError: If tissue file is not found
        """
        # Try exact match first
        exact_file = self.data_dir / f"gtex_v8_data_{tissue}.pkl"
        if exact_file.exists():
            return exact_file
        
        # Try partial matching
        tissue_files = list(self.data_dir.glob(f"gtex_v8_data_*{tissue}*.pkl"))
        if len(tissue_files) == 1:
            return tissue_files[0]
        elif len(tissue_files) > 1:
            matches = [f.stem for f in tissue_files]
            raise ValueError(
                f"Multiple tissue files match '{tissue}': {matches}. "
                "Please be more specific."
            )
        else:
            available = self.list_available_tissues()
            raise FileNotFoundError(
                f"No tissue file found for '{tissue}'. "
                f"Available tissues: {available[:10]}..." if len(available) > 10 
                else f"Available tissues: {available}"
            )
    
    def show_tissue_genes(self, tissue: str, n_genes: int = 20) -> None:
        """Display available genes and their symbols for a tissue.
        
        Args:
            tissue: Tissue name
            n_genes: Number of genes to display (default: 20)
        """
        # Load tissue data
        tissue_file = self._find_tissue_file(tissue)
        logging.info(f"Loading tissue data from: {tissue_file}")
        
        tissue_data = pd.read_pickle(tissue_file)
        logging.info(f"Tissue data shape: {tissue_data.shape}")
        
        # Get gene IDs and map to symbols
        gene_ids = tissue_data.index.tolist()
        
        # Create mapping lookup for faster access
        gene_mapping = self.gene_mapping.set_index('gene_ens_id')
        
        print(f"\n=== Tissue: {tissue} ===")
        print(f"Total genes: {len(gene_ids):,}")
        print(f"Total samples: {tissue_data.shape[1]:,}")
        print(f"\nFirst {n_genes} genes:")
        print("-" * 60)
        print(f"{'#':<4} {'Gene Symbol':<15} {'Ensembl ID':<20}")
        print("-" * 60)
        
        for i, gene_id in enumerate(gene_ids[:n_genes], 1):
            # Remove version from gene ID for mapping lookup
            clean_gene_id = gene_id.split('.')[0] if '.' in gene_id else gene_id
            
            # Look up symbol
            symbol = "N/A"
            if gene_id in gene_mapping.index:
                symbol = gene_mapping.loc[gene_id, 'gene_symbol']
            elif clean_gene_id in gene_mapping.index:
                symbol = gene_mapping.loc[clean_gene_id, 'gene_symbol']
            else:
                # Search in original mapping
                matches = self.gene_mapping[
                    self.gene_mapping['gene_ens_id'].str.startswith(clean_gene_id)
                ]
                if len(matches) > 0:
                    symbol = matches.iloc[0]['gene_symbol']
            
            print(f"{i:<4} {symbol:<15} {gene_id:<20}")
        
        if len(gene_ids) > n_genes:
            print(f"... and {len(gene_ids) - n_genes:,} more genes")
        print()
    
    def _resolve_gene(self, gene_input: str) -> Tuple[str, str]:
        """Resolve gene input to Ensembl ID and symbol.
        
        Args:
            gene_input: Gene symbol or Ensembl ID
            
        Returns:
            Tuple of (ensembl_id, gene_symbol)
            
        Raises:
            ValueError: If gene cannot be resolved
        """
        # Check if it's already an Ensembl ID
        if gene_input.startswith('ENSG'):
            # Look up the symbol
            matches = self.gene_mapping[self.gene_mapping['gene_ens_id'] == gene_input]
            if len(matches) == 0:
                # Try without version
                base_id = gene_input.split('.')[0]
                matches = self.gene_mapping[
                    self.gene_mapping['gene_ens_id'].str.startswith(base_id)
                ]
            
            if len(matches) > 0:
                return matches.iloc[0]['gene_ens_id'], matches.iloc[0]['gene_symbol']
            else:
                raise ValueError(f"Ensembl ID '{gene_input}' not found in mapping")
        else:
            # Assume it's a gene symbol
            matches = self.gene_mapping[self.gene_mapping['gene_symbol'] == gene_input]
            if len(matches) > 0:
                return matches.iloc[0]['gene_ens_id'], matches.iloc[0]['gene_symbol']
            else:
                # Try case-insensitive search
                matches = self.gene_mapping[
                    self.gene_mapping['gene_symbol'].str.upper() == gene_input.upper()
                ]
                if len(matches) > 0:
                    return matches.iloc[0]['gene_ens_id'], matches.iloc[0]['gene_symbol']
                else:
                    raise ValueError(
                        f"Gene symbol '{gene_input}' not found. "
                        "Use --show-genes to see available genes."
                    )
    
    def compute_gene_pair_correlations(
        self, 
        gene1: str, 
        gene2: str, 
        tissue: str
    ) -> Dict[str, Union[float, str]]:
        """Compute correlations between two genes in a specific tissue.
        
        Args:
            gene1: First gene (symbol or Ensembl ID)
            gene2: Second gene (symbol or Ensembl ID)  
            tissue: Tissue name
            
        Returns:
            Dictionary with correlation results
        """
        # Resolve genes
        gene1_id, gene1_symbol = self._resolve_gene(gene1)
        gene2_id, gene2_symbol = self._resolve_gene(gene2)
        
        # Load tissue data
        tissue_file = self._find_tissue_file(tissue)
        logging.info(f"Loading tissue data from: {tissue_file}")
        
        tissue_data = pd.read_pickle(tissue_file)
        logging.info(f"Tissue data shape: {tissue_data.shape}")
        
        # Extract gene expression data
        gene1_expr = self._extract_gene_expression(tissue_data, gene1_id, gene1_symbol)
        gene2_expr = self._extract_gene_expression(tissue_data, gene2_id, gene2_symbol)
        
        # Ensure we have the same samples
        common_samples = gene1_expr.index.intersection(gene2_expr.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between the two genes")
        
        gene1_values = gene1_expr.loc[common_samples].values
        gene2_values = gene2_expr.loc[common_samples].values
        
        # Remove any NaN values
        mask = ~(np.isnan(gene1_values) | np.isnan(gene2_values))
        gene1_clean = gene1_values[mask]
        gene2_clean = gene2_values[mask]
        
        if len(gene1_clean) < 3:
            raise ValueError("Insufficient valid data points for correlation analysis")
        
        logging.info(f"Computing correlations for {len(gene1_clean)} samples")
        
        # Compute correlations
        results = {
            'gene1_symbol': gene1_symbol,
            'gene1_ensembl_id': gene1_id,
            'gene2_symbol': gene2_symbol, 
            'gene2_ensembl_id': gene2_id,
            'tissue': tissue,
            'n_samples': len(gene1_clean),
        }
        
        # Create DataFrame for correlation computation (genes as rows, samples as columns)
        # This matches the format expected by ccc.corr functions
        data_df = pd.DataFrame({
            f'sample_{i}': [gene1_clean[i], gene2_clean[i]] 
            for i in range(len(gene1_clean))
        }, index=[gene1_symbol, gene2_symbol])
        
        try:
            # Compute CCC
            logging.info("Computing CCC correlation...")
            ccc_result = ccc_gpu(data_df, n_jobs=1)  # Use single job for pair
            results['ccc'] = float(ccc_result.iloc[0, 1])  # Off-diagonal element
        except Exception as e:
            logging.warning(f"CCC computation failed: {e}")
            results['ccc'] = None
        
        try:
            # Compute Pearson correlation
            logging.info("Computing Pearson correlation...")
            pearson_result = pearson(data_df)
            results['pearson'] = float(pearson_result.iloc[0, 1])
        except Exception as e:
            logging.warning(f"Pearson computation failed: {e}")
            results['pearson'] = None
        
        try:
            # Compute Spearman correlation
            logging.info("Computing Spearman correlation...")
            spearman_result = spearman(data_df)
            results['spearman'] = float(spearman_result.iloc[0, 1])
        except Exception as e:
            logging.warning(f"Spearman computation failed: {e}")
            results['spearman'] = None
        
        return results
    
    def _extract_gene_expression(self, tissue_data: pd.DataFrame, gene_id: str, gene_symbol: str) -> pd.Series:
        """Extract expression data for a specific gene.
        
        Args:
            tissue_data: Tissue expression DataFrame
            gene_id: Ensembl gene ID
            gene_symbol: Gene symbol
            
        Returns:
            Series with gene expression values
            
        Raises:
            ValueError: If gene is not found in tissue data
        """
        # Try exact match first
        if gene_id in tissue_data.index:
            return tissue_data.loc[gene_id]
        
        # Try without version
        base_id = gene_id.split('.')[0]
        matches = [idx for idx in tissue_data.index if idx.startswith(base_id)]
        
        if len(matches) == 1:
            return tissue_data.loc[matches[0]]
        elif len(matches) > 1:
            logging.warning(f"Multiple matches for {gene_symbol} ({gene_id}), using first match")
            return tissue_data.loc[matches[0]]
        else:
            raise ValueError(f"Gene {gene_symbol} ({gene_id}) not found in tissue data")


def save_results(results: Dict[str, Union[float, str]], output_dir: Path) -> Tuple[Path, Path]:
    """Save correlation results to files.
    
    Args:
        results: Dictionary containing correlation results
        output_dir: Directory to save files
        
    Returns:
        Tuple of (json_file_path, pickle_file_path)
    """
    import json
    import pickle
    from datetime import datetime
    
    # Create filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gene1_symbol = results['gene1_symbol']
    gene2_symbol = results['gene2_symbol'] 
    tissue = results['tissue']
    
    base_filename = f"{gene1_symbol}_{gene2_symbol}_{tissue}_{timestamp}"
    json_file = output_dir / f"{base_filename}_correlation_results.json"
    pickle_file = output_dir / f"{base_filename}_correlation_results.pkl"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON (human readable)
    json_data = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str)):
            json_data[key] = value
        else:
            json_data[key] = str(value)
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Save as pickle (preserves data types)
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    logging.info(f"Results saved to: {json_file}")
    logging.info(f"Results saved to: {pickle_file}")
    
    return json_file, pickle_file


def main():
    """Main function to handle command line arguments and execute analysis."""
    parser = argparse.ArgumentParser(
        description="Single Gene Pair Correlation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show available tissues
  python compute_single_gene_pair_correlations_cli.py --list-tissues
  
  # Show genes in whole blood tissue
  python compute_single_gene_pair_correlations_cli.py --show-genes whole_blood
  
  # Compute correlations between TP53 and BRCA1 in whole blood
  python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue whole_blood
  
  # Save results and logs to output directory
  python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue liver \\
    --output-dir ./results --debug
  
  # Use custom data directory and gene mapping
  python compute_single_gene_pair_correlations_cli.py TP53 BRCA1 --tissue liver \\
    --data-dir /custom/path/data \\
    --gene-mapping /custom/path/mappings.pkl \\
    --output-dir ./custom_results
        """
    )
    
    # Positional arguments for gene pair analysis
    parser.add_argument(
        'genes',
        nargs='*',
        help='Two gene symbols or Ensembl IDs for correlation analysis (e.g., TP53 BRCA1)'
    )
    
    # Main options
    parser.add_argument(
        '--tissue',
        type=str,
        help='Tissue name for analysis (required for correlation analysis)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/mnt/data/proj_data/ccc-gpu/data/tutorial/data_by_tissue',
        help='Directory containing tissue expression data files'
    )
    
    parser.add_argument(
        '--gene-mapping',
        type=str,
        default='/mnt/data/proj_data/ccc-gpu/data/tutorial/gtex_gene_id_symbol_mappings.pkl',
        help='Path to gene ID to symbol mapping file'
    )
    
    # Discovery options
    parser.add_argument(
        '--list-tissues',
        action='store_true',
        help='List all available tissues and exit'
    )
    
    parser.add_argument(
        '--show-genes',
        type=str,
        metavar='TISSUE',
        help='Show available genes for specified tissue and exit'
    )
    
    parser.add_argument(
        '--n-genes',
        type=int,
        default=20,
        help='Number of genes to show (default: 20)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save output files and logs (optional)'
    )
    
    # Utility options
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Setup logging
    log_file = setup_logging(debug=args.debug, output_dir=output_dir)
    
    try:
        # Initialize analyzer
        analyzer = GeneExpressionAnalyzer(args.data_dir, args.gene_mapping)
        
        # Handle discovery commands
        if args.list_tissues:
            tissues = analyzer.list_available_tissues()
            print(f"\n=== Available Tissues ({len(tissues)}) ===")
            for i, tissue in enumerate(tissues, 1):
                print(f"{i:2d}. {tissue}")
            print()
            return
        
        if args.show_genes:
            analyzer.show_tissue_genes(args.show_genes, args.n_genes)
            return
        
        # Handle correlation analysis
        if len(args.genes) != 2:
            parser.error(
                "Exactly two genes are required for correlation analysis. "
                "Use --show-genes to see available genes, or --list-tissues to see available tissues."
            )
        
        if not args.tissue:
            parser.error(
                "Tissue is required for correlation analysis. "
                "Use --list-tissues to see available tissues."
            )
        
        gene1, gene2 = args.genes
        results = analyzer.compute_gene_pair_correlations(gene1, gene2, args.tissue)
        
        # Save results to files if output directory provided
        saved_files = None
        if output_dir:
            try:
                saved_files = save_results(results, output_dir)
                logging.info(f"Results saved to output directory: {output_dir}")
            except Exception as e:
                logging.error(f"Failed to save results: {e}")
        
        # Print results
        print("\n" + "="*60)
        print("GENE PAIR CORRELATION RESULTS")
        print("="*60)
        print(f"Gene 1: {results['gene1_symbol']} ({results['gene1_ensembl_id']})")
        print(f"Gene 2: {results['gene2_symbol']} ({results['gene2_ensembl_id']})")
        print(f"Tissue: {results['tissue']}")
        print(f"Samples: {results['n_samples']:,}")
        print("-" * 60)
        
        for method in ['ccc', 'pearson', 'spearman']:
            value = results.get(method)
            if value is not None:
                print(f"{method.upper():>12}: {value:.6f}")
            else:
                print(f"{method.upper():>12}: Failed to compute")
        
        print("="*60)
        
        # Show saved files info
        if saved_files:
            print(f"Results saved to:")
            print(f"  JSON: {saved_files[0].name}")
            print(f"  Pickle: {saved_files[1].name}")
        
        if log_file:
            print(f"Log file: {log_file.name}")
        
        print()
        
        # Also return as dict for programmatic use
        return results
        
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 