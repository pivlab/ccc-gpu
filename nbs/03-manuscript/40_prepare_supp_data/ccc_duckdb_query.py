#!/usr/bin/env python3
"""
Python wrapper for fast CCC correlation queries from DuckDB databases.

This module provides a simple interface for querying gene pair correlations
from the DuckDB databases created by process_ccc_to_duckdb.py.

Example usage:
    from ccc_duckdb_query import CCCDatabase

    # Single tissue database
    db = CCCDatabase("/path/to/bladder_ccc.duckdb")

    # Query single gene pair
    ccc_value = db.get_correlation("ENSG00000141510.16", "ENSG00000133703.11")

    # Get all correlations for a gene
    correlations = db.get_gene_correlations("ENSG00000141510.16")

    # Get top correlations
    top_pairs = db.get_top_correlations(threshold=0.9, limit=100)

    # Batch query multiple pairs
    pairs = [("gene1", "gene2"), ("gene3", "gene4")]
    results = db.get_batch_correlations(pairs)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import duckdb
import logging

logger = logging.getLogger(__name__)


class CCCDatabase:
    """Wrapper for querying CCC correlation data from DuckDB databases."""

    def __init__(self, db_path: Union[str, Path], tissue: Optional[str] = None):
        """
        Initialize connection to DuckDB database.

        Args:
            db_path: Path to DuckDB database file
            tissue: Tissue name (for consolidated database)
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.con = duckdb.connect(str(self.db_path), read_only=True)
        self.tissue = tissue

        # Detect database type (single tissue or consolidated)
        tables = self.con.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]

        if "tissues" in table_names:
            # Consolidated database
            self.db_type = "consolidated"
            self.tissues = self._get_available_tissues()

            if tissue:
                if tissue not in self.tissues:
                    raise ValueError(f"Tissue '{tissue}' not found. Available: {self.tissues}")
                self.table_name = f"ccc_{tissue}"
            else:
                logger.info(f"Consolidated database with {len(self.tissues)} tissues")
                logger.info(f"Available tissues: {', '.join(self.tissues[:5])}...")
        else:
            # Single tissue database
            self.db_type = "single"
            self.table_name = "ccc_data"
            self.tissues = None

    def _get_available_tissues(self) -> List[str]:
        """Get list of available tissues in consolidated database."""
        result = self.con.execute("SELECT tissue_name FROM tissues ORDER BY tissue_name").fetchall()
        return [r[0] for r in result]

    def get_correlation(self, gene1: str, gene2: str) -> Optional[float]:
        """
        Get CCC correlation for a specific gene pair.

        Args:
            gene1: First gene ID
            gene2: Second gene ID

        Returns:
            CCC correlation value or None if not found
        """
        if self.tissue is None and self.db_type == "consolidated":
            raise ValueError("Tissue must be specified for consolidated database")

        # Try both orientations since correlation is symmetric
        query = f"""
            SELECT ccc FROM {self.table_name}
            WHERE (gene1 = ? AND gene2 = ?)
               OR (gene1 = ? AND gene2 = ?)
            LIMIT 1
        """

        result = self.con.execute(query, [gene1, gene2, gene2, gene1]).fetchone()
        return result[0] if result else None

    def get_gene_correlations(
        self,
        gene: str,
        min_ccc: Optional[float] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get all correlations for a specific gene.

        Args:
            gene: Gene ID
            min_ccc: Minimum CCC threshold (optional)
            limit: Maximum number of results (optional)

        Returns:
            DataFrame with columns: gene_pair, ccc
        """
        if self.tissue is None and self.db_type == "consolidated":
            raise ValueError("Tissue must be specified for consolidated database")

        where_clause = ""
        if min_ccc is not None:
            where_clause = f"AND ccc >= {min_ccc}"

        limit_clause = ""
        if limit is not None:
            limit_clause = f"LIMIT {limit}"

        query = f"""
            SELECT
                CASE
                    WHEN gene1 = ? THEN gene2
                    ELSE gene1
                END as gene_pair,
                ccc
            FROM {self.table_name}
            WHERE (gene1 = ? OR gene2 = ?)
                {where_clause}
            ORDER BY ccc DESC
            {limit_clause}
        """

        result = self.con.execute(query, [gene, gene, gene]).df()
        return result

    def get_top_correlations(
        self,
        threshold: float = 0.9,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get top correlations above a threshold.

        Args:
            threshold: Minimum CCC value
            limit: Maximum number of results

        Returns:
            DataFrame with columns: gene1, gene2, ccc
        """
        if self.tissue is None and self.db_type == "consolidated":
            raise ValueError("Tissue must be specified for consolidated database")

        query = f"""
            SELECT gene1, gene2, ccc
            FROM {self.table_name}
            WHERE ccc >= ?
            ORDER BY ccc DESC
            LIMIT ?
        """

        result = self.con.execute(query, [threshold, limit]).df()
        return result

    def get_batch_correlations(
        self,
        pairs: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Optional[float]]:
        """
        Get correlations for multiple gene pairs efficiently.

        Args:
            pairs: List of (gene1, gene2) tuples

        Returns:
            Dictionary mapping (gene1, gene2) to CCC values
        """
        if self.tissue is None and self.db_type == "consolidated":
            raise ValueError("Tissue must be specified for consolidated database")

        if not pairs:
            return {}

        # Create temporary table for batch lookup
        self.con.execute("CREATE TEMPORARY TABLE query_pairs (gene1 VARCHAR, gene2 VARCHAR)")

        # Insert pairs
        for g1, g2 in pairs:
            self.con.execute("INSERT INTO query_pairs VALUES (?, ?)", [g1, g2])

        # Batch query with joins
        query = f"""
            SELECT
                COALESCE(qp.gene1, qp2.gene1) as query_gene1,
                COALESCE(qp.gene2, qp2.gene2) as query_gene2,
                c.ccc
            FROM query_pairs qp
            LEFT JOIN {self.table_name} c
                ON qp.gene1 = c.gene1 AND qp.gene2 = c.gene2
            LEFT JOIN query_pairs qp2
                ON qp2.gene1 = c.gene2 AND qp2.gene2 = c.gene1
            WHERE c.ccc IS NOT NULL
        """

        results = self.con.execute(query).fetchall()

        # Drop temporary table
        self.con.execute("DROP TABLE query_pairs")

        # Convert to dictionary
        result_dict = {}
        for row in results:
            result_dict[(row[0], row[1])] = row[2]

        # Add None for missing pairs
        for pair in pairs:
            if pair not in result_dict and (pair[1], pair[0]) not in result_dict:
                result_dict[pair] = None

        return result_dict

    def get_cross_tissue_correlation(
        self,
        gene1: str,
        gene2: str
    ) -> pd.DataFrame:
        """
        Get correlation values across all tissues (consolidated database only).

        Args:
            gene1: First gene ID
            gene2: Second gene ID

        Returns:
            DataFrame with columns: tissue, ccc
        """
        if self.db_type != "consolidated":
            raise ValueError("Cross-tissue query requires consolidated database")

        query = """
            SELECT tissue, ccc
            FROM all_correlations
            WHERE (gene1 = ? AND gene2 = ?)
               OR (gene1 = ? AND gene2 = ?)
            ORDER BY ccc DESC
        """

        result = self.con.execute(query, [gene1, gene2, gene2, gene1]).df()
        return result

    def query(self, sql: str, parameters: Optional[List] = None) -> pd.DataFrame:
        """
        Execute custom SQL query on the database.

        Args:
            sql: SQL query string
            parameters: Query parameters (optional)

        Returns:
            Query results as DataFrame
        """
        if parameters:
            return self.con.execute(sql, parameters).df()
        else:
            return self.con.execute(sql).df()

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats = {}

        if self.db_type == "consolidated":
            # Get tissue statistics
            tissue_stats = self.con.execute("""
                SELECT
                    COUNT(*) as num_tissues,
                    SUM(num_pairs) as total_pairs,
                    MIN(min_ccc) as global_min_ccc,
                    MAX(max_ccc) as global_max_ccc,
                    AVG(mean_ccc) as avg_mean_ccc
                FROM tissues
            """).fetchone()

            stats['type'] = 'consolidated'
            stats['num_tissues'] = tissue_stats[0]
            stats['total_pairs'] = tissue_stats[1]
            stats['global_min_ccc'] = tissue_stats[2]
            stats['global_max_ccc'] = tissue_stats[3]
            stats['avg_mean_ccc'] = tissue_stats[4]

            if self.tissue:
                # Get specific tissue stats
                tissue_info = self.con.execute("""
                    SELECT num_pairs, min_ccc, max_ccc, mean_ccc
                    FROM tissues
                    WHERE tissue_name = ?
                """, [self.tissue]).fetchone()

                if tissue_info:
                    stats['tissue'] = self.tissue
                    stats['tissue_pairs'] = tissue_info[0]
                    stats['tissue_min_ccc'] = tissue_info[1]
                    stats['tissue_max_ccc'] = tissue_info[2]
                    stats['tissue_mean_ccc'] = tissue_info[3]

        else:
            # Single tissue database statistics
            result = self.con.execute(f"""
                SELECT
                    COUNT(*) as num_pairs,
                    MIN(ccc) as min_ccc,
                    MAX(ccc) as max_ccc,
                    AVG(ccc) as mean_ccc
                FROM {self.table_name}
            """).fetchone()

            stats['type'] = 'single'
            stats['num_pairs'] = result[0]
            stats['min_ccc'] = result[1]
            stats['max_ccc'] = result[2]
            stats['mean_ccc'] = result[3]

        # Database file size
        stats['database_size_gb'] = self.db_path.stat().st_size / (1024**3)

        return stats

    def close(self):
        """Close database connection."""
        self.con.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def main():
    """Example usage and simple CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Query CCC correlation database")
    parser.add_argument("database", help="Path to DuckDB database")
    parser.add_argument("--tissue", help="Tissue name (for consolidated database)")
    parser.add_argument("--gene1", help="First gene ID")
    parser.add_argument("--gene2", help="Second gene ID")
    parser.add_argument("--gene", help="Get all correlations for this gene")
    parser.add_argument("--top", type=float, help="Get top correlations above threshold")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of results")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    args = parser.parse_args()

    # Initialize database
    with CCCDatabase(args.database, tissue=args.tissue) as db:

        if args.stats:
            stats = db.get_statistics()
            print("\nDatabase Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        elif args.gene1 and args.gene2:
            # Query specific pair
            ccc = db.get_correlation(args.gene1, args.gene2)
            if ccc is not None:
                print(f"CCC({args.gene1}, {args.gene2}) = {ccc:.6f}")
            else:
                print(f"No correlation found for pair ({args.gene1}, {args.gene2})")

        elif args.gene:
            # Get all correlations for gene
            results = db.get_gene_correlations(args.gene, limit=args.limit)
            print(f"\nTop {len(results)} correlations for {args.gene}:")
            print(results.to_string())

        elif args.top:
            # Get top correlations
            results = db.get_top_correlations(threshold=args.top, limit=args.limit)
            print(f"\nTop {len(results)} correlations above {args.top}:")
            print(results.to_string())

        else:
            print("Please specify a query option (--gene1/--gene2, --gene, --top, or --stats)")


if __name__ == "__main__":
    main()