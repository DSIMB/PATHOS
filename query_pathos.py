#!/usr/bin/env python3
"""
PATHOS Database Query Script

Query the PATHOS SQLite database for protein variant pathogenicity predictions.
Supports multiple query modes: single protein, batch file, specific mutations.
"""

import argparse
import sqlite3
import sys
import os
import csv
from typing import List, Tuple, Optional

# Default database path (can be overridden with --database flag)
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "database", "pathos.db")


class PathosDB:
    """Interface for querying PATHOS database"""
    
    def __init__(self, db_path: str):
        """Initialize database connection"""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_protein_variants(self, protein_id: str, min_score: Optional[float] = None, 
                           max_score: Optional[float] = None) -> List[Tuple]:
        """
        Get all variants for a protein
        
        Args:
            protein_id: UniProt ID
            min_score: Minimum pathogenicity score (optional)
            max_score: Maximum pathogenicity score (optional)
        
        Returns:
            List of (mutation, score) tuples
        """
        query = "SELECT mutation, score FROM mutations WHERE protein_id = ?"
        params = [protein_id]
        
        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)
        
        if max_score is not None:
            query += " AND score <= ?"
            params.append(max_score)
        
        query += " ORDER BY score DESC"
        
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def get_specific_mutation(self, protein_id: str, mutation: str) -> Optional[float]:
        """
        Get score for a specific mutation
        
        Args:
            protein_id: UniProt ID
            mutation: Mutation notation (e.g., "M1A")
        
        Returns:
            Pathogenicity score or None if not found
        """
        self.cursor.execute(
            "SELECT score FROM mutations WHERE protein_id = ? AND mutation = ?",
            (protein_id, mutation)
        )
        result = self.cursor.fetchone()
        return result[0] if result else None
    
    def get_multiple_mutations(self, variants: List[Tuple[str, str]]) -> List[Tuple]:
        """
        Get scores for multiple protein-mutation pairs
        
        Args:
            variants: List of (protein_id, mutation) tuples
        
        Returns:
            List of (protein_id, mutation, score) tuples
        """
        results = []
        for protein_id, mutation in variants:
            score = self.get_specific_mutation(protein_id, mutation)
            results.append((protein_id, mutation, score))
        return results
    
    def get_all_proteins(self) -> List[str]:
        """Get list of all proteins in database"""
        self.cursor.execute("SELECT DISTINCT protein_id FROM mutations ORDER BY protein_id")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        stats = {}
        
        # Total proteins
        self.cursor.execute("SELECT COUNT(DISTINCT protein_id) FROM mutations")
        stats['total_proteins'] = self.cursor.fetchone()[0]
        
        # Total mutations
        self.cursor.execute("SELECT COUNT(*) FROM mutations")
        stats['total_mutations'] = self.cursor.fetchone()[0]
        
        # Average mutations per protein
        stats['avg_mutations_per_protein'] = stats['total_mutations'] / stats['total_proteins']
        
        # Score distribution
        self.cursor.execute("SELECT AVG(score), MIN(score), MAX(score) FROM mutations")
        avg, min_score, max_score = self.cursor.fetchone()
        stats['avg_score'] = avg
        stats['min_score'] = min_score
        stats['max_score'] = max_score
        
        return stats


def parse_input_file(file_path: str) -> List[Tuple[str, Optional[str]]]:
    """
    Parse input file with protein IDs and optional mutations
    
    Format:
        P16501              # All mutations for this protein
        Q9Y6X3 M1A          # Specific mutation
        P10635 R56V W695F   # Multiple mutations
    
    Returns:
        List of (protein_id, mutation) tuples (mutation is None for all-mutations query)
    """
    variants = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            protein_id = parts[0]
            
            if len(parts) == 1:
                # Query all mutations for this protein
                variants.append((protein_id, None))
            else:
                # Query specific mutations
                for mutation in parts[1:]:
                    variants.append((protein_id, mutation))
    
    return variants


def format_score(score: Optional[float]) -> str:
    """Format score with interpretation"""
    if score is None:
        return "NOT FOUND"
    
    if score < 0.63:
        interpretation = "BENIGN"
    else:
        interpretation = "PATHOGENIC"
    
    return f"{score:.4f} ({interpretation})"


def print_results_table(results: List[Tuple], headers: List[str]):
    """Print results in a formatted table"""
    if not results:
        print("No results found.")
        return
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in results:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Print header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in results:
        print("  ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def export_to_csv(results: List[Tuple], headers: List[str], output_file: str):
    """Export results to CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Query PATHOS database for protein variant predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query all mutations for a protein
  %(prog)s --protein P16501
  
  # Query specific mutation
  %(prog)s --protein P16501 --mutation M1A
  
  # Query from input file
  %(prog)s --file variants.txt
  
  # Filter by pathogenicity score
  %(prog)s --protein P16501 --min-score 0.9
  
  # Export results to CSV
  %(prog)s --protein P16501 --output results.csv
  
  # Show database statistics
  %(prog)s --stats
  
  # List all proteins
  %(prog)s --list-proteins

Input file format (one per line):
  P16501              # All mutations for this protein
  Q9Y6X3 M1A          # Specific mutation
  P10635 R56V W695F   # Multiple mutations
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--protein', '-p', help='UniProt ID to query')
    input_group.add_argument('--file', '-f', help='Input file with protein IDs and mutations')
    input_group.add_argument('--stats', action='store_true', help='Show database statistics')
    input_group.add_argument('--list-proteins', action='store_true', help='List all proteins in database')
    
    # Query options
    parser.add_argument('--mutation', '-m', help='Specific mutation to query (e.g., M1A)')
    parser.add_argument('--min-score', type=float, help='Minimum pathogenicity score')
    parser.add_argument('--max-score', type=float, help='Maximum pathogenicity score')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--database', '-d', default=DEFAULT_DB_PATH, 
                       help=f'Path to PATHOS database (default: {DEFAULT_DB_PATH})')
    parser.add_argument('--limit', '-l', type=int, help='Limit number of results')
    
    args = parser.parse_args()
    
    # Open database
    try:
        with PathosDB(args.database) as db:
            
            # Handle stats query
            if args.stats:
                stats = db.get_database_stats()
                print("\n=== PATHOS Database Statistics ===")
                print(f"Total proteins:              {stats['total_proteins']:,}")
                print(f"Total mutations:             {stats['total_mutations']:,}")
                print(f"Avg mutations per protein:   {stats['avg_mutations_per_protein']:.1f}")
                print(f"Score range:                 {stats['min_score']:.4f} - {stats['max_score']:.4f}")
                print(f"Average score:               {stats['avg_score']:.4f}")
                return
            
            # Handle list proteins query
            if args.list_proteins:
                proteins = db.get_all_proteins()
                print(f"\n=== {len(proteins)} Proteins in Database ===")
                for i, protein in enumerate(proteins, 1):
                    print(protein, end='  ')
                    if i % 10 == 0:
                        print()  # New line every 10 proteins
                print()
                return
            
            # Handle single protein query
            if args.protein:
                protein_id = args.protein
                
                if args.mutation:
                    # Query specific mutation
                    score = db.get_specific_mutation(protein_id, args.mutation)
                    print(f"\n{protein_id} {args.mutation}: {format_score(score)}")
                    
                else:
                    # Query all mutations for protein
                    results = db.get_protein_variants(protein_id, args.min_score, args.max_score)
                    
                    if args.limit:
                        results = results[:args.limit]
                    
                    if not results:
                        print(f"No mutations found for protein {protein_id}")
                        return
                    
                    print(f"\n=== {len(results)} mutations for {protein_id} ===")
                    
                    # Format results with interpretation
                    formatted_results = [
                        (mut, f"{score:.4f}", format_score(score).split('(')[1].rstrip(')'))
                        for mut, score in results
                    ]
                    
                    if args.output:
                        export_to_csv(formatted_results, 
                                    ['Mutation', 'Score', 'Interpretation'], 
                                    args.output)
                    else:
                        print_results_table(formatted_results, 
                                          ['Mutation', 'Score', 'Interpretation'])
            
            # Handle file input query
            elif args.file:
                if not os.path.exists(args.file):
                    print(f"Error: Input file not found: {args.file}", file=sys.stderr)
                    sys.exit(1)
                
                variants = parse_input_file(args.file)
                print(f"\n=== Processing {len(variants)} queries from {args.file} ===\n")
                
                results = []
                for protein_id, mutation in variants:
                    if mutation is None:
                        # Get all mutations for this protein
                        protein_results = db.get_protein_variants(protein_id, args.min_score, args.max_score)
                        
                        if args.limit:
                            protein_results = protein_results[:args.limit]
                        
                        for mut, score in protein_results:
                            results.append((protein_id, mut, f"{score:.4f}", 
                                          format_score(score).split('(')[1].rstrip(')')))
                    else:
                        # Get specific mutation
                        score = db.get_specific_mutation(protein_id, mutation)
                        results.append((protein_id, mutation, 
                                      f"{score:.4f}" if score else "N/A",
                                      format_score(score).split('(')[1].rstrip(')') if score else "NOT FOUND"))
                
                if not results:
                    print("No results found.")
                    return
                
                if args.output:
                    export_to_csv(results, 
                                ['Protein', 'Mutation', 'Score', 'Interpretation'], 
                                args.output)
                else:
                    print_results_table(results, 
                                      ['Protein', 'Mutation', 'Score', 'Interpretation'])
                    print(f"\nTotal results: {len(results)}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nPlease ensure the database file exists at: {args.database}", file=sys.stderr)
        print("You can specify a different path with --database", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
