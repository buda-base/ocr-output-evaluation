"""
Utility script to explore and query the computed statistics using DuckDB
"""
import duckdb
import argparse
from pathlib import Path


def connect_to_stats(output_dir: str = 'output'):
    """Create a DuckDB connection with the stats files loaded"""
    con = duckdb.connect(':memory:')
    
    gb_path = Path(output_dir) / 'google_books_stats.parquet'
    gv_path = Path(output_dir) / 'google_vision_stats.parquet'
    
    if gb_path.exists():
        con.execute(f"CREATE VIEW google_books AS SELECT * FROM '{gb_path}'")
        print(f"✓ Loaded Google Books stats: {gb_path}")
    
    if gv_path.exists():
        con.execute(f"CREATE VIEW google_vision AS SELECT * FROM '{gv_path}'")
        print(f"✓ Loaded Google Vision stats: {gv_path}")
    
    return con


def print_summary(con: duckdb.DuckDBPyConnection):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Google Books summary
    try:
        result = con.execute("""
            SELECT 
                COUNT(*) as num_volumes,
                SUM(total_pages) as total_pages,
                AVG(mean_confidence) as avg_mean_confidence,
                AVG(median_confidence) as avg_median_confidence,
                AVG(pct_high_conf) as avg_pct_high_conf,
                AVG(pct_medium_conf) as avg_pct_medium_conf,
                AVG(pct_low_conf) as avg_pct_low_conf
            FROM google_books
        """).fetchone()
        
        print("\nGoogle Books:")
        print(f"  Volumes: {result[0]:,}")
        print(f"  Total pages: {result[1]:,}")
        print(f"  Avg mean confidence: {result[2]:.3f}")
        print(f"  Avg median confidence: {result[3]:.3f}")
        print(f"  Avg % high confidence (>=90%): {result[4]:.1f}%")
        print(f"  Avg % medium confidence (70-90%): {result[5]:.1f}%")
        print(f"  Avg % low confidence (<70%): {result[6]:.1f}%")
    except:
        print("\nGoogle Books: No data available")
    
    # Google Vision summary
    try:
        result = con.execute("""
            SELECT 
                COUNT(*) as num_volumes,
                SUM(total_records) as total_pages,
                AVG(mean_confidence) as avg_mean_confidence,
                AVG(median_confidence) as avg_median_confidence,
                AVG(pct_high_conf) as avg_pct_high_conf,
                AVG(pct_medium_conf) as avg_pct_medium_conf,
                AVG(pct_low_conf) as avg_pct_low_conf
            FROM google_vision
        """).fetchone()
        
        print("\nGoogle Vision:")
        print(f"  Volumes: {result[0]:,}")
        print(f"  Total pages: {result[1]:,}")
        print(f"  Avg mean confidence: {result[2]:.3f}")
        print(f"  Avg median confidence: {result[3]:.3f}")
        print(f"  Avg % high confidence (>=90%): {result[4]:.1f}%")
        print(f"  Avg % medium confidence (70-90%): {result[5]:.1f}%")
        print(f"  Avg % low confidence (<70%): {result[6]:.1f}%")
    except:
        print("\nGoogle Vision: No data available")


def print_low_confidence_volumes(con: duckdb.DuckDBPyConnection, limit: int = 20):
    """Print volumes with lowest confidence"""
    print("\n" + "="*80)
    print(f"TOP {limit} VOLUMES WITH LOWEST CONFIDENCE")
    print("="*80)
    
    # Google Books
    try:
        result = con.execute(f"""
            SELECT w_id, i_id, i_version, mean_confidence, median_confidence, 
                   pct_low_conf, total_pages
            FROM google_books
            ORDER BY mean_confidence ASC
            LIMIT {limit}
        """).fetchdf()
        
        print("\nGoogle Books (lowest mean confidence):")
        print(result.to_string(index=False))
    except:
        print("\nGoogle Books: No data available")
    
    # Google Vision
    try:
        result = con.execute(f"""
            SELECT w_id, i_id, i_version, mean_confidence, median_confidence, 
                   pct_low_conf, total_records as total_pages
            FROM google_vision
            ORDER BY mean_confidence ASC
            LIMIT {limit}
        """).fetchdf()
        
        print("\nGoogle Vision (lowest mean confidence):")
        print(result.to_string(index=False))
    except:
        print("\nGoogle Vision: No data available")


def run_custom_query(con: duckdb.DuckDBPyConnection, query: str):
    """Run a custom SQL query"""
    print("\n" + "="*80)
    print("CUSTOM QUERY RESULTS")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    try:
        result = con.execute(query).fetchdf()
        print(result.to_string(index=False))
    except Exception as e:
        print(f"Error executing query: {e}")


def interactive_mode(con: duckdb.DuckDBPyConnection):
    """Interactive SQL query mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("\nAvailable tables: google_books, google_vision")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'schema' to see table schemas")
    print()
    
    while True:
        try:
            query = input("SQL> ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if query.lower() == 'schema':
                print("\nGoogle Books schema:")
                print(con.execute("DESCRIBE google_books").fetchdf().to_string(index=False))
                print("\nGoogle Vision schema:")
                print(con.execute("DESCRIBE google_vision").fetchdf().to_string(index=False))
                continue
            
            if not query:
                continue
            
            result = con.execute(query).fetchdf()
            print(result.to_string(index=False))
            print()
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Explore OCR confidence statistics')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing stats files')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--low-confidence', type=int, metavar='N',
                       help='Show top N volumes with lowest confidence')
    parser.add_argument('--query', type=str,
                       help='Run a custom SQL query')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive SQL mode')
    
    args = parser.parse_args()
    
    # Connect to stats
    con = connect_to_stats(args.output_dir)
    
    # If no arguments, show summary by default
    if not any([args.summary, args.low_confidence, args.query, args.interactive]):
        args.summary = True
    
    if args.summary:
        print_summary(con)
    
    if args.low_confidence:
        print_low_confidence_volumes(con, args.low_confidence)
    
    if args.query:
        run_custom_query(con, args.query)
    
    if args.interactive:
        interactive_mode(con)


if __name__ == '__main__':
    main()
