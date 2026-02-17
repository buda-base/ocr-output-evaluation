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
    ocrv1_path = Path(output_dir) / 'ocrv1_ws_ldv1_stats.parquet'
    
    has_perplexity = False
    
    if gb_path.exists():
        con.execute(f"CREATE VIEW google_books AS SELECT * FROM '{gb_path}'")
        print(f"✓ Loaded Google Books stats: {gb_path}")
        # Check if perplexity data is available
        try:
            result = con.execute("SELECT mean_perplexity FROM google_books LIMIT 1").fetchone()
            if result is not None:
                has_perplexity = True
        except:
            pass
    
    if gv_path.exists():
        con.execute(f"CREATE VIEW google_vision AS SELECT * FROM '{gv_path}'")
        print(f"✓ Loaded Google Vision stats: {gv_path}")
        if not has_perplexity:
            try:
                result = con.execute("SELECT mean_perplexity FROM google_vision LIMIT 1").fetchone()
                if result is not None:
                    has_perplexity = True
            except:
                pass
    
    if ocrv1_path.exists():
        con.execute(f"CREATE VIEW ocrv1_ws_ldv1 AS SELECT * FROM '{ocrv1_path}'")
        print(f"✓ Loaded OCRv1-WS-LDv1 stats: {ocrv1_path}")
        if not has_perplexity:
            try:
                result = con.execute("SELECT mean_perplexity FROM ocrv1_ws_ldv1 LIMIT 1").fetchone()
                if result is not None:
                    has_perplexity = True
            except:
                pass
    
    if has_perplexity:
        print(f"  ✓ Perplexity data available")
    else:
        print(f"  ℹ No perplexity data (disabled during analysis)")
    
    return con


def print_summary(con: duckdb.DuckDBPyConnection):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Google Books summary
    try:
        # Check if perplexity columns exist
        has_perplexity = False
        try:
            con.execute("SELECT mean_perplexity FROM google_books LIMIT 1")
            has_perplexity = True
        except:
            pass
        
        if has_perplexity:
            # Check for detailed perplexity columns
            has_detailed_ppl = False
            try:
                con.execute("SELECT pages_no_tibetan_text FROM google_books LIMIT 1")
                has_detailed_ppl = True
            except:
                pass

            if has_detailed_ppl:
                result = con.execute("""
                    SELECT 
                        COUNT(*) as num_volumes,
                        SUM(total_pages) as total_pages,
                        AVG(mean_confidence) as avg_mean_confidence,
                        AVG(median_confidence) as avg_median_confidence,
                        AVG(pct_high_conf) as avg_pct_high_conf,
                        AVG(pct_medium_conf) as avg_pct_medium_conf,
                        AVG(pct_low_conf) as avg_pct_low_conf,
                        AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END) as avg_mean_perplexity,
                        AVG(CASE WHEN median_perplexity != 'inf' THEN median_perplexity ELSE NULL END) as avg_median_perplexity,
                        SUM(CASE WHEN mean_perplexity = 'inf' THEN 1 ELSE 0 END) as inf_perplexity_count,
                        SUM(pages_no_tibetan_text) as total_no_tibetan,
                        SUM(pages_model_rejection) as total_model_rejection
                    FROM google_books
                """).fetchone()
            else:
                result = con.execute("""
                    SELECT 
                        COUNT(*) as num_volumes,
                        SUM(total_pages) as total_pages,
                        AVG(mean_confidence) as avg_mean_confidence,
                        AVG(median_confidence) as avg_median_confidence,
                        AVG(pct_high_conf) as avg_pct_high_conf,
                        AVG(pct_medium_conf) as avg_pct_medium_conf,
                        AVG(pct_low_conf) as avg_pct_low_conf,
                        AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END) as avg_mean_perplexity,
                        AVG(CASE WHEN median_perplexity != 'inf' THEN median_perplexity ELSE NULL END) as avg_median_perplexity,
                        SUM(CASE WHEN mean_perplexity = 'inf' THEN 1 ELSE 0 END) as inf_perplexity_count
                    FROM google_books
                """).fetchone()
        else:
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
        
        if has_perplexity:
            print(f"  Avg mean perplexity: {result[7]:.2f}")
            print(f"  Avg median perplexity: {result[8]:.2f}")
            if result[9] > 0:
                print(f"  Volumes with infinite perplexity: {result[9]:,}")
            
            if has_detailed_ppl:
                print(f"  Total pages with no Tibetan text: {result[10]:,}")
                print(f"  Total pages with model rejection: {result[11]:,}")
    except Exception as e:
        print(f"\nGoogle Books: No data available ({e})")
    
    # Google Vision summary
    try:
        # Check if perplexity columns exist
        has_perplexity = False
        try:
            con.execute("SELECT mean_perplexity FROM google_vision LIMIT 1")
            has_perplexity = True
        except:
            pass
        
        if has_perplexity:
            # Check for detailed perplexity columns
            has_detailed_ppl = False
            try:
                con.execute("SELECT pages_no_tibetan_text FROM google_vision LIMIT 1")
                has_detailed_ppl = True
            except:
                pass

            if has_detailed_ppl:
                result = con.execute("""
                    SELECT 
                        COUNT(*) as num_volumes,
                        SUM(total_records) as total_pages,
                        AVG(mean_confidence) as avg_mean_confidence,
                        AVG(median_confidence) as avg_median_confidence,
                        AVG(pct_high_conf) as avg_pct_high_conf,
                        AVG(pct_medium_conf) as avg_pct_medium_conf,
                        AVG(pct_low_conf) as avg_pct_low_conf,
                        AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END) as avg_mean_perplexity,
                        AVG(CASE WHEN median_perplexity != 'inf' THEN median_perplexity ELSE NULL END) as avg_median_perplexity,
                        SUM(CASE WHEN mean_perplexity = 'inf' THEN 1 ELSE 0 END) as inf_perplexity_count,
                        SUM(pages_no_tibetan_text) as total_no_tibetan,
                        SUM(pages_model_rejection) as total_model_rejection
                    FROM google_vision
                """).fetchone()
            else:
                result = con.execute("""
                    SELECT 
                        COUNT(*) as num_volumes,
                        SUM(total_records) as total_pages,
                        AVG(mean_confidence) as avg_mean_confidence,
                        AVG(median_confidence) as avg_median_confidence,
                        AVG(pct_high_conf) as avg_pct_high_conf,
                        AVG(pct_medium_conf) as avg_pct_medium_conf,
                        AVG(pct_low_conf) as avg_pct_low_conf,
                        AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END) as avg_mean_perplexity,
                        AVG(CASE WHEN median_perplexity != 'inf' THEN median_perplexity ELSE NULL END) as avg_median_perplexity,
                        SUM(CASE WHEN mean_perplexity = 'inf' THEN 1 ELSE 0 END) as inf_perplexity_count
                    FROM google_vision
                """).fetchone()
        else:
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
        
        if has_perplexity:
            print(f"  Avg mean perplexity: {result[7]:.2f}")
            print(f"  Avg median perplexity: {result[8]:.2f}")
            if result[9] > 0:
                print(f"  Volumes with infinite perplexity: {result[9]:,}")
            
            if has_detailed_ppl:
                print(f"  Total pages with no Tibetan text: {result[10]:,}")
                print(f"  Total pages with model rejection: {result[11]:,}")
    except Exception as e:
        print(f"\nGoogle Vision: No data available ({e})")


def print_low_confidence_volumes(con: duckdb.DuckDBPyConnection, limit: int = 20):
    """Print volumes with lowest confidence"""
    print("\n" + "="*80)
    print(f"TOP {limit} VOLUMES WITH LOWEST CONFIDENCE")
    print("="*80)
    
    # Google Books
    try:
        # Check if perplexity columns exist
        has_perplexity = False
        try:
            con.execute("SELECT mean_perplexity FROM google_books LIMIT 1")
            has_perplexity = True
        except:
            pass
        
        if has_perplexity:
            result = con.execute(f"""
                SELECT w_id, i_id, i_version, mean_confidence, median_confidence, 
                       mean_perplexity, pct_low_conf, total_pages
                FROM google_books
                ORDER BY mean_confidence ASC
                LIMIT {limit}
            """).fetchdf()
        else:
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
        # Check if perplexity columns exist
        has_perplexity = False
        try:
            con.execute("SELECT mean_perplexity FROM google_vision LIMIT 1")
            has_perplexity = True
        except:
            pass
        
        if has_perplexity:
            result = con.execute(f"""
                SELECT w_id, i_id, i_version, mean_confidence, median_confidence, 
                       mean_perplexity, pct_low_conf, total_records as total_pages
                FROM google_vision
                ORDER BY mean_confidence ASC
                LIMIT {limit}
            """).fetchdf()
        else:
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


def print_high_perplexity_volumes(con: duckdb.DuckDBPyConnection, limit: int = 20):
    """Print volumes with highest perplexity (poorest quality text)"""
    print("\n" + "="*80)
    print(f"TOP {limit} VOLUMES WITH HIGHEST PERPLEXITY (POOREST TEXT QUALITY)")
    print("="*80)
    
    # Google Books
    try:
        result = con.execute(f"""
            SELECT w_id, i_id, i_version, mean_confidence, mean_perplexity, 
                   median_perplexity, total_pages
            FROM google_books
            WHERE mean_perplexity != 'inf'
            ORDER BY mean_perplexity DESC
            LIMIT {limit}
        """).fetchdf()
        
        print("\nGoogle Books (highest perplexity):")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"\nGoogle Books: No perplexity data available")
    
    # Google Vision
    try:
        result = con.execute(f"""
            SELECT w_id, i_id, i_version, mean_confidence, mean_perplexity, 
                   median_perplexity, total_records as total_pages
            FROM google_vision
            WHERE mean_perplexity != 'inf'
            ORDER BY mean_perplexity DESC
            LIMIT {limit}
        """).fetchdf()
        
        print("\nGoogle Vision (highest perplexity):")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"\nGoogle Vision: No perplexity data available")


def print_quality_matrix(con: duckdb.DuckDBPyConnection):
    """Print quality matrix showing confidence vs perplexity distribution"""
    print("\n" + "="*80)
    print("QUALITY MATRIX (Confidence vs Perplexity - Percentile Based)")
    print("="*80)
    print("\nNote: Perplexity thresholds are dataset-relative (P33 = best third, P66 = worst third)")
    
    # Google Books
    try:
        # First get percentiles
        percentiles = con.execute("""
            SELECT 
                PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY mean_perplexity) as p33,
                PERCENTILE_CONT(0.66) WITHIN GROUP (ORDER BY mean_perplexity) as p66
            FROM google_books
            WHERE mean_perplexity != 'inf'
        """).fetchone()
        
        p33, p66 = percentiles[0], percentiles[1]
        
        result = con.execute(f"""
            SELECT 
                CASE 
                    WHEN mean_confidence >= 0.9 THEN 'High Conf (≥0.9)'
                    WHEN mean_confidence >= 0.7 THEN 'Med Conf (0.7-0.9)'
                    ELSE 'Low Conf (<0.7)'
                END as confidence_category,
                CASE 
                    WHEN mean_perplexity = 'inf' THEN 'Invalid (inf)'
                    WHEN mean_perplexity > {p66} THEN 'Poor (>P66={p66:.0f})'
                    WHEN mean_perplexity > {p33} THEN 'Medium (P33-P66)'
                    ELSE 'Good (<P33={p33:.0f})'
                END as perplexity_category,
                COUNT(*) as volume_count,
                ROUND(AVG(mean_confidence), 3) as avg_confidence,
                ROUND(AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END), 2) as avg_perplexity
            FROM google_books
            GROUP BY confidence_category, perplexity_category
            ORDER BY confidence_category DESC, perplexity_category
        """).fetchdf()
        
        print(f"\nGoogle Books (P33={p33:.0f}, P66={p66:.0f}):")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"\nGoogle Books: No perplexity data available")
    
    # Google Vision
    try:
        # First get percentiles
        percentiles = con.execute("""
            SELECT 
                PERCENTILE_CONT(0.33) WITHIN GROUP (ORDER BY mean_perplexity) as p33,
                PERCENTILE_CONT(0.66) WITHIN GROUP (ORDER BY mean_perplexity) as p66
            FROM google_vision
            WHERE mean_perplexity != 'inf'
        """).fetchone()
        
        p33, p66 = percentiles[0], percentiles[1]
        
        result = con.execute(f"""
            SELECT 
                CASE 
                    WHEN mean_confidence >= 0.9 THEN 'High Conf (≥0.9)'
                    WHEN mean_confidence >= 0.7 THEN 'Med Conf (0.7-0.9)'
                    ELSE 'Low Conf (<0.7)'
                END as confidence_category,
                CASE 
                    WHEN mean_perplexity = 'inf' THEN 'Invalid (inf)'
                    WHEN mean_perplexity > {p66} THEN 'Poor (>P66={p66:.0f})'
                    WHEN mean_perplexity > {p33} THEN 'Medium (P33-P66)'
                    ELSE 'Good (<P33={p33:.0f})'
                END as perplexity_category,
                COUNT(*) as volume_count,
                ROUND(AVG(mean_confidence), 3) as avg_confidence,
                ROUND(AVG(CASE WHEN mean_perplexity != 'inf' THEN mean_perplexity ELSE NULL END), 2) as avg_perplexity
            FROM google_vision
            GROUP BY confidence_category, perplexity_category
            ORDER BY confidence_category DESC, perplexity_category
        """).fetchdf()
        
        print(f"\nGoogle Vision (P33={p33:.0f}, P66={p66:.0f}):")
        print(result.to_string(index=False))
    except Exception as e:
        print(f"\nGoogle Vision: No perplexity data available")


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
    print("\nAvailable tables: google_books, google_vision, ocrv1_ws_ldv1")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'schema' to see table schemas")
    print("Type 'examples' to see example queries")
    print()
    
    while True:
        try:
            query = input("SQL> ").strip()
            
            if query.lower() in ['exit', 'quit']:
                break
            
            if query.lower() == 'schema':
                try:
                    print("\nGoogle Books schema:")
                    print(con.execute("DESCRIBE google_books").fetchdf().to_string(index=False))
                except:
                    print("  (not loaded)")
                try:
                    print("\nGoogle Vision schema:")
                    print(con.execute("DESCRIBE google_vision").fetchdf().to_string(index=False))
                except:
                    print("  (not loaded)")
                try:
                    print("\nOCRv1-WS-LDv1 schema:")
                    print(con.execute("DESCRIBE ocrv1_ws_ldv1").fetchdf().to_string(index=False))
                except:
                    print("  (not loaded)")
                continue
            
            if query.lower() == 'examples':
                print("\nExample queries:")
                print("\n1. Find best quality volumes (OCRv1):")
                print("   SELECT w_id, i_id, mean_perplexity")
                print("   FROM ocrv1_ws_ldv1")
                print("   ORDER BY mean_perplexity ASC LIMIT 20;")
                print("\n2. Compare OCR types:")
                print("   SELECT 'Google Books' as ocr_type, AVG(mean_perplexity) as avg_ppl FROM google_books")
                print("   UNION ALL")
                print("   SELECT 'Google Vision', AVG(mean_perplexity) FROM google_vision")
                print("   UNION ALL")
                print("   SELECT 'OCRv1-WS-LDv1', AVG(mean_perplexity) FROM ocrv1_ws_ldv1;")
                print("\n3. Find best OCRv1 volumes:")
                print("   SELECT w_id, i_id, mean_perplexity, total_records")
                print("   FROM ocrv1_ws_ldv1")
                print("   WHERE mean_perplexity != 'inf'")
                print("   ORDER BY mean_perplexity ASC LIMIT 50;")
                print()
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
    parser = argparse.ArgumentParser(description='Explore OCR confidence and perplexity statistics')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory containing stats files')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--low-confidence', type=int, metavar='N',
                       help='Show top N volumes with lowest confidence')
    parser.add_argument('--high-perplexity', type=int, metavar='N',
                       help='Show top N volumes with highest perplexity (poorest quality)')
    parser.add_argument('--quality-matrix', action='store_true',
                       help='Show quality matrix (confidence vs perplexity distribution)')
    parser.add_argument('--query', type=str,
                       help='Run a custom SQL query')
    parser.add_argument('--interactive', action='store_true',
                       help='Enter interactive SQL mode')
    
    args = parser.parse_args()
    
    # Connect to stats
    con = connect_to_stats(args.output_dir)
    
    # If no arguments, show summary by default
    if not any([args.summary, args.low_confidence, args.high_perplexity, 
                args.quality_matrix, args.query, args.interactive]):
        args.summary = True
    
    if args.summary:
        print_summary(con)
    
    if args.low_confidence:
        print_low_confidence_volumes(con, args.low_confidence)
    
    if args.high_perplexity:
        print_high_perplexity_volumes(con, args.high_perplexity)
    
    if args.quality_matrix:
        print_quality_matrix(con)
    
    if args.query:
        run_custom_query(con, args.query)
    
    if args.interactive:
        interactive_mode(con)


if __name__ == '__main__':
    main()
