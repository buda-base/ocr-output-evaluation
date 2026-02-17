#!/usr/bin/env python3
"""
Analyze OCR quality by system, print method, and script
"""
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path

from metadata_loader import load_all_stats_with_metadata, print_metadata_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_by_system(df: pd.DataFrame):
    """Analyze perplexity and confidence by OCR system"""
    print("\n" + "="*80)
    print("ANALYSIS BY OCR SYSTEM")
    print("="*80)
    
    if 'ocr_system' not in df.columns:
        print("No OCR system data available")
        return
    
    # Group by system
    for system in ['google_ocr', 'ocrv1-ws-ldv1']:
        system_df = df[df['ocr_system'] == system]
        if len(system_df) == 0:
            continue
        
        print(f"\n{system.upper()}:")
        print(f"  Volumes: {len(system_df):,}")
        
        # Confidence stats (if available)
        if 'mean_confidence' in system_df.columns:
            conf_valid = system_df['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(conf_valid) > 0:
                print(f"  Avg mean confidence: {conf_valid.mean():.3f}")
                print(f"  Median confidence: {conf_valid.median():.3f}")
        
        # Perplexity stats
        if 'mean_perplexity' in system_df.columns:
            perp_valid = system_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(perp_valid) > 0:
                print(f"  Avg mean perplexity: {perp_valid.mean():.2f}")
                print(f"  Median perplexity: {perp_valid.median():.2f}")
                print(f"  P33 (best third): {perp_valid.quantile(0.33):.2f}")
                print(f"  P66 (worst third): {perp_valid.quantile(0.66):.2f}")
                print(f"  Volumes with valid perplexity: {len(perp_valid):,} ({len(perp_valid)/len(system_df)*100:.1f}%)")
            
            # New detailed metrics
            if 'pages_no_tibetan_text' in system_df.columns:
                total_pages = system_df['total_records'].sum() if 'total_records' in system_df.columns else 0
                no_tibetan = system_df['pages_no_tibetan_text'].sum()
                model_rejection = system_df['pages_model_rejection'].sum()
                
                print(f"\n  Perplexity Issues Breakdown:")
                if total_pages > 0:
                    print(f"    Pages with no Tibetan text: {no_tibetan:,} ({no_tibetan/total_pages*100:.1f}%)")
                    print(f"    Pages with model rejection: {model_rejection:,} ({model_rejection/total_pages*100:.1f}%)")
                else:
                    print(f"    Pages with no Tibetan text: {no_tibetan:,}")
                    print(f"    Pages with model rejection: {model_rejection:,}")
                
                # Check for lines but no text (OCR failure)
                if 'pages_with_lines' in system_df.columns and 'pages_with_text' in system_df.columns:
                    lines_but_no_text = (system_df['pages_with_lines'] - system_df['pages_with_text']).clip(lower=0).sum()
                    print(f"    Pages with lines but no text: {lines_but_no_text:,}")


def analyze_by_print_method(df: pd.DataFrame):
    """Analyze perplexity by print method"""
    print("\n" + "="*80)
    print("ANALYSIS BY PRINT METHOD")
    print("="*80)
    
    if 'print_method' not in df.columns:
        print("No print method data available")
        return
    
    # Filter to rows with print method
    df_with_pm = df[df['print_method'].notna()].copy()
    
    if len(df_with_pm) == 0:
        print("No volumes with print method metadata")
        return
    
    print(f"\nAnalyzing {len(df_with_pm):,} volumes with print method metadata")
    
    # Group by print method
    pm_groups = df_with_pm.groupby('print_method')
    
    results = []
    for pm, group in pm_groups:
        if 'mean_perplexity' in group.columns:
            perp_valid = group['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            
            result = {
                'print_method': pm,
                'volume_count': len(group),
                'avg_perplexity': perp_valid.mean() if len(perp_valid) > 0 else np.nan,
                'median_perplexity': perp_valid.median() if len(perp_valid) > 0 else np.nan,
                'p33_perplexity': perp_valid.quantile(0.33) if len(perp_valid) > 0 else np.nan,
                'p66_perplexity': perp_valid.quantile(0.66) if len(perp_valid) > 0 else np.nan,
            }
            
            # Add confidence if available
            if 'mean_confidence' in group.columns:
                conf_valid = group['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
                result['avg_confidence'] = conf_valid.mean() if len(conf_valid) > 0 else np.nan
            
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('avg_perplexity')
        print("\n" + results_df.to_string(index=False))


def analyze_by_script(df: pd.DataFrame):
    """Analyze perplexity by script"""
    print("\n" + "="*80)
    print("ANALYSIS BY SCRIPT")
    print("="*80)
    
    if 'script' not in df.columns:
        print("No script data available")
        return
    
    # Filter to rows with script
    df_with_script = df[df['script'].notna()].copy()
    
    if len(df_with_script) == 0:
        print("No volumes with script metadata")
        return
    
    print(f"\nAnalyzing {len(df_with_script):,} volumes with script metadata")
    
    # Group by script
    script_groups = df_with_script.groupby('script')
    
    results = []
    for script, group in script_groups:
        if 'mean_perplexity' in group.columns:
            perp_valid = group['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            
            result = {
                'script': script,
                'volume_count': len(group),
                'avg_perplexity': perp_valid.mean() if len(perp_valid) > 0 else np.nan,
                'median_perplexity': perp_valid.median() if len(perp_valid) > 0 else np.nan,
                'p33_perplexity': perp_valid.quantile(0.33) if len(perp_valid) > 0 else np.nan,
                'p66_perplexity': perp_valid.quantile(0.66) if len(perp_valid) > 0 else np.nan,
            }
            
            # Add confidence if available
            if 'mean_confidence' in group.columns:
                conf_valid = group['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
                result['avg_confidence'] = conf_valid.mean() if len(conf_valid) > 0 else np.nan
            
            results.append(result)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('avg_perplexity')
        print("\n" + results_df.to_string(index=False))


def analyze_cross_dimensions(df: pd.DataFrame):
    """Analyze by system × print method and system × script"""
    print("\n" + "="*80)
    print("CROSS-DIMENSIONAL ANALYSIS")
    print("="*80)
    
    # System × Print Method
    if 'ocr_system' in df.columns and 'print_method' in df.columns:
        print("\n--- OCR System × Print Method ---\n")
        
        df_filtered = df[(df['ocr_system'].notna()) & (df['print_method'].notna())].copy()
        
        if len(df_filtered) > 0 and 'mean_perplexity' in df_filtered.columns:
            results = []
            for (system, pm), group in df_filtered.groupby(['ocr_system', 'print_method']):
                perp_valid = group['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(perp_valid) > 0:
                    results.append({
                        'ocr_system': system,
                        'print_method': pm,
                        'volume_count': len(group),
                        'avg_perplexity': perp_valid.mean(),
                        'median_perplexity': perp_valid.median(),
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(['ocr_system', 'avg_perplexity'])
                print(results_df.to_string(index=False))
    
    # System × Script
    if 'ocr_system' in df.columns and 'script' in df.columns:
        print("\n--- OCR System × Script ---\n")
        
        df_filtered = df[(df['ocr_system'].notna()) & (df['script'].notna())].copy()
        
        if len(df_filtered) > 0 and 'mean_perplexity' in df_filtered.columns:
            results = []
            for (system, script), group in df_filtered.groupby(['ocr_system', 'script']):
                perp_valid = group['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(perp_valid) > 0:
                    results.append({
                        'ocr_system': system,
                        'script': script,
                        'volume_count': len(group),
                        'avg_perplexity': perp_valid.mean(),
                        'median_perplexity': perp_valid.median(),
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values(['ocr_system', 'avg_perplexity'])
                print(results_df.to_string(index=False))


def export_enriched_data(df: pd.DataFrame, output_dir: str = 'output'):
    """Export combined data with metadata for further analysis"""
    output_path = Path(output_dir) / 'combined_stats_with_metadata.parquet'
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Exported enriched data to: {output_path}")
    
    # Also export CSV
    csv_path = Path(output_dir) / 'combined_stats_with_metadata.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Exported CSV to: {csv_path}")
    
    print(f"\n✓ Enriched data exported to:")
    print(f"  - {output_path}")
    print(f"  - {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze OCR quality by system, print method, and script'
    )
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Directory containing stats parquet files')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing metadata CSV files')
    parser.add_argument('--export', action='store_true',
                       help='Export combined data with metadata')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--by-system', action='store_true',
                       help='Analyze by OCR system')
    parser.add_argument('--by-print-method', action='store_true',
                       help='Analyze by print method')
    parser.add_argument('--by-script', action='store_true',
                       help='Analyze by script')
    parser.add_argument('--cross', action='store_true',
                       help='Cross-dimensional analysis')
    
    args = parser.parse_args()
    
    # Load all data
    print("Loading statistics and metadata...")
    combined_df = load_all_stats_with_metadata(args.output_dir, args.data_dir)
    
    if combined_df.empty:
        print("No data available. Run analysis first.")
        return
    
    # Print metadata summary
    print_metadata_summary(combined_df)
    
    # If no specific analysis requested, run all
    if not any([args.by_system, args.by_print_method, args.by_script, args.cross]):
        args.all = True
    
    # Run requested analyses
    if args.all or args.by_system:
        analyze_by_system(combined_df)
    
    if args.all or args.by_print_method:
        analyze_by_print_method(combined_df)
    
    if args.all or args.by_script:
        analyze_by_script(combined_df)
    
    if args.all or args.cross:
        analyze_cross_dimensions(combined_df)
    
    # Export if requested
    if args.export:
        export_enriched_data(combined_df, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print("\nFor more detailed queries, use the exported data:")
    print("  python explore_stats.py --interactive")
    print("  SQL> CREATE VIEW combined AS SELECT * FROM 'output/combined_stats_with_metadata.parquet'")
    print("  SQL> SELECT * FROM combined WHERE print_method = 'PrintMethod_Modern' LIMIT 10;")


if __name__ == '__main__':
    main()
