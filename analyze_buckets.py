
#!/usr/bin/env python3
"""
Analyze OCR quality by specific buckets:
- Modern (all scripts)
- Woodblock (all scripts)
- Manuscript (ScriptTibt/ScriptDbuCan)
- Manuscript (ScriptDbuMed)
"""
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
import matplotlib.pyplot as plt

from metadata_loader import load_all_stats_with_metadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_SCRIPTS = ['ScriptTibt', 'ScriptDbuCan', 'ScriptDbuMed']

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and prepare data for analysis
    """
    # 1. Filter by script
    if 'script' not in df.columns:
        logger.warning("No script column found. Cannot filter by script.")
        return pd.DataFrame()
    
    df_filtered = df[df['script'].isin(ALLOWED_SCRIPTS)].copy()
    logger.info(f"Filtered to {len(df_filtered)} volumes with allowed scripts")
    
    # 2. Merge Google Books and Google Vision
    if 'ocr_system' in df_filtered.columns:
        # Normalize Google Books confidence (0-100 -> 0-1)
        # We detect this by ocr_type if available, or heuristic
        if 'ocr_type' in df_filtered.columns and 'mean_confidence' in df_filtered.columns:
            # Check if Google Books data is on 0-100 scale
            gb_mask = df_filtered['ocr_type'] == 'google_books'
            if gb_mask.any():
                # Check if values are > 1.0
                if df_filtered.loc[gb_mask, 'mean_confidence'].max() > 1.0:
                    logger.info("Normalizing Google Books confidence from 0-100 to 0-1 scale")
                    df_filtered.loc[gb_mask, 'mean_confidence'] /= 100.0
        
        df_filtered['ocr_system_merged'] = df_filtered['ocr_system'].replace({
            'google_ocr': 'Google OCR',
            'ocrv1-ws-ldv1': 'OCRv1'
        })
    
    return df_filtered

def assign_bucket(row):
    """Assign volume to a bucket based on print method and script"""
    pm = row.get('print_method')
    script = row.get('script')
    
    if pd.isna(pm):
        return None
    
    if pm == 'PrintMethod_Modern':
        return 'Modern'
    elif pm == 'PrintMethod_Relief_WoodBlock':
        return 'Woodblock'
    elif pm == 'PrintMethod_Manuscript':
        if script in ['ScriptTibt', 'ScriptDbuCan']:
            return 'Manuscript (Uchen)'
        elif script == 'ScriptDbuMed':
            return 'Manuscript (Umed)'
    
    return None

def analyze_buckets(df: pd.DataFrame):
    """
    Compute statistics for each bucket
    """
    df['bucket'] = df.apply(assign_bucket, axis=1)
    
    # Filter out rows without a bucket
    df_buckets = df[df['bucket'].notna()].copy()
    
    print("\n" + "="*80)
    print("ANALYSIS BY BUCKET")
    print("="*80)
    
    buckets = ['Modern', 'Woodblock', 'Manuscript (Uchen)', 'Manuscript (Umed)']
    
    for bucket in buckets:
        bucket_df = df_buckets[df_buckets['bucket'] == bucket]
        
        print(f"\nBUCKET: {bucket}")
        print(f"  Total Volumes: {len(bucket_df):,}")
        
        # Breakdown by System
        for system in ['Google OCR', 'OCRv1']:
            sys_df = bucket_df[bucket_df['ocr_system_merged'] == system]
            
            print(f"  {system}:")
            print(f"    Volumes: {len(sys_df):,}")
            
            # Confidence
            if 'mean_confidence' in sys_df.columns:
                conf_valid = sys_df['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(conf_valid) > 0:
                    print(f"    Avg Confidence: {conf_valid.mean():.3f}")
            
            # Perplexity
            if 'mean_perplexity' in sys_df.columns:
                perp_valid = sys_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(perp_valid) > 0:
                    print(f"    Avg Perplexity: {perp_valid.mean():.0f}")
                    print(f"    Median Perplexity: {perp_valid.median():.0f}")
                    
            # OCRv1 specific issues
            if system == 'OCRv1' and 'pages_no_tibetan_text' in sys_df.columns:
                no_tib = sys_df['pages_no_tibetan_text'].sum()
                total_recs = sys_df['total_records'].sum() if 'total_records' in sys_df.columns else 0
                if total_recs > 0:
                     print(f"    Pages no Tibetan: {no_tib:,} ({no_tib/total_recs*100:.1f}%)")

def plot_buckets(df: pd.DataFrame, output_dir: str):
    """
    Generate plots for buckets
    """
    df['bucket'] = df.apply(assign_bucket, axis=1)
    df_buckets = df[df['bucket'].notna()].copy()
    
    buckets = ['Modern', 'Woodblock', 'Manuscript (Uchen)', 'Manuscript (Umed)']
    systems = ['Google OCR', 'OCRv1']
    
    # 1. Perplexity Comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    data_to_plot = []
    labels = []
    colors = []
    
    for bucket in buckets:
        for system in systems:
            # Skip OCRv1 for Manuscript (Umed)
            if bucket == 'Manuscript (Umed)' and system == 'OCRv1':
                continue
                
            subset = df_buckets[(df_buckets['bucket'] == bucket) & (df_buckets['ocr_system_merged'] == system)]
            if 'mean_perplexity' in subset.columns:
                perp_valid = subset['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(perp_valid) > 0:
                    data_to_plot.append(np.log10(perp_valid))
                    labels.append(f"{bucket}\n{system}")
                    colors.append('lightblue' if system == 'Google OCR' else 'lightgreen')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_ylabel('Mean Perplexity (log10 scale)')
        ax.set_title('Perplexity Distribution by Bucket and System')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        
        # Add secondary y-axis
        ax2 = ax.secondary_yaxis('right', functions=(lambda x: 10**x, lambda x: np.log10(x)))
        ax2.set_ylabel('Actual Perplexity')
        
        plt.tight_layout()
        save_path = Path(output_dir) / 'buckets_perplexity_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        plt.close()

    # 2. Confidence vs Perplexity (Google OCR only)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, bucket in enumerate(buckets):
        ax = axes[i]
        subset = df_buckets[(df_buckets['bucket'] == bucket) & (df_buckets['ocr_system_merged'] == 'Google OCR')]
        
        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(bucket)
            continue
            
        # Filter valid data
        if 'mean_confidence' in subset.columns and 'mean_perplexity' in subset.columns:
            valid = subset[['mean_confidence', 'mean_perplexity']].replace([np.inf, -np.inf], np.nan).dropna()
            
            # Filter low confidence (< 30%)
            valid = valid[valid['mean_confidence'] >= 0.3]
            
            if len(valid) > 0:
                # Calculate correlation
                corr = valid['mean_confidence'].corr(valid['mean_perplexity'])
                
                # Add correlation to title
                ax.set_title(f'{bucket}\n(n={len(valid)}, r={corr:.2f})')
                
                scatter = ax.scatter(valid['mean_confidence'], valid['mean_perplexity'],
                                   alpha=0.5, s=10, c=valid['mean_perplexity'],
                                   cmap='RdYlGn_r', norm=plt.matplotlib.colors.LogNorm())
                
                # Add trend line (linear regression on log perplexity)
                try:
                    z = np.polyfit(valid['mean_confidence'], np.log10(valid['mean_perplexity']), 1)
                    p = np.poly1d(z)
                    # Generate x values for the line
                    x_range = np.linspace(valid['mean_confidence'].min(), valid['mean_confidence'].max(), 100)
                    # Plot the line (convert back from log for y)
                    ax.plot(x_range, 10**p(x_range), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
                    ax.legend()
                except Exception as e:
                    logger.warning(f"Could not plot trend line for {bucket}: {e}")
                
                ax.set_yscale('log')
                ax.set_xlabel('Mean Confidence')
                ax.set_ylabel('Mean Perplexity (log)')
                ax.grid(True, alpha=0.3)
                
                # Add thresholds
                ax.axvline(0.9, color='green', linestyle='--', alpha=0.5)
                ax.axvline(0.7, color='orange', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'buckets_confidence_vs_perplexity.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze OCR buckets')
    parser.add_argument('--output-dir', default='output', help='Stats directory')
    parser.add_argument('--data-dir', default='data', help='Metadata directory')
    parser.add_argument('--plots-dir', default='plots', help='Plots directory')
    
    args = parser.parse_args()
    
    print("Loading data...")
    df = load_all_stats_with_metadata(args.output_dir, args.data_dir)
    
    if df.empty:
        print("No data found.")
        return
        
    print("Preparing data...")
    df_prep = prepare_data(df)
    
    analyze_buckets(df_prep)
    
    Path(args.plots_dir).mkdir(exist_ok=True)
    plot_buckets(df_prep, args.plots_dir)

if __name__ == '__main__':
    main()
