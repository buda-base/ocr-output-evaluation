"""
Visualization utilities for OCR confidence analysis

Note: Requires matplotlib. Install with: pip install matplotlib
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_confidence_distribution(stats_df: pd.DataFrame, 
                                 ocr_type: str = 'Google Books',
                                 save_path: str = None):
    """
    Plot histogram of mean confidence across volumes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out invalid values (NaN, inf, -inf)
    valid_data = stats_df['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) == 0:
        print(f"Warning: No valid data for {ocr_type}")
        plt.close()
        return
    
    ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0.7, color='red', linestyle='--', label='Low threshold (0.7)')
    ax.axvline(0.9, color='green', linestyle='--', label='High threshold (0.9)')
    
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Number of Volumes')
    ax.set_title(f'{ocr_type} - Distribution of Mean Confidence Across Volumes\n(n={len(valid_data)} volumes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_comparison(gb_df: pd.DataFrame, gv_df: pd.DataFrame,
                               save_path: str = None):
    """
    Compare confidence distributions between Google Books and Google Vision
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter invalid values for both
    gb_valid = gb_df['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
    gv_valid = gv_df['mean_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
    
    # Google Books
    if len(gb_valid) > 0:
        axes[0].hist(gb_valid, bins=50, alpha=0.7, 
                    color='blue', edgecolor='black')
        axes[0].axvline(0.7, color='red', linestyle='--')
        axes[0].axvline(0.9, color='green', linestyle='--')
    axes[0].set_xlabel('Mean Confidence')
    axes[0].set_ylabel('Number of Volumes')
    axes[0].set_title(f'Google Books (n={len(gb_valid)})')
    axes[0].grid(True, alpha=0.3)
    
    # Google Vision
    if len(gv_valid) > 0:
        axes[1].hist(gv_valid, bins=50, alpha=0.7,
                    color='orange', edgecolor='black')
        axes[1].axvline(0.7, color='red', linestyle='--', label='Low (0.7)')
        axes[1].axvline(0.9, color='green', linestyle='--', label='High (0.9)')
    axes[1].set_xlabel('Mean Confidence')
    axes[1].set_ylabel('Number of Volumes')
    axes[1].set_title(f'Google Vision (n={len(gv_valid)})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_categories(stats_df: pd.DataFrame,
                               ocr_type: str = 'Google Books',
                               save_path: str = None):
    """
    Plot pie chart of confidence categories
    """
    # Calculate averages, filtering out invalid values
    avg_high = stats_df['pct_high_conf'].replace([np.inf, -np.inf], np.nan).mean()
    avg_medium = stats_df['pct_medium_conf'].replace([np.inf, -np.inf], np.nan).mean()
    avg_low = stats_df['pct_low_conf'].replace([np.inf, -np.inf], np.nan).mean()
    
    # Check if we have valid data
    if pd.isna(avg_high) and pd.isna(avg_medium) and pd.isna(avg_low):
        print(f"Warning: No valid data for {ocr_type} pie chart")
        return
    
    # Replace NaN with 0
    avg_high = avg_high if not pd.isna(avg_high) else 0
    avg_medium = avg_medium if not pd.isna(avg_medium) else 0
    avg_low = avg_low if not pd.isna(avg_low) else 0
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [avg_high, avg_medium, avg_low]
    labels = [f'High (≥90%)\n{avg_high:.1f}%',
              f'Medium (70-90%)\n{avg_medium:.1f}%', 
              f'Low (<70%)\n{avg_low:.1f}%']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='', 
           startangle=90, textprops={'fontsize': 12})
    ax.set_title(f'{ocr_type} - Average Confidence Category Distribution', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_vs_pages(stats_df: pd.DataFrame,
                             ocr_type: str = 'Google Books',
                             save_path: str = None):
    """
    Scatter plot of confidence vs number of pages
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    page_col = 'total_pages' if 'total_pages' in stats_df.columns else 'total_records'
    
    # Filter out invalid values
    plot_df = stats_df[[page_col, 'mean_confidence']].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(plot_df) == 0:
        print(f"Warning: No valid data for {ocr_type} scatter plot")
        plt.close()
        return
    
    scatter = ax.scatter(plot_df[page_col], plot_df['mean_confidence'],
                        alpha=0.5, s=20, c=plot_df['mean_confidence'],
                        cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='Low threshold')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='High threshold')
    
    ax.set_xlabel('Number of Pages')
    ax.set_ylabel('Mean Confidence')
    ax.set_title(f'{ocr_type} - Confidence vs Volume Size (n={len(plot_df)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Confidence')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_percentile_ranges(stats_df: pd.DataFrame,
                           ocr_type: str = 'Google Books',
                           save_path: str = None):
    """
    Box plot showing confidence percentile ranges
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot, filtering out invalid values
    data = [
        stats_df['p10_confidence'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p25_confidence'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['median_confidence'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p75_confidence'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p90_confidence'].replace([np.inf, -np.inf], np.nan).dropna()
    ]
    
    # Check if we have any valid data
    if all(len(d) == 0 for d in data):
        print(f"Warning: No valid data for {ocr_type} percentile plot")
        plt.close()
        return
    
    positions = [10, 25, 50, 75, 90]
    
    bp = ax.boxplot(data, positions=positions, widths=8, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    ax.axhline(0.7, color='red', linestyle='--', alpha=0.3, label='Low threshold')
    ax.axhline(0.9, color='green', linestyle='--', alpha=0.3, label='High threshold')
    
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Confidence')
    ax.set_title(f'{ocr_type} - Confidence Distribution Across Percentiles')
    ax.set_xticks(positions)
    ax.set_xticklabels(['P10', 'P25', 'P50', 'P75', 'P90'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_perplexity_distribution(stats_df: pd.DataFrame,
                                  ocr_type: str = 'Google Books',
                                  save_path: str = None):
    """
    Plot histogram of mean perplexity across volumes (log scale)
    """
    if 'mean_perplexity' not in stats_df.columns:
        print(f"Warning: No perplexity data available for {ocr_type}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out invalid values (NaN, inf)
    valid_data = stats_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_data) == 0:
        print(f"Warning: No valid perplexity data for {ocr_type}")
        plt.close()
        return
    
    # Calculate percentile-based thresholds
    p33 = valid_data.quantile(0.33)
    p66 = valid_data.quantile(0.66)
    
    # Use log scale for perplexity
    ax.hist(np.log10(valid_data), bins=50, alpha=0.7, edgecolor='black', color='purple')
    
    # Add percentile threshold lines (relative to dataset)
    ax.axvline(np.log10(p33), color='green', linestyle='--', 
               label=f'P33 (best third) = {p33:.0f}', linewidth=2)
    ax.axvline(np.log10(p66), color='red', linestyle='--', 
               label=f'P66 (worst third) = {p66:.0f}', linewidth=2)
    
    ax.set_xlabel('Perplexity (log10 scale)')
    ax.set_ylabel('Number of Volumes')
    ax.set_title(f'{ocr_type} - Distribution of Mean Perplexity Across Volumes\n' + 
                f'(n={len(valid_data)} volumes, median={valid_data.median():.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add second x-axis with actual values
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # Use actual percentiles for tick labels
    tick_locs = [np.log10(valid_data.min()), np.log10(valid_data.quantile(0.5)), np.log10(valid_data.max())]
    tick_labels = [f'{valid_data.min():.0f}', f'{valid_data.median():.0f}', f'{valid_data.max():.0f}']
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel('Perplexity (actual value)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_vs_perplexity(stats_df: pd.DataFrame,
                                   ocr_type: str = 'Google Books',
                                   save_path: str = None):
    """
    Scatter plot of confidence vs perplexity
    """
    if 'mean_perplexity' not in stats_df.columns:
        print(f"Warning: No perplexity data available for {ocr_type}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out invalid values
    plot_df = stats_df[['mean_confidence', 'mean_perplexity']].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(plot_df) == 0:
        print(f"Warning: No valid data for {ocr_type} confidence vs perplexity plot")
        plt.close()
        return
    
    # Calculate percentile thresholds
    p33 = plot_df['mean_perplexity'].quantile(0.33)
    p66 = plot_df['mean_perplexity'].quantile(0.66)
    
    # Plot with log scale on y-axis
    scatter = ax.scatter(plot_df['mean_confidence'], plot_df['mean_perplexity'],
                        alpha=0.5, s=20, c=plot_df['mean_perplexity'],
                        cmap='RdYlGn_r', norm=plt.matplotlib.colors.LogNorm())
    
    # Add percentile threshold lines (dataset-relative)
    ax.axhline(p33, color='green', linestyle='--', alpha=0.5, 
               label=f'P33 (best) = {p33:.0f}')
    ax.axhline(p66, color='red', linestyle='--', alpha=0.5, 
               label=f'P66 (worst) = {p66:.0f}')
    ax.axvline(0.7, color='orange', linestyle='--', alpha=0.3)
    ax.axvline(0.9, color='blue', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Mean Perplexity (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{ocr_type} - Confidence vs Perplexity (n={len(plot_df)})\n' +
                f'Thresholds are dataset-relative (P33={p33:.0f}, P66={p66:.0f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Perplexity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_perplexity_percentiles(stats_df: pd.DataFrame,
                                 ocr_type: str = 'Google Books',
                                 save_path: str = None):
    """
    Box plot showing perplexity percentile ranges
    """
    if 'mean_perplexity' not in stats_df.columns:
        print(f"Warning: No perplexity data available for {ocr_type}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot, filtering out invalid values
    data = [
        stats_df['p10_perplexity'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p25_perplexity'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['median_perplexity'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p75_perplexity'].replace([np.inf, -np.inf], np.nan).dropna(),
        stats_df['p90_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
    ]
    
    # Check if we have any valid data
    if all(len(d) == 0 for d in data):
        print(f"Warning: No valid perplexity data for {ocr_type} percentile plot")
        plt.close()
        return
    
    # Calculate overall dataset percentiles for reference lines
    all_perp = stats_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
    p33_overall = all_perp.quantile(0.33)
    p66_overall = all_perp.quantile(0.66)
    
    positions = [10, 25, 50, 75, 90]
    
    bp = ax.boxplot(data, positions=positions, widths=8, patch_artist=True,
                    boxprops=dict(facecolor='plum', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
    
    # Add percentile reference lines
    ax.axhline(p33_overall, color='green', linestyle='--', alpha=0.3, 
               label=f'Dataset P33 = {p33_overall:.0f}')
    ax.axhline(p66_overall, color='red', linestyle='--', alpha=0.3, 
               label=f'Dataset P66 = {p66_overall:.0f}')
    
    ax.set_xlabel('Percentile')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_yscale('log')
    ax.set_title(f'{ocr_type} - Perplexity Distribution Across Percentiles\n' +
                f'Dataset median = {all_perp.median():.0f}')
    ax.set_xticks(positions)
    ax.set_xticklabels(['P10', 'P25', 'P50', 'P75', 'P90'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(output_dir: str = 'output', 
                      plots_dir: str = 'plots'):
    """
    Generate all visualization plots
    """
    # Create plots directory
    plots_path = Path(plots_dir)
    plots_path.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    # Load data
    gb_path = Path(output_dir) / 'google_books_stats.parquet'
    gv_path = Path(output_dir) / 'google_vision_stats.parquet'
    
    gb_df = None
    gv_df = None
    
    if gb_path.exists():
        gb_df = pd.read_parquet(gb_path)
        print(f"Loaded Google Books: {len(gb_df)} volumes")
        has_perplexity_gb = 'mean_perplexity' in gb_df.columns
        if has_perplexity_gb:
            print(f"  ✓ Perplexity data available")
        else:
            print(f"  ℹ No perplexity data (disabled during analysis)")
    
    if gv_path.exists():
        gv_df = pd.read_parquet(gv_path)
        print(f"Loaded Google Vision: {len(gv_df)} volumes")
        has_perplexity_gv = 'mean_perplexity' in gv_df.columns
        if has_perplexity_gv:
            print(f"  ✓ Perplexity data available")
        else:
            print(f"  ℹ No perplexity data (disabled during analysis)")
    
    # Generate plots
    if gb_df is not None:
        print("\nGenerating Google Books plots...")
        plot_confidence_distribution(gb_df, 'Google Books',
                                    plots_path / 'gb_confidence_dist.png')
        plot_confidence_categories(gb_df, 'Google Books',
                                  plots_path / 'gb_confidence_categories.png')
        plot_confidence_vs_pages(gb_df, 'Google Books',
                                plots_path / 'gb_confidence_vs_pages.png')
        plot_percentile_ranges(gb_df, 'Google Books',
                              plots_path / 'gb_percentile_ranges.png')
        
        # Perplexity plots
        if 'mean_perplexity' in gb_df.columns:
            print("  Generating perplexity plots...")
            plot_perplexity_distribution(gb_df, 'Google Books',
                                        plots_path / 'gb_perplexity_dist.png')
            plot_confidence_vs_perplexity(gb_df, 'Google Books',
                                         plots_path / 'gb_confidence_vs_perplexity.png')
            plot_perplexity_percentiles(gb_df, 'Google Books',
                                       plots_path / 'gb_perplexity_percentiles.png')
    
    if gv_df is not None:
        print("\nGenerating Google Vision plots...")
        plot_confidence_distribution(gv_df, 'Google Vision',
                                    plots_path / 'gv_confidence_dist.png')
        plot_confidence_categories(gv_df, 'Google Vision',
                                  plots_path / 'gv_confidence_categories.png')
        plot_confidence_vs_pages(gv_df, 'Google Vision',
                                plots_path / 'gv_confidence_vs_pages.png')
        plot_percentile_ranges(gv_df, 'Google Vision',
                              plots_path / 'gv_percentile_ranges.png')
        
        # Perplexity plots
        if 'mean_perplexity' in gv_df.columns:
            print("  Generating perplexity plots...")
            plot_perplexity_distribution(gv_df, 'Google Vision',
                                        plots_path / 'gv_perplexity_dist.png')
            plot_confidence_vs_perplexity(gv_df, 'Google Vision',
                                         plots_path / 'gv_confidence_vs_perplexity.png')
            plot_perplexity_percentiles(gv_df, 'Google Vision',
                                       plots_path / 'gv_perplexity_percentiles.png')
    
    if gb_df is not None and gv_df is not None:
        print("\nGenerating comparison plots...")
        plot_confidence_comparison(gb_df, gv_df,
                                  plots_path / 'gb_vs_gv_comparison.png')
    
    print(f"\n✓ All plots saved to: {plots_path}")
    print(f"\nGenerated plots:")
    print(f"  Confidence: distribution, categories, vs pages, percentiles")
    if (gb_df is not None and 'mean_perplexity' in gb_df.columns) or \
       (gv_df is not None and 'mean_perplexity' in gv_df.columns):
        print(f"  Perplexity: distribution, vs confidence, percentiles")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization plots')
    parser.add_argument('--output-dir', default='output',
                       help='Directory containing stats files')
    parser.add_argument('--plots-dir', default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    try:
        generate_all_plots(args.output_dir, args.plots_dir)
    except ImportError:
        print("Error: matplotlib not installed")
        print("Install with: pip install matplotlib")
        exit(1)
