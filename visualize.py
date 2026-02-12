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
    
    if gv_path.exists():
        gv_df = pd.read_parquet(gv_path)
        print(f"Loaded Google Vision: {len(gv_df)} volumes")
    
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
    
    if gb_df is not None and gv_df is not None:
        print("\nGenerating comparison plots...")
        plot_confidence_comparison(gb_df, gv_df,
                                  plots_path / 'gb_vs_gv_comparison.png')
    
    print(f"\nAll plots saved to: {plots_path}")


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
