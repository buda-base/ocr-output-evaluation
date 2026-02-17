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
    ocrv1_path = Path(output_dir) / 'ocrv1_ws_ldv1_stats.parquet'
    
    gb_df = None
    gv_df = None
    ocrv1_df = None
    
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
    
    if ocrv1_path.exists():
        ocrv1_df = pd.read_parquet(ocrv1_path)
        print(f"Loaded OCRv1-WS-LDv1: {len(ocrv1_df)} volumes")
        has_perplexity_ocrv1 = 'mean_perplexity' in ocrv1_df.columns
        if has_perplexity_ocrv1:
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
    
    if ocrv1_df is not None:
        print("\nGenerating OCRv1-WS-LDv1 plots...")
        # OCRv1 only has perplexity, no confidence scores
        if 'mean_perplexity' in ocrv1_df.columns:
            print("  Generating perplexity plots...")
            plot_perplexity_distribution(ocrv1_df, 'OCRv1-WS-LDv1',
                                        plots_path / 'ocrv1_perplexity_dist.png')
            plot_perplexity_percentiles(ocrv1_df, 'OCRv1-WS-LDv1',
                                       plots_path / 'ocrv1_perplexity_percentiles.png')
    
    if gb_df is not None and gv_df is not None:
        print("\nGenerating comparison plots...")
        plot_confidence_comparison(gb_df, gv_df,
                                  plots_path / 'gb_vs_gv_comparison.png')
    
    print(f"\n✓ All plots saved to: {plots_path}")
    print(f"\nGenerated plots:")
    print(f"  Confidence: distribution, categories, vs pages, percentiles")
    if (gb_df is not None and 'mean_perplexity' in gb_df.columns) or \
       (gv_df is not None and 'mean_perplexity' in gv_df.columns) or \
       (ocrv1_df is not None and 'mean_perplexity' in ocrv1_df.columns):
        print(f"  Perplexity: distribution, vs confidence, percentiles")


def plot_perplexity_by_system(combined_df: pd.DataFrame, save_path: str = None):
    """
    Compare perplexity distributions between OCR systems
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    systems = ['google_ocr', 'ocrv1-ws-ldv1']
    data_to_plot = []
    labels = []
    
    for system in systems:
        system_df = combined_df[combined_df['ocr_system'] == system]
        if len(system_df) > 0 and 'mean_perplexity' in system_df.columns:
            perp_valid = system_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(perp_valid) > 0:
                data_to_plot.append(np.log10(perp_valid))
                labels.append(f"{system}\n(n={len(perp_valid)})")
    
    if not data_to_plot:
        print("Warning: No valid perplexity data for system comparison")
        plt.close()
        return
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Mean Perplexity (log10 scale)')
    ax.set_title('Perplexity Distribution by OCR System')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add secondary y-axis with actual perplexity values
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: 10**x, lambda x: np.log10(x)))
    ax2.set_ylabel('Actual Perplexity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_perplexity_by_print_method(combined_df: pd.DataFrame, save_path: str = None, top_n: int = 10):
    """
    Compare perplexity across print methods (top N by volume count)
    """
    if 'print_method' not in combined_df.columns:
        print("Warning: No print_method column in data")
        return
    
    df_with_pm = combined_df[combined_df['print_method'].notna()].copy()
    
    if len(df_with_pm) == 0:
        print("Warning: No volumes with print method metadata")
        return
    
    # Get top N print methods by volume count
    pm_counts = df_with_pm['print_method'].value_counts().head(top_n)
    top_pms = pm_counts.index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    data_to_plot = []
    labels = []
    
    for pm in top_pms:
        pm_df = df_with_pm[df_with_pm['print_method'] == pm]
        if 'mean_perplexity' in pm_df.columns:
            perp_valid = pm_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(perp_valid) > 0:
                data_to_plot.append(np.log10(perp_valid))
                # Truncate long names
                pm_short = pm.replace('PrintMethod_', '')
                labels.append(f"{pm_short}\n(n={len(perp_valid)})")
    
    if not data_to_plot:
        print("Warning: No valid perplexity data for print method comparison")
        plt.close()
        return
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes with a gradient
    colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Mean Perplexity (log10 scale)')
    ax.set_xlabel('Print Method')
    ax.set_title(f'Perplexity Distribution by Print Method (Top {len(data_to_plot)})')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add secondary y-axis
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: 10**x, lambda x: np.log10(x)))
    ax2.set_ylabel('Actual Perplexity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_perplexity_by_script(combined_df: pd.DataFrame, save_path: str = None):
    """
    Compare perplexity across scripts
    """
    if 'script' not in combined_df.columns:
        print("Warning: No script column in data")
        return
    
    df_with_script = combined_df[combined_df['script'].notna()].copy()
    
    if len(df_with_script) == 0:
        print("Warning: No volumes with script metadata")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scripts = df_with_script['script'].unique()
    data_to_plot = []
    labels = []
    
    for script in scripts:
        script_df = df_with_script[df_with_script['script'] == script]
        if 'mean_perplexity' in script_df.columns:
            perp_valid = script_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(perp_valid) > 0:
                data_to_plot.append(np.log10(perp_valid))
                script_short = script.replace('Script', '')
                labels.append(f"{script_short}\n(n={len(perp_valid)})")
    
    if not data_to_plot:
        print("Warning: No valid perplexity data for script comparison")
        plt.close()
        return
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Mean Perplexity (log10 scale)')
    ax.set_xlabel('Script')
    ax.set_title('Perplexity Distribution by Script')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add secondary y-axis
    ax2 = ax.secondary_yaxis('right', functions=(lambda x: 10**x, lambda x: np.log10(x)))
    ax2.set_ylabel('Actual Perplexity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_system_by_print_method_heatmap(combined_df: pd.DataFrame, save_path: str = None):
    """
    Heatmap showing average perplexity for system × print method combinations
    """
    if 'ocr_system' not in combined_df.columns or 'print_method' not in combined_df.columns:
        print("Warning: Missing required columns for heatmap")
        return
    
    df_filtered = combined_df[(combined_df['ocr_system'].notna()) & 
                               (combined_df['print_method'].notna())].copy()
    
    if len(df_filtered) == 0 or 'mean_perplexity' not in df_filtered.columns:
        print("Warning: No valid data for heatmap")
        return
    
    # Calculate average perplexity for each combination
    pivot_data = df_filtered.groupby(['ocr_system', 'print_method'])['mean_perplexity'].agg([
        ('avg_perplexity', lambda x: x.replace([np.inf, -np.inf], np.nan).dropna().mean()),
        ('count', 'count')
    ]).reset_index()
    
    # Only keep combinations with at least 5 volumes
    pivot_data = pivot_data[pivot_data['count'] >= 5]
    
    if len(pivot_data) == 0:
        print("Warning: Not enough data for heatmap")
        return
    
    # Create pivot table
    pivot_table = pivot_data.pivot(index='print_method', columns='ocr_system', values='avg_perplexity')
    
    # Sort by overall average
    pivot_table['_avg'] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values('_avg')
    pivot_table = pivot_table.drop('_avg', axis=1)
    
    # Limit to top 15 print methods
    if len(pivot_table) > 15:
        pivot_table = pivot_table.head(15)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(pivot_table) * 0.4)))
    
    im = ax.imshow(pivot_table.values, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels([pm.replace('PrintMethod_', '') for pm in pivot_table.index])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Perplexity', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.0f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('Average Perplexity: OCR System × Print Method')
    ax.set_xlabel('OCR System')
    ax.set_ylabel('Print Method')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_metadata_plots(output_dir: str = 'output', data_dir: str = 'data', 
                            plots_dir: str = 'plots'):
    """
    Generate all plots with metadata dimensions
    """
    from metadata_loader import load_all_stats_with_metadata
    
    # Create plots directory if it doesn't exist
    Path(plots_dir).mkdir(exist_ok=True)
    
    print("Loading statistics and metadata...")
    combined_df = load_all_stats_with_metadata(output_dir, data_dir)
    
    if combined_df.empty:
        print("No data available")
        return
    
    print(f"Loaded {len(combined_df)} volumes with metadata")
    
    # Generate plots
    print("\nGenerating metadata-based plots...")
    
    if 'mean_perplexity' in combined_df.columns:
        print("  - Perplexity by OCR system...")
        plot_perplexity_by_system(combined_df, 
                                  save_path=f"{plots_dir}/perplexity_by_system.png")
        
        if 'print_method' in combined_df.columns:
            print("  - Perplexity by print method...")
            plot_perplexity_by_print_method(combined_df,
                                           save_path=f"{plots_dir}/perplexity_by_print_method.png")
            
            print("  - System × Print method heatmap...")
            plot_system_by_print_method_heatmap(combined_df,
                                               save_path=f"{plots_dir}/system_x_print_method_heatmap.png")
        
        if 'script' in combined_df.columns:
            print("  - Perplexity by script...")
            plot_perplexity_by_script(combined_df,
                                     save_path=f"{plots_dir}/perplexity_by_script.png")
    
    print("\n✓ Metadata plots generated successfully!")
    print(f"  Plots saved in: {plots_dir}/")



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate visualization plots')
    parser.add_argument('--output-dir', default='output',
                       help='Directory containing stats files')
    parser.add_argument('--plots-dir', default='plots',
                       help='Directory to save plots')
    parser.add_argument('--metadata', action='store_true',
                       help='Generate metadata-based plots (by system, print method, script)')
    parser.add_argument('--data-dir', default='data',
                       help='Directory containing metadata CSV files (for --metadata)')
    
    args = parser.parse_args()
    
    try:
        if args.metadata:
            generate_metadata_plots(args.output_dir, args.data_dir, args.plots_dir)
        else:
            generate_all_plots(args.output_dir, args.plots_dir)
    except ImportError:
        print("Error: matplotlib not installed")
        print("Install with: pip install matplotlib")
        exit(1)
