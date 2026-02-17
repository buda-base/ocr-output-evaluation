"""
Metadata loading and enrichment for OCR analysis
"""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_metadata(data_dir: str = 'data') -> tuple:
    """
    Load print methods and scripts metadata from CSV files
    
    Args:
        data_dir: Directory containing metadata CSV files
    
    Returns:
        Tuple of (print_methods_df, scripts_df)
    """
    data_path = Path(data_dir)
    
    print_methods_df = None
    scripts_df = None
    
    # Load print methods
    pm_path = data_path / 'print_methods.csv'
    if pm_path.exists():
        print_methods_df = pd.read_csv(pm_path)
        # Rename columns for clarity
        print_methods_df.columns = ['w_id', 'print_method']
        logger.info(f"Loaded {len(print_methods_df)} print methods from {pm_path}")
    else:
        logger.warning(f"Print methods file not found: {pm_path}")
    
    # Load scripts
    scripts_path = data_path / 'scripts.csv'
    if scripts_path.exists():
        scripts_df = pd.read_csv(scripts_path)
        # Rename columns for clarity
        scripts_df.columns = ['w_id', 'script']
        logger.info(f"Loaded {len(scripts_df)} scripts from {scripts_path}")
    else:
        logger.warning(f"Scripts file not found: {scripts_path}")
    
    return print_methods_df, scripts_df


def enrich_stats_with_metadata(stats_df: pd.DataFrame, 
                                print_methods_df: pd.DataFrame = None,
                                scripts_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Enrich statistics DataFrame with print method and script metadata
    
    Args:
        stats_df: DataFrame with OCR statistics (must have w_id column)
        print_methods_df: DataFrame with print methods
        scripts_df: DataFrame with scripts
    
    Returns:
        Enriched DataFrame with metadata columns
    """
    if stats_df.empty:
        return stats_df
    
    result_df = stats_df.copy()
    
    # Join print methods
    if print_methods_df is not None and 'w_id' in stats_df.columns:
        result_df = result_df.merge(
            print_methods_df,
            on='w_id',
            how='left'
        )
        logger.info(f"Joined print methods: {result_df['print_method'].notna().sum()} matches")
    
    # Join scripts
    if scripts_df is not None and 'w_id' in stats_df.columns:
        result_df = result_df.merge(
            scripts_df,
            on='w_id',
            how='left'
        )
        logger.info(f"Joined scripts: {result_df['script'].notna().sum()} matches")
    
    return result_df


def load_all_stats_with_metadata(output_dir: str = 'output', 
                                  data_dir: str = 'data') -> pd.DataFrame:
    """
    Load all OCR statistics and enrich with metadata, adding system column
    
    Args:
        output_dir: Directory containing stats parquet files
        data_dir: Directory containing metadata CSV files
    
    Returns:
        Combined DataFrame with all OCR types and metadata
    """
    output_path = Path(output_dir)
    
    # Load metadata
    print_methods_df, scripts_df = load_metadata(data_dir)
    
    all_dfs = []
    
    # Load Google Books
    gb_path = output_path / 'google_books_stats.parquet'
    if gb_path.exists():
        gb_df = pd.read_parquet(gb_path)
        gb_df['ocr_system'] = 'google_ocr'
        gb_df['ocr_type'] = 'google_books'
        gb_df = enrich_stats_with_metadata(gb_df, print_methods_df, scripts_df)
        all_dfs.append(gb_df)
        logger.info(f"Loaded Google Books: {len(gb_df)} volumes")
    
    # Load Google Vision
    gv_path = output_path / 'google_vision_stats.parquet'
    if gv_path.exists():
        gv_df = pd.read_parquet(gv_path)
        gv_df['ocr_system'] = 'google_ocr'
        gv_df['ocr_type'] = 'google_vision'
        gv_df = enrich_stats_with_metadata(gv_df, print_methods_df, scripts_df)
        all_dfs.append(gv_df)
        logger.info(f"Loaded Google Vision: {len(gv_df)} volumes")
    
    # Load OCRv1
    ocrv1_path = output_path / 'ocrv1_ws_ldv1_stats.parquet'
    if ocrv1_path.exists():
        ocrv1_df = pd.read_parquet(ocrv1_path)
        ocrv1_df['ocr_system'] = 'ocrv1-ws-ldv1'
        ocrv1_df['ocr_type'] = 'ocrv1-ws-ldv1'
        ocrv1_df = enrich_stats_with_metadata(ocrv1_df, print_methods_df, scripts_df)
        all_dfs.append(ocrv1_df)
        logger.info(f"Loaded OCRv1-WS-LDv1: {len(ocrv1_df)} volumes")
    
    if not all_dfs:
        logger.warning("No stats files found")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined_df)} volumes")
    
    return combined_df


def print_metadata_summary(combined_df: pd.DataFrame):
    """Print summary of metadata coverage and distributions"""
    print("\n" + "="*80)
    print("METADATA SUMMARY")
    print("="*80)
    
    if combined_df.empty:
        print("No data available")
        return
    
    # OCR System distribution
    if 'ocr_system' in combined_df.columns:
        print("\nOCR System Distribution:")
        system_counts = combined_df['ocr_system'].value_counts()
        for system, count in system_counts.items():
            pct = count / len(combined_df) * 100
            print(f"  {system}: {count:,} ({pct:.1f}%)")
    
    # Print Method distribution
    if 'print_method' in combined_df.columns:
        print("\nPrint Method Distribution:")
        pm_counts = combined_df['print_method'].value_counts()
        total_with_pm = combined_df['print_method'].notna().sum()
        print(f"  Coverage: {total_with_pm:,} / {len(combined_df):,} ({total_with_pm/len(combined_df)*100:.1f}%)")
        for pm, count in pm_counts.head(10).items():
            pct = count / len(combined_df) * 100
            print(f"  {pm}: {count:,} ({pct:.1f}%)")
        if len(pm_counts) > 10:
            print(f"  ... and {len(pm_counts) - 10} more")
    
    # Script distribution
    if 'script' in combined_df.columns:
        print("\nScript Distribution:")
        script_counts = combined_df['script'].value_counts()
        total_with_script = combined_df['script'].notna().sum()
        print(f"  Coverage: {total_with_script:,} / {len(combined_df):,} ({total_with_script/len(combined_df)*100:.1f}%)")
        for script, count in script_counts.items():
            pct = count / len(combined_df) * 100
            print(f"  {script}: {count:,} ({pct:.1f}%)")
