"""
Statistics computation functions for OCR confidence analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def compute_confidence_stats(df: pd.DataFrame, confidence_col: str = 'confidence') -> Dict[str, Any]:
    """
    Compute comprehensive confidence statistics from a dataframe
    
    Args:
        df: DataFrame with confidence values
        confidence_col: Name of the confidence column
    
    Returns:
        Dictionary with computed statistics
    """
    if df.empty or confidence_col not in df.columns:
        return {}
    
    # Remove null/nan/inf values
    confidence = df[confidence_col].replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(confidence) == 0:
        return {}
    
    stats = {
        # Basic statistics
        'mean_confidence': float(confidence.mean()),
        'median_confidence': float(confidence.median()),
        'std_confidence': float(confidence.std()),
        'min_confidence': float(confidence.min()),
        'max_confidence': float(confidence.max()),
        
        # Percentiles
        'p10_confidence': float(confidence.quantile(0.10)),
        'p25_confidence': float(confidence.quantile(0.25)),
        'p75_confidence': float(confidence.quantile(0.75)),
        'p90_confidence': float(confidence.quantile(0.90)),
        'p95_confidence': float(confidence.quantile(0.95)),
        
        # Counts and thresholds
        'total_pages': len(df),
        'pages_with_confidence': len(confidence),
        'pages_high_conf': int((confidence >= 0.9).sum()),  # >= 90%
        'pages_medium_conf': int(((confidence >= 0.7) & (confidence < 0.9)).sum()),  # 70-90%
        'pages_low_conf': int((confidence < 0.7).sum()),  # < 70%
        
        # Percentage metrics
        'pct_high_conf': float((confidence >= 0.9).sum() / len(confidence) * 100),
        'pct_medium_conf': float(((confidence >= 0.7) & (confidence < 0.9)).sum() / len(confidence) * 100),
        'pct_low_conf': float((confidence < 0.7).sum() / len(confidence) * 100),
    }
    
    return stats


def compute_google_books_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics specific to Google Books format
    
    Args:
        df: DataFrame with Google Books schema
    
    Returns:
        Dictionary with computed statistics
    """
    stats = compute_confidence_stats(df, 'confidence')
    
    if df.empty:
        return stats
    
    # Add Google Books specific metrics
    stats.update({
        'total_records': len(df),
        'successful_pages': int(df['ok'].sum()) if 'ok' in df.columns else 0,
        'failed_pages': int((~df['ok']).sum()) if 'ok' in df.columns else 0,
        'total_lines': int(df['nb_lines'].sum()) if 'nb_lines' in df.columns else 0,
        'mean_lines_per_page': float(df['nb_lines'].mean()) if 'nb_lines' in df.columns else 0.0,
        'total_text_length': int(df['text'].str.len().sum()) if 'text' in df.columns else 0,
        'mean_text_length_per_page': float(df['text'].str.len().mean()) if 'text' in df.columns else 0.0,
    })
    
    # Language distribution (count pages by language)
    if 'languages' in df.columns:
        # Flatten the list of languages and count
        all_langs = []
        for langs in df['languages'].dropna():
            if isinstance(langs, list):
                all_langs.extend(langs)
        
        if all_langs:
            lang_counts = pd.Series(all_langs).value_counts()
            # Store top 3 languages
            for i, (lang, count) in enumerate(lang_counts.head(3).items(), 1):
                stats[f'top_lang_{i}'] = lang
                stats[f'top_lang_{i}_count'] = int(count)
    
    return stats


def compute_google_vision_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics specific to Google Vision format
    
    Args:
        df: DataFrame with Google Vision schema
    
    Returns:
        Dictionary with computed statistics
    """
    stats = compute_confidence_stats(df, 'confidence')
    
    if df.empty:
        return stats
    
    # Add Google Vision specific metrics
    stats.update({
        'total_records': len(df),
        'total_tibetan_lines': int(df['nb_lines_tib'].sum()) if 'nb_lines_tib' in df.columns else 0,
        'mean_tibetan_lines_per_page': float(df['nb_lines_tib'].mean()) if 'nb_lines_tib' in df.columns else 0.0,
        'total_text_length': int(df['text_len'].sum()) if 'text_len' in df.columns else 0,
        'mean_text_length_per_page': float(df['text_len'].mean()) if 'text_len' in df.columns else 0.0,
    })
    
    # Language distribution
    if 'languages' in df.columns:
        all_langs = []
        for langs in df['languages'].dropna():
            if isinstance(langs, list):
                all_langs.extend(langs)
        
        if all_langs:
            lang_counts = pd.Series(all_langs).value_counts()
            for i, (lang, count) in enumerate(lang_counts.head(3).items(), 1):
                stats[f'top_lang_{i}'] = lang
                stats[f'top_lang_{i}_count'] = int(count)
    
    return stats


def analyze_confidence_distribution(df: pd.DataFrame, confidence_col: str = 'confidence', 
                                    bins: int = 20) -> pd.DataFrame:
    """
    Create a histogram of confidence values
    
    Args:
        df: DataFrame with confidence values
        confidence_col: Name of the confidence column
        bins: Number of bins for histogram
    
    Returns:
        DataFrame with bin edges and counts
    """
    confidence = df[confidence_col].dropna()
    
    if len(confidence) == 0:
        return pd.DataFrame()
    
    hist, bin_edges = np.histogram(confidence, bins=bins, range=(0, 1))
    
    return pd.DataFrame({
        'bin_start': bin_edges[:-1],
        'bin_end': bin_edges[1:],
        'count': hist
    })
