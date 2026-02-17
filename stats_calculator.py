"""
Statistics computation functions for OCR confidence analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from config import ENABLE_PERPLEXITY

logger = logging.getLogger(__name__)

# Lazy import for perplexity - only load when needed
_perplexity_module = None
_perplexity_models = None


def _get_perplexity_scorer():
    """
    Lazy load perplexity module and models.
    Models should already be loaded by worker initializer.
    """
    global _perplexity_module, _perplexity_models
    
    if not ENABLE_PERPLEXITY:
        return None, None, None
    
    if _perplexity_module is None:
        try:
            import perplexity_scorer
            _perplexity_module = perplexity_scorer
        except ImportError as e:
            logger.warning(f"Could not import perplexity_scorer: {e}")
            return None, None, None
    
    # Check if models are already loaded (by worker initializer)
    if _perplexity_models is None:
        try:
            # Try to load with local_files_only first (should already be cached)
            _perplexity_models = _perplexity_module.load_models(local_files_only=True)
        except Exception as e:
            # Fallback: download if needed (first time only)
            try:
                _perplexity_models = _perplexity_module.load_models(local_files_only=False)
            except Exception as e2:
                logger.error(f"Failed to load perplexity models: {e2}")
                return _perplexity_module, None, None
    
    kenlm_model, sp_model = _perplexity_models
    return _perplexity_module, kenlm_model, sp_model


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
    
    # Convert to float64 to prevent overflow, then remove null/nan/inf values
    # Source data uses float16 which can overflow in numpy operations
    confidence = df[confidence_col].astype('float64').replace([np.inf, -np.inf], np.nan).dropna()
    
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
    # Convert numeric columns to float64 to prevent overflow
    nb_lines = df['nb_lines'].astype('float64') if 'nb_lines' in df.columns else pd.Series([])
    text_lengths = df['text'].str.len().astype('float64') if 'text' in df.columns else pd.Series([])
    
    stats.update({
        'total_records': len(df),
        'successful_pages': int(df['ok'].sum()) if 'ok' in df.columns else 0,
        'failed_pages': int((~df['ok']).sum()) if 'ok' in df.columns else 0,
        'total_lines': int(nb_lines.sum()) if len(nb_lines) > 0 else 0,
        'mean_lines_per_page': float(nb_lines.mean()) if len(nb_lines) > 0 else 0.0,
        'total_text_length': int(text_lengths.sum()) if len(text_lengths) > 0 else 0,
        'mean_text_length_per_page': float(text_lengths.mean()) if len(text_lengths) > 0 else 0.0,
        'pages_with_lines': int((nb_lines > 0).sum()) if len(nb_lines) > 0 else 0,
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
    
    # Add perplexity statistics if text column is available
    if 'text' in df.columns:
        perplexity_module, kenlm_model, sp_model = _get_perplexity_scorer()
        if perplexity_module and kenlm_model and sp_model:
            try:
                # Calculate perplexity for all pages
                perplexities = perplexity_module.calculate_perplexity_batch(
                    df['text'].fillna(''), 
                    kenlm_model, 
                    sp_model
                )
                
                # Compute perplexity statistics
                perplexity_stats = perplexity_module.compute_perplexity_stats(perplexities)
                stats.update(perplexity_stats)
            except Exception as e:
                logger.error(f"Error calculating perplexity: {e}")
    
    return stats


def compute_ocrv1_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics for OCRv1-WS-LDv1 format
    
    Args:
        df: DataFrame with OCRv1 schema (line_texts field is list of strings)
    
    Returns:
        Dictionary with computed statistics
    """
    stats = {}
    
    # Log what columns we actually have
    logger.debug(f"OCRv1 DataFrame columns: {df.columns.tolist()}")
    logger.debug(f"OCRv1 DataFrame shape: {df.shape}")
    
    # For OCRv1, the field is called 'line_texts' (not 'texts')
    # It's a list of strings - one per line - that should be concatenated
    if 'line_texts' in df.columns:
        # Convert list of strings to single text per page
        df = df.copy()
        df['text'] = df['line_texts'].apply(lambda x: '\n'.join(x) if isinstance(x, list) else '')
        logger.debug(f"Created 'text' column from 'line_texts' field")
    else:
        logger.warning(f"OCRv1 DataFrame missing 'line_texts' column. Columns: {df.columns.tolist()}")
    
    if df.empty or 'text' not in df.columns:
        logger.warning(f"OCRv1: Empty DataFrame or missing 'text' column. Returning empty stats.")
        return stats
    
    # Compute perplexity statistics if text is available
    if 'text' in df.columns:
        perplexity_module, kenlm_model, sp_model = _get_perplexity_scorer()
        if perplexity_module and kenlm_model and sp_model:
            try:
                # Calculate perplexity for all pages
                perplexities = perplexity_module.calculate_perplexity_batch(
                    df['text'].fillna(''), 
                    kenlm_model, 
                    sp_model
                )
                
                # Compute perplexity statistics
                perplexity_stats = perplexity_module.compute_perplexity_stats(perplexities)
                stats.update(perplexity_stats)
            except Exception as e:
                logger.error(f"Error calculating perplexity: {e}")
    
    # Add basic metrics
    text_lengths = df['text'].str.len().astype('float64') if 'text' in df.columns else pd.Series([])
    nb_lines = df['nb_lines'].astype('float64') if 'nb_lines' in df.columns else pd.Series([])
    
    stats.update({
        'total_records': len(df),
        'total_text_length': int(text_lengths.sum()) if len(text_lengths) > 0 else 0,
        'mean_text_length_per_page': float(text_lengths.mean()) if len(text_lengths) > 0 else 0.0,
        'total_lines': int(nb_lines.sum()) if len(nb_lines) > 0 else 0,
        'mean_lines_per_page': float(nb_lines.mean()) if len(nb_lines) > 0 else 0.0,
        'pages_with_lines': int((nb_lines > 0).sum()) if len(nb_lines) > 0 else 0,
    })
    
    # Count pages with text
    if 'text' in df.columns:
        pages_with_text = (df['text'].str.len() > 0).sum()
        stats['pages_with_text'] = int(pages_with_text)
    
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
    # Convert numeric columns to float64 to prevent overflow
    nb_lines_tib = df['nb_lines_tib'].astype('float64') if 'nb_lines_tib' in df.columns else pd.Series([])
    text_len = df['text_len'].astype('float64') if 'text_len' in df.columns else pd.Series([])
    
    stats.update({
        'total_records': len(df),
        'total_tibetan_lines': int(nb_lines_tib.sum()) if len(nb_lines_tib) > 0 else 0,
        'mean_tibetan_lines_per_page': float(nb_lines_tib.mean()) if len(nb_lines_tib) > 0 else 0.0,
        'total_text_length': int(text_len.sum()) if len(text_len) > 0 else 0,
        'mean_text_length_per_page': float(text_len.mean()) if len(text_len) > 0 else 0.0,
        'pages_with_tibetan_lines': int((nb_lines_tib > 0).sum()) if len(nb_lines_tib) > 0 else 0,
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
    
    # Add perplexity statistics if text column is available
    if 'text' in df.columns:
        perplexity_module, kenlm_model, sp_model = _get_perplexity_scorer()
        if perplexity_module and kenlm_model and sp_model:
            try:
                # Calculate perplexity for all pages
                perplexities = perplexity_module.calculate_perplexity_batch(
                    df['text'].fillna(''), 
                    kenlm_model, 
                    sp_model
                )
                
                # Compute perplexity statistics
                perplexity_stats = perplexity_module.compute_perplexity_stats(perplexities)
                stats.update(perplexity_stats)
            except Exception as e:
                logger.error(f"Error calculating perplexity: {e}")
    
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
