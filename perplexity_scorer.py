"""
KenLM-based perplexity scoring for OCR text quality evaluation
"""
import kenlm
import sentencepiece as spm
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger(__name__)

# Global models cache - loaded once per process
_kenlm_model = None
_sp_model = None


def load_models() -> Tuple[kenlm.Model, spm.SentencePieceProcessor]:
    """
    Load KenLM and SentencePiece models from Hugging Face Hub.
    Models are cached globally to avoid reloading in each process.
    
    Returns:
        Tuple of (kenlm_model, sp_model)
    """
    global _kenlm_model, _sp_model
    
    if _kenlm_model is not None and _sp_model is not None:
        return _kenlm_model, _sp_model
    
    logger.info("Downloading perplexity models from Hugging Face Hub...")
    
    arpa_path = hf_hub_download(
        repo_id="openpecha/BoKenlm", 
        filename="lm.arpa"
    )
    sp_model_path = hf_hub_download(
        repo_id="openpecha/BoSentencePiece", 
        filename="sentencepiece.model"
    )
    
    logger.info("Loading perplexity models into memory...")
    _kenlm_model = kenlm.Model(arpa_path)
    _sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
    
    logger.info("Perplexity models loaded successfully")
    return _kenlm_model, _sp_model


def calculate_perplexity(text: str, kenlm_model: kenlm.Model, sp_model: spm.SentencePieceProcessor) -> float:
    """
    Calculate perplexity of text using KenLM and SentencePiece.
    
    Perplexity is computed as: 10^(-log10_score / token_count)
    Lower perplexity indicates better quality text.
    
    Args:
        text: Input text to score
        kenlm_model: Loaded KenLM model
        sp_model: Loaded SentencePiece model
    
    Returns:
        Perplexity score (float), or inf if text is empty/invalid
    """
    if not text or not text.strip():
        return float('inf')
    
    log_score = 0.0
    token_count = 0
    
    # Process line by line to handle multiline text
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Tokenize with SentencePiece - encode_as_pieces returns list directly
        tokens = sp_model.encode_as_pieces(line)
        
        if not tokens:
            continue
        
        # Join tokens as space-separated string for KenLM
        # This is required by KenLM's API - no way to avoid this step
        tokens_str = " ".join(tokens)
        
        # Get log10 probability from KenLM
        log_score += kenlm_model.score(tokens_str, bos=True, eos=True)
        token_count += len(tokens) + 1  # +1 for </s> end token
    
    if token_count == 0:
        return float('inf')
    
    # Calculate perplexity: 10^(-log_score / token_count)
    perplexity = 10.0 ** (-log_score / token_count)
    return perplexity


def calculate_perplexity_batch(texts: pd.Series, 
                                kenlm_model: Optional[kenlm.Model] = None,
                                sp_model: Optional[spm.SentencePieceProcessor] = None) -> pd.Series:
    """
    Calculate perplexity for a batch of texts efficiently.
    
    Args:
        texts: Pandas Series of text strings
        kenlm_model: Optional pre-loaded KenLM model (loads if None)
        sp_model: Optional pre-loaded SentencePiece model (loads if None)
    
    Returns:
        Pandas Series of perplexity scores
    """
    if kenlm_model is None or sp_model is None:
        kenlm_model, sp_model = load_models()
    
    # Vectorized calculation - process each text
    # Using apply is necessary here as we need per-text tokenization
    perplexities = texts.apply(
        lambda text: calculate_perplexity(text, kenlm_model, sp_model)
    )
    
    return perplexities


def compute_perplexity_stats(perplexities: pd.Series) -> dict:
    """
    Compute statistics for a series of perplexity scores.
    
    Args:
        perplexities: Series of perplexity scores
    
    Returns:
        Dictionary of statistics
    """
    # Remove infinite values for statistics
    valid_perplexities = perplexities.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_perplexities) == 0:
        return {
            'mean_perplexity': float('inf'),
            'median_perplexity': float('inf'),
            'std_perplexity': 0.0,
            'min_perplexity': float('inf'),
            'max_perplexity': float('inf'),
            'p10_perplexity': float('inf'),
            'p25_perplexity': float('inf'),
            'p75_perplexity': float('inf'),
            'p90_perplexity': float('inf'),
            'p95_perplexity': float('inf'),
            'pages_with_valid_perplexity': 0,
        }
    
    return {
        'mean_perplexity': float(valid_perplexities.mean()),
        'median_perplexity': float(valid_perplexities.median()),
        'std_perplexity': float(valid_perplexities.std()),
        'min_perplexity': float(valid_perplexities.min()),
        'max_perplexity': float(valid_perplexities.max()),
        'p10_perplexity': float(valid_perplexities.quantile(0.10)),
        'p25_perplexity': float(valid_perplexities.quantile(0.25)),
        'p75_perplexity': float(valid_perplexities.quantile(0.75)),
        'p90_perplexity': float(valid_perplexities.quantile(0.90)),
        'p95_perplexity': float(valid_perplexities.quantile(0.95)),
        'pages_with_valid_perplexity': len(valid_perplexities),
    }
