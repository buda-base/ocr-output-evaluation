"""
Parallel processor for analyzing OCR parquet files from S3
"""
import pandas as pd
import s3fs
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import logging
import os

from config import S3_BUCKET, S3_GB_PATH_TEMPLATE, S3_GV_PATH_TEMPLATE, ENABLE_PERPLEXITY
from stats_calculator import compute_google_books_stats, compute_google_vision_stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _init_worker_process():
    """
    Initialize worker process by pre-loading perplexity models.
    This is called once per worker process when it starts.
    """
    # Check if perplexity is enabled
    enable_perplexity = os.getenv('ENABLE_PERPLEXITY', 'true').lower() in ('true', '1', 'yes')
    
    if not enable_perplexity:
        return
    
    try:
        # Import and load models once per worker
        import perplexity_scorer
        # Load models with local_files_only after first download
        # This prevents HTTP requests on every call
        perplexity_scorer.load_models(local_files_only=True)
        logger.info(f"Worker process {os.getpid()} initialized with perplexity models")
    except Exception as e:
        # If local files don't exist yet, try downloading
        try:
            import perplexity_scorer
            perplexity_scorer.load_models(local_files_only=False)
            logger.info(f"Worker process {os.getpid()} downloaded and initialized perplexity models")
        except Exception as e2:
            logger.warning(f"Worker process {os.getpid()} failed to load perplexity models: {e2}")
            # Continue without perplexity


def process_google_books_volume(volume_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Process a single Google Books volume and compute statistics
    
    Args:
        volume_info: Dict with w_id, i_id, i_version, volume_id
    
    Returns:
        Dictionary with volume info and computed statistics, or None if error
    """
    w_id = volume_info['w_id']
    i_id = volume_info['i_id']
    i_version = volume_info['i_version']
    volume_id = volume_info['volume_id']
    
    s3_path = S3_GB_PATH_TEMPLATE.format(
        bucket=S3_BUCKET,
        w_id=w_id,
        i_id=i_id,
        i_version=i_version
    )
    
    try:
        # Read parquet file from S3
        df = pd.read_parquet(s3_path)
        
        # Compute statistics
        stats = compute_google_books_stats(df)
        
        # Add volume identifiers
        result = {
            'w_id': w_id,
            'i_id': i_id,
            'i_version': i_version,
            'volume_id': volume_id,
            's3_path': s3_path,
            **stats
        }
        
        return result
        
    except FileNotFoundError:
        logger.warning(f"File not found: {s3_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing {s3_path}: {str(e)}")
        return None


def process_google_vision_volume(volume_info: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    Process a single Google Vision volume and compute statistics
    
    Args:
        volume_info: Dict with w_id, i_id, i_version, volume_id
    
    Returns:
        Dictionary with volume info and computed statistics, or None if error
    """
    w_id = volume_info['w_id']
    i_id = volume_info['i_id']
    i_version = volume_info['i_version']
    volume_id = volume_info['volume_id']
    
    s3_path = S3_GV_PATH_TEMPLATE.format(
        bucket=S3_BUCKET,
        w_id=w_id,
        i_id=i_id,
        i_version=i_version
    )
    
    try:
        # Read parquet file from S3
        df = pd.read_parquet(s3_path)
        
        # Compute statistics
        stats = compute_google_vision_stats(df)
        
        # Add volume identifiers
        result = {
            'w_id': w_id,
            'i_id': i_id,
            'i_version': i_version,
            'volume_id': volume_id,
            's3_path': s3_path,
            **stats
        }
        
        return result
        
    except FileNotFoundError:
        logger.warning(f"File not found: {s3_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing {s3_path}: {str(e)}")
        return None


def process_volumes_parallel(volumes: List[Dict[str, str]], 
                            processor_func,
                            max_workers: int = 4,
                            desc: str = "Processing volumes") -> List[Dict[str, Any]]:
    """
    Process volumes in parallel using multiprocessing
    
    Args:
        volumes: List of volume info dicts
        processor_func: Function to process each volume
        max_workers: Maximum number of parallel workers
        desc: Description for progress bar
    
    Returns:
        List of results (statistics for each volume)
    """
    results = []
    
    # Use initializer to load models once per worker process
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
        # Submit all tasks
        future_to_volume = {
            executor.submit(processor_func, vol): vol 
            for vol in volumes
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(volumes), desc=desc) as pbar:
            for future in as_completed(future_to_volume):
                result = future.result()
                if result is not None:
                    results.append(result)
                pbar.update(1)
    
    return results


def process_volumes_batch(volumes: List[Dict[str, str]], 
                          processor_func,
                          batch_size: int = 100,
                          max_workers: int = 4,
                          desc: str = "Processing volumes") -> pd.DataFrame:
    """
    Process volumes in batches to manage memory.
    Creates worker pool ONCE and reuses it for all batches.
    
    Args:
        volumes: List of volume info dicts
        processor_func: Function to process each volume
        batch_size: Number of volumes per batch (for memory management)
        max_workers: Maximum number of parallel workers
        desc: Description for progress bar
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    total_batches = (len(volumes) + batch_size - 1) // batch_size
    
    logger.info(f"Processing {len(volumes)} volumes in {total_batches} batches with {max_workers} workers")
    logger.info("Initializing worker pool (models will load once per worker)...")
    
    # Create ProcessPoolExecutor ONCE for all batches
    # Workers are initialized once and reused across all batches
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker_process) as executor:
        # Process in batches
        for i in range(0, len(volumes), batch_size):
            batch = volumes[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Submitting batch {batch_num}/{total_batches} ({len(batch)} volumes)")
            
            # Submit all tasks in this batch to the EXISTING worker pool
            future_to_volume = {
                executor.submit(processor_func, vol): vol 
                for vol in batch
            }
            
            # Process completed tasks with progress bar
            batch_results = []
            with tqdm(total=len(batch), desc=f"{desc} (batch {batch_num}/{total_batches})") as pbar:
                for future in as_completed(future_to_volume):
                    result = future.result()
                    if result is not None:
                        batch_results.append(result)
                    pbar.update(1)
            
            all_results.extend(batch_results)
            logger.info(f"Batch {batch_num} complete. Processed {len(batch_results)}/{len(batch)} volumes successfully")
    
    # Convert to DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        logger.info(f"Total volumes processed successfully: {len(df)}/{len(volumes)}")
        return df
    else:
        logger.warning("No results to return")
        return pd.DataFrame()
