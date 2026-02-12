"""
Parallel processor for analyzing OCR parquet files from S3
"""
import pandas as pd
import s3fs
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import logging

from config import S3_BUCKET, S3_GB_PATH_TEMPLATE, S3_GV_PATH_TEMPLATE
from stats_calculator import compute_google_books_stats, compute_google_vision_stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
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
    Process volumes in batches to manage memory
    
    Args:
        volumes: List of volume info dicts
        processor_func: Function to process each volume
        batch_size: Number of volumes per batch
        max_workers: Maximum number of parallel workers
        desc: Description for progress bar
    
    Returns:
        DataFrame with all results
    """
    all_results = []
    
    # Process in batches
    for i in range(0, len(volumes), batch_size):
        batch = volumes[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(volumes) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} volumes)")
        
        batch_results = process_volumes_parallel(
            batch, 
            processor_func, 
            max_workers=max_workers,
            desc=f"{desc} (batch {batch_num}/{total_batches})"
        )
        
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
