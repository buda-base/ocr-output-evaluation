
#!/usr/bin/env python3
"""
Select and download challenging images for manual evaluation.
Criteria: "Low range but not the lowest" (avoiding garbage/empty, focusing on hard-but-possible).
"""
import pandas as pd
import numpy as np
import os
import requests
import logging
import random
from pathlib import Path
from tqdm import tqdm
import time

from metadata_loader import load_all_stats_with_metadata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "challenging_images"
TARGET_COUNT = 500
VOLUMES_TO_SAMPLE = 60  # Sample ~8-9 images per volume
IIIF_BASE_URL = "https://iiif.bdrc.io/bdr:{i_id}::{filename}/full/max/0/default.jpg"

def download_image(url, save_path):
    """Download image from IIIF"""
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            logger.warning(f"Failed to download {url}: Status {response.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Error downloading {url}: {e}")
        return False

def select_volumes(df):
    """
    Select volumes that fit the 'challenging but not garbage' criteria.
    """
    # Filter 1: Google OCR with confidence between 0.6 and 0.85
    # (Assuming normalized 0-1 scale)
    google_candidates = pd.DataFrame()
    if 'mean_confidence' in df.columns:
        # Ensure confidence is 0-1
        conf = df['mean_confidence'].copy()
        if conf.max() > 1.0:
            conf = conf / 100.0
        
        mask = (conf >= 0.60) & (conf <= 0.85)
        google_candidates = df[mask].copy()
        logger.info(f"Found {len(google_candidates)} Google OCR volumes in confidence range 0.60-0.85")

    # Filter 2: OCRv1 with perplexity between 3000 and 8000
    ocrv1_candidates = pd.DataFrame()
    if 'mean_perplexity' in df.columns:
        # Filter out inf
        perp = df['mean_perplexity'].replace([np.inf, -np.inf], np.nan)
        mask = (perp >= 3000) & (perp <= 8000)
        ocrv1_candidates = df[mask].copy()
        logger.info(f"Found {len(ocrv1_candidates)} OCRv1 volumes in perplexity range 3000-8000")

    # Combine and sample
    candidates = pd.concat([google_candidates, ocrv1_candidates]).drop_duplicates(subset=['s3_path'])
    
    if len(candidates) == 0:
        logger.error("No volumes found matching criteria!")
        return pd.DataFrame()

    # Sample random volumes
    if len(candidates) > VOLUMES_TO_SAMPLE:
        selected = candidates.sample(n=VOLUMES_TO_SAMPLE, random_state=42)
    else:
        selected = candidates
        
    logger.info(f"Selected {len(selected)} volumes for sampling")
    return selected

def process_volume_and_select_images(row):
    """
    Read volume parquet and select specific pages
    """
    s3_path = row['s3_path']
    w_id = row['w_id']
    i_id = row['i_id']
    
    try:
        # Read parquet file (supports S3 if env vars are set, otherwise assumes local mount or accessible)
        # Note: This relies on the environment having S3 access configured
        df = pd.read_parquet(s3_path)
        
        candidates = []
        
        # Criteria for selecting pages within the volume
        if 'confidence' in df.columns:
            # Google Books/Vision: Pick pages with low confidence
            # Normalize if needed
            conf = df['confidence'].copy()
            if conf.max() > 1.0:
                conf = conf / 100.0
            
            # Filter for pages that are challenging (0.5 - 0.8)
            # Avoid < 0.4 as they might be empty/garbage
            mask = (conf >= 0.5) & (conf <= 0.85)
            page_candidates = df[mask]
            
            if len(page_candidates) < 5:
                # Relax criteria if not enough
                mask = (conf >= 0.4) & (conf <= 0.9)
                page_candidates = df[mask]
                
        else:
            # OCRv1: No confidence column.
            # Filter for pages that have text (not empty)
            if 'line_texts' in df.columns:
                # Check if it has text
                has_text = df['line_texts'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)
                page_candidates = df[has_text]
            elif 'text' in df.columns:
                page_candidates = df[df['text'].str.len() > 10]
            else:
                page_candidates = df
        
        # Sample pages from this volume
        if len(page_candidates) > 0:
            # Take up to 10 pages per volume
            n_sample = min(10, len(page_candidates))
            sampled_pages = page_candidates.sample(n=n_sample)
            
            for _, page in sampled_pages.iterrows():
                filename = page.get('img_file_name') or page.get('filename')
                if filename:
                    candidates.append({
                        'w_id': w_id,
                        'i_id': i_id,
                        'filename': filename,
                        's3_path': s3_path
                    })
                    
        return candidates

    except Exception as e:
        logger.warning(f"Error processing volume {s3_path}: {e}")
        return []

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Stats
    print("Loading statistics...")
    df = load_all_stats_with_metadata('output', 'data')
    
    if df.empty:
        print("No stats data found.")
        return

    # 2. Select Volumes
    selected_volumes = select_volumes(df)
    if selected_volumes.empty:
        return

    # 3. Select Images (Drill down)
    print("Drilling down to select images from volumes...")
    all_image_candidates = []
    
    for _, row in tqdm(selected_volumes.iterrows(), total=len(selected_volumes)):
        images = process_volume_and_select_images(row)
        if images:
            all_image_candidates.extend(images)
            
    print(f"Found {len(all_image_candidates)} candidate images.")
    
    # 4. Final Selection
    if len(all_image_candidates) > TARGET_COUNT:
        final_selection = random.sample(all_image_candidates, TARGET_COUNT)
    else:
        final_selection = all_image_candidates
        
    print(f"Downloading {len(final_selection)} images to {OUTPUT_DIR}/...")
    
    # 5. Download
    metadata_records = []
    
    for item in tqdm(final_selection):
        w_id = item['w_id']
        i_id = item['i_id']
        filename = item['filename']
        
        # Construct IIIF URL
        url = IIIF_BASE_URL.format(i_id=i_id, filename=filename)
        
        # Construct local filename
        save_name = f"{w_id}_{i_id}_{filename}"
        if not save_name.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):
            save_name += ".jpg"
            
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        if not os.path.exists(save_path):
            success = download_image(url, save_path)
            if success:
                metadata_records.append(item)
                time.sleep(0.1) # Be nice to the server
        else:
            metadata_records.append(item)
            
    # Save manifest
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.csv")
    pd.DataFrame(metadata_records).to_csv(manifest_path, index=False)
    print(f"Done! Images and manifest saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
