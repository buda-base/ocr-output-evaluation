"""
Configuration for OCR output analysis
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
PGSQL_URL = os.getenv('PGSQL_URL')

# S3 configuration
S3_BUCKET = 'bec.bdrc.io'
S3_GB_PREFIX = 'google_books'
S3_GV_PREFIX = 'gv'

# S3 path templates
S3_GB_PATH_TEMPLATE = 's3://{bucket}/google_books/{w_id}/{i_id}/{i_version}/{w_id}_{i_id}_{i_version}_gb.parquet'
S3_GV_PATH_TEMPLATE = 's3://{bucket}/gv/{w_id}/{i_id}/{i_version}/{w_id}-{i_id}-{i_version}-gv.parquet'

# Output configuration
OUTPUT_DIR = 'output'
GB_STATS_OUTPUT = os.path.join(OUTPUT_DIR, 'google_books_stats.parquet')
GV_STATS_OUTPUT = os.path.join(OUTPUT_DIR, 'google_vision_stats.parquet')
SUMMARY_CSV = os.path.join(OUTPUT_DIR, 'summary_stats.csv')

# Processing configuration
MAX_WORKERS = min(8, os.cpu_count() or 4)  # Limit parallel workers for memory efficiency
BATCH_SIZE = 100  # Process volumes in batches to manage memory
