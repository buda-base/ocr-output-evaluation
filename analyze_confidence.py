"""
Main script to analyze OCR output confidence statistics
"""
import os
import argparse
import logging
from pathlib import Path

from config import (
    OUTPUT_DIR, GB_STATS_OUTPUT, GV_STATS_OUTPUT, OCRV1_STATS_OUTPUT, SUMMARY_CSV,
    MAX_WORKERS, BATCH_SIZE
)
from db_queries import get_all_volumes
from parallel_processor import process_volumes_batch, process_google_books_volume, process_google_vision_volume, process_ocrv1_volume

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Analyze OCR output confidence statistics')
    parser.add_argument('--ocr-type', choices=['google_books', 'google_vision', 'ocrv1', 'all'], 
                       default='all', help='Which OCR type to process')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS,
                       help=f'Maximum number of parallel workers (default: {MAX_WORKERS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for processing (default: {BATCH_SIZE})')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of volumes to process (for testing)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--enable-perplexity', action='store_true', default=None,
                       help='Enable perplexity scoring (slower but provides quality metrics)')
    parser.add_argument('--disable-perplexity', action='store_true', default=False,
                       help='Disable perplexity scoring')
    parser.add_argument('--force', action='store_true', default=False,
                       help='Force reprocessing even if output file exists')
    
    args = parser.parse_args()
    
    # Handle perplexity flags
    if args.enable_perplexity:
        os.environ['ENABLE_PERPLEXITY'] = 'true'
        logger.info("Perplexity scoring ENABLED")
    elif args.disable_perplexity:
        os.environ['ENABLE_PERPLEXITY'] = 'false'
        logger.info("Perplexity scoring DISABLED")
    else:
        # Use default from config
        from config import ENABLE_PERPLEXITY
        logger.info(f"Perplexity scoring: {'ENABLED' if ENABLE_PERPLEXITY else 'DISABLED'} (default)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Fetching volume list from database...")
    gb_volumes, gv_volumes, ocrv1_volumes = get_all_volumes()
    
    logger.info(f"Found {len(gb_volumes)} Google Books volumes")
    logger.info(f"Found {len(gv_volumes)} Google Vision volumes")
    logger.info(f"Found {len(ocrv1_volumes)} OCRv1-WS-LDv1 volumes")
    
    # Apply limit if specified
    if args.limit:
        gb_volumes = gb_volumes[:args.limit]
        gv_volumes = gv_volumes[:args.limit]
        ocrv1_volumes = ocrv1_volumes[:args.limit]
        logger.info(f"Limited to {args.limit} volumes per OCR type")
    
    # Process Google Books
    if args.ocr_type in ['google_books', 'all'] and gb_volumes:
        gb_output_path = os.path.join(args.output_dir, 'google_books_stats.parquet')
        
        if os.path.exists(gb_output_path) and not args.force:
            logger.info(f"\n{'='*80}")
            logger.info("Google Books output already exists (skipping)")
            logger.info(f"File: {gb_output_path}")
            logger.info(f"Use --force to reprocess")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info("Processing Google Books volumes")
            logger.info(f"{'='*80}")
            
            gb_stats_df = process_volumes_batch(
                gb_volumes,
                process_google_books_volume,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                desc="Google Books"
            )
            
            if not gb_stats_df.empty:
                gb_stats_df.to_parquet(gb_output_path, index=False)
                logger.info(f"Google Books statistics saved to: {gb_output_path}")
                
                # Also save CSV for quick viewing
                gb_csv_path = os.path.join(args.output_dir, 'google_books_stats.csv')
                gb_stats_df.to_csv(gb_csv_path, index=False)
                logger.info(f"Google Books CSV saved to: {gb_csv_path}")
                
                # Print summary statistics
                logger.info("\nGoogle Books Summary:")
                logger.info(f"  Volumes processed: {len(gb_stats_df)}")
                logger.info(f"  Total pages: {gb_stats_df['total_pages'].sum():,.0f}")
                logger.info(f"  Mean confidence: {gb_stats_df['mean_confidence'].mean():.3f}")
                logger.info(f"  Median confidence: {gb_stats_df['median_confidence'].median():.3f}")
    
    # Process Google Vision
    if args.ocr_type in ['google_vision', 'all'] and gv_volumes:
        gv_output_path = os.path.join(args.output_dir, 'google_vision_stats.parquet')
        
        if os.path.exists(gv_output_path) and not args.force:
            logger.info(f"\n{'='*80}")
            logger.info("Google Vision output already exists (skipping)")
            logger.info(f"File: {gv_output_path}")
            logger.info(f"Use --force to reprocess")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info("Processing Google Vision volumes")
            logger.info(f"{'='*80}")
            
            gv_stats_df = process_volumes_batch(
                gv_volumes,
                process_google_vision_volume,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                desc="Google Vision"
            )
            
            if not gv_stats_df.empty:
                gv_stats_df.to_parquet(gv_output_path, index=False)
                logger.info(f"Google Vision statistics saved to: {gv_output_path}")
                
                # Also save CSV for quick viewing
                gv_csv_path = os.path.join(args.output_dir, 'google_vision_stats.csv')
                gv_stats_df.to_csv(gv_csv_path, index=False)
                logger.info(f"Google Vision CSV saved to: {gv_csv_path}")
                
                # Print summary statistics
                logger.info("\nGoogle Vision Summary:")
                logger.info(f"  Volumes processed: {len(gv_stats_df)}")
                logger.info(f"  Total pages: {gv_stats_df['total_records'].sum():,.0f}")
                logger.info(f"  Mean confidence: {gv_stats_df['mean_confidence'].mean():.3f}")
                logger.info(f"  Median confidence: {gv_stats_df['median_confidence'].median():.3f}")
    
    # Process OCRv1-WS-LDv1
    if args.ocr_type in ['ocrv1', 'all'] and ocrv1_volumes:
        ocrv1_output_path = os.path.join(args.output_dir, 'ocrv1_ws_ldv1_stats.parquet')
        
        if os.path.exists(ocrv1_output_path) and not args.force:
            logger.info(f"\n{'='*80}")
            logger.info("OCRv1-WS-LDv1 output already exists (skipping)")
            logger.info(f"File: {ocrv1_output_path}")
            logger.info(f"Use --force to reprocess")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info("Processing OCRv1-WS-LDv1 volumes")
            logger.info(f"{'='*80}")
            
            ocrv1_stats_df = process_volumes_batch(
                ocrv1_volumes,
                process_ocrv1_volume,
                batch_size=args.batch_size,
                max_workers=args.max_workers,
                desc="OCRv1-WS-LDv1"
            )
            
            if not ocrv1_stats_df.empty:
                ocrv1_stats_df.to_parquet(ocrv1_output_path, index=False)
                logger.info(f"OCRv1-WS-LDv1 statistics saved to: {ocrv1_output_path}")
                
                # Also save CSV for quick viewing
                ocrv1_csv_path = os.path.join(args.output_dir, 'ocrv1_ws_ldv1_stats.csv')
                ocrv1_stats_df.to_csv(ocrv1_csv_path, index=False)
                logger.info(f"OCRv1-WS-LDv1 CSV saved to: {ocrv1_csv_path}")
                
                # Print summary statistics
                logger.info("\nOCRv1-WS-LDv1 Summary:")
                logger.info(f"  Volumes processed: {len(ocrv1_stats_df)}")
                logger.info(f"  Total pages: {ocrv1_stats_df['total_records'].sum():,.0f}")
                if 'mean_perplexity' in ocrv1_stats_df.columns:
                    import numpy as np
                    valid_perp = ocrv1_stats_df['mean_perplexity'].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid_perp) > 0:
                        logger.info(f"  Mean perplexity: {valid_perp.mean():.2f}")
                        logger.info(f"  Median perplexity: {valid_perp.median():.2f}")

    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
