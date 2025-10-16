"""
S1: Ingest & Canonicalize
Loads raw reviews and performs initial cleaning and validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging, parse_stars, parse_date

def ingest_and_canonicalize(config):
    """
    Pipeline Stage 1: Ingest & Canonicalize
    
    Steps:
    1. Load raw CSV
    2. Parse stars to numeric
    3. Parse dates
    4. Remove duplicates
    5. Validate data quality
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S1: INGEST & CANONICALIZE")
    logger.info("=" * 80)
    
    # Load configuration
    raw_csv = config['data']['raw_csv']
    output_path = Path(config['data']['processed_dir']) / '01_ingested.parquet'
    star_patterns = config['config']['stars']['regex_patterns']
    
    # Load raw data
    logger.info(f"Loading raw data from {raw_csv}")
    df = pd.read_csv(raw_csv)
    initial_count = len(df)
    logger.info(f"Loaded {initial_count:,} reviews")
    
    # Create copy of stars column
    df['stars_raw'] = df['stars'].copy()
    
    # Parse stars
    logger.info("Parsing star ratings...")
    df['stars_num'] = df['stars_raw'].apply(lambda x: parse_stars(x, star_patterns))
    
    # Clip to [1, 5]
    clip_range = config['config']['stars']['clip_range']
    df['stars_num'] = df['stars_num'].clip(clip_range[0], clip_range[1])
    
    # Parse dates
    logger.info("Parsing dates...")
    tqdm.pandas(desc="Parsing dates")
    df[['date_parsed', 'month_start']] = df['date'].progress_apply(
        lambda x: pd.Series(parse_date(x))
    )
    
    # Drop rows with invalid data
    before_drop = len(df)
    df = df.dropna(subset=['stars_num', 'month_start'])
    after_drop = len(df)
    logger.info(f"Dropped {before_drop - after_drop:,} rows with invalid stars/dates")
    
    # Remove exact duplicates
    logger.info("Removing duplicates...")
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['review_text', 'stars_raw', 'date'], keep='first')
    after_dedup = len(df)
    logger.info(f"Removed {before_dedup - after_dedup:,} duplicate reviews ({(before_dedup - after_dedup) / before_dedup * 100:.2f}%)")
    
    # Validation checks
    logger.info("Running validation checks...")
    
    # Check 1: >= 99% rows have valid stars_num
    valid_stars_pct = df['stars_num'].notna().mean() * 100
    logger.info(f"✓ Valid stars: {valid_stars_pct:.2f}% (target: >= 99%)")
    assert valid_stars_pct >= 99, f"Failed: Only {valid_stars_pct:.2f}% have valid stars"
    
    # Check 2: 100% rows have valid month_start
    valid_date_pct = df['month_start'].notna().mean() * 100
    logger.info(f"✓ Valid dates: {valid_date_pct:.2f}% (target: 100%)")
    assert valid_date_pct == 100, f"Failed: Only {valid_date_pct:.2f}% have valid dates"
    
    # Check 3: Duplicates removed
    logger.info(f"✓ Duplicates removed: {before_dedup - after_dedup:,} rows")
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Final count: {len(df):,} reviews")
    logger.info(f"  Date range: {df['date_parsed'].min()} to {df['date_parsed'].max()}")
    logger.info(f"  Star distribution:")
    for star in sorted(df['stars_num'].unique()):
        count = (df['stars_num'] == star).sum()
        pct = count / len(df) * 100
        logger.info(f"    {star:.1f} stars: {count:,} ({pct:.1f}%)")
    
    # Save processed data
    logger.info(f"\nSaving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info("✓ S1 completed successfully")
    return df

if __name__ == "__main__":
    config = load_config()
    df = ingest_and_canonicalize(config)
    print(f"\n✓ Stage 1 complete. Processed {len(df):,} reviews.")

