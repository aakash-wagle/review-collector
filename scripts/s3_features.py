"""
S3: Text Normalization & Feature Engineering
Cleans text and extracts features for analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging, clean_text, compute_text_features

def lemmatize_text(text, nlp=None):
    """Lemmatize text using spaCy."""
    if nlp is None:
        return text
    
    try:
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    except:
        return text

def text_normalization_and_features(config):
    """
    Pipeline Stage 3: Text Normalization & Feature Engineering
    
    Steps:
    1. Clean and normalize text
    2. Lemmatization (optional)
    3. Compute text features
    4. Flag low-info reviews
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S3: TEXT NORMALIZATION & FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Load input data
    input_path = Path(config['data']['processed_dir']) / '02_translated.parquet'
    output_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} reviews")
    
    # Load spaCy model for lemmatization
    nlp = None
    if config['config']['normalization']['lemmatize']:
        try:
            import spacy
            logger.info("Loading spaCy model...")
            try:
                nlp = spacy.load('en_core_web_sm')
            except:
                logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                logger.warning("Skipping lemmatization...")
        except ImportError:
            logger.warning("spaCy not installed. Skipping lemmatization...")
    
    # Clean text
    logger.info("Cleaning text...")
    tqdm.pandas(desc="Text cleaning")
    df['text_en_clean'] = df['text_en'].progress_apply(lambda x: clean_text(x, config))
    
    # Lemmatization
    if nlp is not None:
        logger.info("Lemmatizing text...")
        tqdm.pandas(desc="Lemmatization")
        df['text_en_lemma'] = df['text_en_clean'].progress_apply(
            lambda x: lemmatize_text(x, nlp)
        )
    else:
        df['text_en_lemma'] = df['text_en_clean'].copy()
    
    # Compute text features
    logger.info("Computing text features...")
    tqdm.pandas(desc="Feature extraction")
    features_df = df['text_en_clean'].progress_apply(compute_text_features).apply(pd.Series)
    df = pd.concat([df, features_df], axis=1)
    
    # Flag low-information reviews
    threshold = config['config']['normalization']['low_info_word_threshold']
    df['low_info_flag'] = df['word_len'] < threshold
    low_info_count = df['low_info_flag'].sum()
    logger.info(f"Low-information reviews (< {threshold} words): {low_info_count:,} ({low_info_count / len(df) * 100:.1f}%)")
    
    # Validation
    logger.info("\nValidation:")
    empty_clean = (df['text_en_clean'].str.len() == 0).sum()
    logger.info(f"✓ Empty cleaned text: {empty_clean} (target: 0)")
    assert empty_clean == 0, f"Found {empty_clean} empty cleaned texts"
    
    # Feature statistics
    logger.info("\nFeature Statistics:")
    for col in ['word_len', 'char_len', 'exclaim_count', 'uppercase_ratio']:
        logger.info(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, variance={df[col].var():.2f}")
        # Note: uppercase_ratio may have zero variance if all text is lowercased
        if col not in ['uppercase_ratio'] and df[col].var() == 0:
            logger.warning(f"Warning: Feature {col} has zero variance")
    
    # Save processed data
    logger.info(f"\nSaving to {output_path}")
    df.to_parquet(output_path, index=False)
    
    logger.info("✓ S3 completed successfully")
    return df

if __name__ == "__main__":
    config = load_config()
    df = text_normalization_and_features(config)
    print(f"\n✓ Stage 3 complete. Processed {len(df):,} reviews.")

