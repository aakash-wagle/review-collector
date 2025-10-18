"""
S1: Ingest, Drop Date, Language Scan, Canonicalize
"""
import pandas as pd
import numpy as np
import regex as re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from langdetect import detect, LangDetectException
from collections import Counter
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class StarParser:
    """Parse star ratings from text using regex patterns and word mappings."""
    
    def __init__(self, config: Dict):
        self.patterns = config['stars']['regex_patterns']
        self.word_to_num = config['stars']['word_to_num_map']
        self.normalize_scale = config['stars']['normalize_to_scale']
        self.clip_min, self.clip_max = config['stars']['clip_range']
    
    def parse(self, stars_raw: str) -> float:
        """Parse a star rating string to numeric value."""
        if pd.isna(stars_raw):
            return np.nan
        
        stars_str = str(stars_raw).strip().lower()
        
        # Try direct numeric conversion first
        try:
            value = float(stars_str)
            return self._normalize_and_clip(value, 5.0)
        except ValueError:
            pass
        
        # Pattern 1: "X out of Y" or "X/Y"
        for pattern in self.patterns[:2]:
            match = re.search(pattern, stars_str, re.IGNORECASE)
            if match:
                numerator = float(match.group(1))
                denominator = float(match.group(2))
                return self._normalize_and_clip(numerator, denominator)
        
        # Pattern 2: "rated X stars" or "rating X"
        match = re.search(self.patterns[2], stars_str, re.IGNORECASE)
        if match:
            value = float(match.group(2))
            return self._normalize_and_clip(value, 5.0)
        
        # Pattern 3: Word-based ratings (one star, two stars, etc.)
        match = re.search(self.patterns[3], stars_str, re.IGNORECASE)
        if match:
            word = match.group(1).lower()
            if word in self.word_to_num:
                value = self.word_to_num[word]
                return self._normalize_and_clip(value, 5.0)
        
        return np.nan
    
    def _normalize_and_clip(self, value: float, scale: float) -> float:
        """Normalize to target scale and clip to valid range."""
        normalized = (value / scale) * self.normalize_scale
        return np.clip(normalized, self.clip_min, self.clip_max)


class LanguageDetector:
    """Detect languages in text with fallback handling."""
    
    def detect_language(self, text: str) -> str:
        """Detect language, return 'unknown' on failure."""
        if pd.isna(text) or not str(text).strip():
            return 'unknown'
        
        try:
            return detect(str(text))
        except LangDetectException:
            return 'unknown'
    
    def scan_distribution(self, texts: pd.Series, sample_size: int = None) -> pd.DataFrame:
        """Scan language distribution in corpus."""
        if sample_size and len(texts) > sample_size:
            texts_sample = texts.sample(n=sample_size, random_state=42)
        else:
            texts_sample = texts
        
        logger.info(f"Scanning language distribution on {len(texts_sample)} texts...")
        
        languages = []
        for text in texts_sample:
            lang = self.detect_language(text)
            languages.append(lang)
        
        lang_counts = Counter(languages)
        
        df = pd.DataFrame([
            {'language': lang, 'count': count, 'percentage': count / len(languages) * 100}
            for lang, count in lang_counts.most_common()
        ])
        
        return df


def plot_language_distribution(lang_df: pd.DataFrame, output_path: Path):
    """Create language distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_langs = lang_df.head(15)  # Show top 15 languages
    
    ax.barh(top_langs['language'], top_langs['percentage'], color='steelblue')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Language', fontsize=12)
    ax.set_title('Language Distribution in Reviews', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (lang, pct) in enumerate(zip(top_langs['language'], top_langs['percentage'])):
        ax.text(pct + 0.5, i, f'{pct:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Language distribution plot saved to {output_path}")


def run_s1_ingest(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S1: Ingest pipeline
    - Load raw CSV
    - Drop 'date' column if present
    - Parse stars to numeric
    - Detect languages and create distribution
    - Remove duplicates
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    raw_csv = Path(config['data']['raw_csv'])
    output_parquet = Path(config['data']['processed_dir']) / '01_ingested.parquet'
    outputs_dir = Path(config['data']['outputs_dir'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading raw data from {raw_csv}...")
    
    # Load raw CSV
    df = pd.read_csv(raw_csv, encoding='utf-8')
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Drop 'date' column if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
        logger.info("Dropped 'date' column as per spec")
    
    # Ensure required columns
    required_cols = config['data']['raw_columns']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {list(df.columns)}")
    
    # Rename for consistency
    df = df.rename(columns={'review_text': 'review_text', 'stars': 'stars_raw'})
    
    # Parse stars
    logger.info("Parsing star ratings...")
    parser = StarParser(config['config'])
    df['stars_num'] = df['stars_raw'].apply(parser.parse)
    
    # Check validity
    valid_pct = df['stars_num'].notna().mean()
    logger.info(f"Valid star ratings: {valid_pct:.2%}")
    
    min_valid_pct = config['quality_gates']['min_valid_stars_pct']
    if valid_pct < min_valid_pct:
        raise ValueError(f"Only {valid_pct:.2%} rows have valid stars (threshold: {min_valid_pct:.2%})")
    
    # Language detection
    logger.info("Detecting languages...")
    lang_detector = LanguageDetector()
    sample_size = config['config']['translation']['auto_by_distribution'].get('sample_size', 5000)
    if sample_size == 'all':
        sample_size = None
    
    lang_df = lang_detector.scan_distribution(df['review_text'], sample_size=sample_size)
    
    # Save language distribution
    lang_table_path = outputs_dir / 'tables' / 'language_distribution.csv'
    lang_table_path.parent.mkdir(parents=True, exist_ok=True)
    lang_df.to_csv(lang_table_path, index=False)
    logger.info(f"Language distribution saved to {lang_table_path}")
    
    # Plot language distribution
    lang_fig_path = outputs_dir / 'figures' / 'language_distribution.png'
    lang_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_language_distribution(lang_df, lang_fig_path)
    
    # Detect language for all reviews
    df['lang'] = df['review_text'].apply(lang_detector.detect_language)
    
    # Remove exact duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['review_text', 'stars_raw'], keep='first')
    duplicates_removed = initial_count - len(df)
    dup_pct = duplicates_removed / initial_count * 100
    logger.info(f"Removed {duplicates_removed} duplicates ({dup_pct:.2f}%)")
    
    # Save processed data
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved ingested data to {output_parquet}")
    
    # Summary stats
    logger.info(f"\nS1 Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  Valid stars: {df['stars_num'].notna().sum()} ({df['stars_num'].notna().mean():.2%})")
    logger.info(f"  Star distribution: {df['stars_num'].value_counts().sort_index().to_dict()}")
    logger.info(f"  Language distribution (top 5):")
    for _, row in lang_df.head().iterrows():
        logger.info(f"    {row['language']}: {row['percentage']:.2f}%")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s1_ingest()
    print(f"\nIngestion complete. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

