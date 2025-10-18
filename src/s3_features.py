"""
S3: Text Normalization & Feature Engineering
"""
import pandas as pd
import numpy as np
import yaml
import logging
import re
from pathlib import Path
import emoji
import spacy
from typing import Dict

logger = logging.getLogger(__name__)


class TextNormalizer:
    """Normalize and clean text."""
    
    def __init__(self, config: Dict):
        self.config = config['normalization']
        self.nlp = None
    
    def load_spacy(self, model_name: str = "en_core_web_sm"):
        """Lazy load spaCy model."""
        if self.nlp is None:
            logger.info(f"Loading spaCy model: {model_name}...")
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                logger.info(f"Downloading spaCy model: {model_name}...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", model_name])
                self.nlp = spacy.load(model_name)
    
    def clean(self, text: str) -> str:
        """Apply cleaning transformations."""
        if pd.isna(text) or not str(text).strip():
            return ""
        
        text_clean = str(text)
        
        # Lowercase
        if self.config['lowercase']:
            text_clean = text_clean.lower()
        
        # Demojize (convert emojis to text descriptions)
        if self.config['demojize']:
            text_clean = emoji.demojize(text_clean, delimiters=(" ", " "))
        
        # Strip URLs, HTML tags, handles
        if self.config['strip_urls_html_handles']:
            # Remove URLs
            text_clean = re.sub(r'http\S+|www\.\S+', '', text_clean)
            # Remove HTML tags
            text_clean = re.sub(r'<.*?>', '', text_clean)
            # Remove @mentions
            text_clean = re.sub(r'@\w+', '', text_clean)
        
        # Expand contractions
        if self.config['expand_contractions']:
            text_clean = self._expand_contractions(text_clean)
        
        # Clean up whitespace
        text_clean = ' '.join(text_clean.split())
        
        return text_clean
    
    def lemmatize(self, text: str, model_name: str = "en_core_web_sm") -> str:
        """Lemmatize text using spaCy."""
        if not text or not text.strip():
            return ""
        
        # Lazy load spaCy
        self.load_spacy(model_name)
        
        doc = self.nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        return ' '.join(lemmas)
    
    @staticmethod
    def _expand_contractions(text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'s": " is",
            r"'d": " would",
            r"'ll": " will",
            r"'t": " not",
            r"'ve": " have",
            r"'m": " am"
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class FeatureEngineer:
    """Extract text features."""
    
    @staticmethod
    def word_count(text: str) -> int:
        """Count words in text."""
        if pd.isna(text) or not str(text).strip():
            return 0
        return len(str(text).split())
    
    @staticmethod
    def char_count(text: str) -> int:
        """Count characters in text."""
        if pd.isna(text) or not str(text).strip():
            return 0
        return len(str(text))
    
    @staticmethod
    def exclamation_count(text: str) -> int:
        """Count exclamation marks."""
        if pd.isna(text):
            return 0
        return str(text).count('!')
    
    @staticmethod
    def uppercase_ratio(text: str) -> float:
        """Calculate ratio of uppercase characters."""
        if pd.isna(text) or not str(text).strip():
            return 0.0
        
        text_str = str(text)
        letters = [c for c in text_str if c.isalpha()]
        if not letters:
            return 0.0
        
        uppercase_count = sum(1 for c in letters if c.isupper())
        return uppercase_count / len(letters)
    
    @staticmethod
    def is_low_info(text: str, threshold: int = 5) -> bool:
        """Flag low-information texts (very short)."""
        word_count = len(str(text).split()) if not pd.isna(text) else 0
        return word_count < threshold


def run_s3_features(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S3: Feature Engineering Pipeline
    - Clean and normalize text
    - Lemmatize text
    - Extract text features
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load translated data
    input_path = Path(config['data']['processed_dir']) / '02_translated.parquet'
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} reviews for feature engineering")
    
    # Initialize normalizer
    normalizer = TextNormalizer(config['config'])
    engineer = FeatureEngineer()
    
    # Clean text
    logger.info("Cleaning and normalizing text...")
    df['text_en_clean'] = df['text_en'].apply(normalizer.clean)
    
    # Lemmatize if enabled
    if config['config']['normalization']['lemmatize']:
        logger.info("Lemmatizing text (this may take a while)...")
        spacy_model = config['environment']['models']['spacy_model']
        
        # Process in batches for efficiency
        from tqdm import tqdm
        tqdm.pandas(desc="Lemmatizing")
        df['text_en_lemma'] = df['text_en_clean'].progress_apply(
            lambda x: normalizer.lemmatize(x, spacy_model)
        )
    else:
        df['text_en_lemma'] = df['text_en_clean']
    
    # Extract features
    logger.info("Extracting text features...")
    df['word_len'] = df['text_en_clean'].apply(engineer.word_count)
    df['char_len'] = df['text_en_clean'].apply(engineer.char_count)
    df['exclaim_count'] = df['text_en'].apply(engineer.exclamation_count)
    df['uppercase_ratio'] = df['text_en'].apply(engineer.uppercase_ratio)
    
    threshold = config['config']['normalization']['low_info_word_threshold']
    df['low_info'] = df['text_en_clean'].apply(lambda x: engineer.is_low_info(x, threshold))
    
    # Validation: no empty cleaned texts
    empty_count = (df['text_en_clean'].str.strip() == '').sum()
    if empty_count > 0:
        logger.warning(f"Found {empty_count} empty cleaned texts")
        # Fill with original or mark for removal
        df.loc[df['text_en_clean'].str.strip() == '', 'text_en_clean'] = df['text_en']
    
    # Check feature variance
    logger.info("Feature variance check:")
    for col in ['word_len', 'char_len', 'exclaim_count', 'uppercase_ratio']:
        variance = df[col].var()
        logger.info(f"  {col}: variance = {variance:.4f}")
    
    # Save processed data
    output_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved feature-engineered data to {output_path}")
    
    # Summary
    logger.info(f"\nS3 Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  Avg word count: {df['word_len'].mean():.1f}")
    logger.info(f"  Avg char count: {df['char_len'].mean():.1f}")
    logger.info(f"  Low-info reviews: {df['low_info'].sum()} ({df['low_info'].mean():.2%})")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s3_features()
    print(f"\nFeature engineering complete. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

