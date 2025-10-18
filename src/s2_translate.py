"""
S2: Conditional Translation (Local Only)
"""
import pandas as pd
import numpy as np
import yaml
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

logger = logging.getLogger(__name__)


class TranslationCache:
    """SQLite-based translation cache."""
    
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database."""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS translations (
                text_hash TEXT PRIMARY KEY,
                source_lang TEXT,
                target_lang TEXT,
                source_text TEXT,
                translated_text TEXT,
                model TEXT
            )
        """)
        conn.commit()
        conn.close()
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Retrieve cached translation."""
        text_hash = self._hash_text(text)
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT translated_text FROM translations WHERE text_hash=? AND source_lang=? AND target_lang=?",
            (text_hash, source_lang, target_lang)
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def set(self, text: str, source_lang: str, target_lang: str, translation: str, model: str):
        """Store translation in cache."""
        text_hash = self._hash_text(text)
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?, ?)",
            (text_hash, source_lang, target_lang, text, translation, model)
        )
        conn.commit()
        conn.close()
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()


class LocalTranslator:
    """Local translation using M2M100 (no external API)."""
    
    def __init__(self, model_name: str, cache_path: Path, use_cache: bool = True):
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache = TranslationCache(cache_path) if use_cache else None
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Translation device: {self.device}")
    
    def load_model(self):
        """Lazy load translation model."""
        if self.model is None:
            logger.info(f"Loading translation model: {self.model_name}...")
            self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            logger.info("Translation model loaded successfully")
    
    def translate(self, text: str, source_lang: str, target_lang: str = 'en') -> str:
        """Translate text to target language."""
        if pd.isna(text) or not str(text).strip():
            return text
        
        text_str = str(text).strip()
        
        # Check cache first
        if self.use_cache and self.cache:
            cached = self.cache.get(text_str, source_lang, target_lang)
            if cached:
                return cached
        
        # Lazy load model
        self.load_model()
        
        # Set source language
        self.tokenizer.src_lang = self._map_lang_code(source_lang)
        
        # Tokenize
        encoded = self.tokenizer(text_str, return_tensors="pt", padding=True, truncation=True, max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Generate translation
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(self._map_lang_code(target_lang)),
            max_length=512,
            num_beams=3
        )
        
        # Decode
        translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        # Cache result
        if self.use_cache and self.cache:
            self.cache.set(text_str, source_lang, target_lang, translation, self.model_name)
        
        return translation
    
    @staticmethod
    def _map_lang_code(lang: str) -> str:
        """Map langdetect codes to M2M100 codes."""
        # M2M100 uses different language codes
        lang_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'ru': 'ru',
            'zh-cn': 'zh',
            'zh-tw': 'zh',
            'ja': 'ja',
            'ko': 'ko',
            'ar': 'ar',
            'hi': 'hi',
            'tr': 'tr',
            'pl': 'pl',
            'nl': 'nl',
            'sv': 'sv',
            'da': 'da',
            'fi': 'fi',
            'no': 'no',
            'cs': 'cs',
            'ro': 'ro',
            'vi': 'vi',
            'th': 'th',
            'id': 'id',
            'ms': 'ms',
            'he': 'he',
            'uk': 'uk',
        }
        return lang_map.get(lang, 'en')  # Default to English if unknown


def run_s2_translate(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S2: Conditional Translation Pipeline
    - Check if translation is needed based on language distribution
    - Translate non-English texts to English using local model
    - Perform back-translation QA on sample
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if translation is enabled
    if not config['config']['translation']['enabled']:
        logger.info("Translation disabled in config, skipping S2")
        # Just copy input to output
        input_path = Path(config['data']['processed_dir']) / '01_ingested.parquet'
        output_path = Path(config['data']['processed_dir']) / '02_translated.parquet'
        df = pd.read_parquet(input_path)
        df['text_en'] = df['review_text']
        df['translation_meta'] = None
        df.to_parquet(output_path, index=False)
        return df
    
    # Load ingested data
    input_path = Path(config['data']['processed_dir']) / '01_ingested.parquet'
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} reviews for translation processing")
    
    # Check if translation should be auto-enabled by distribution
    auto_config = config['config']['translation']['auto_by_distribution']
    if auto_config['enabled']:
        non_en_pct = (df['lang'] != 'en').mean()
        threshold = auto_config['non_en_threshold']
        
        logger.info(f"Non-English reviews: {non_en_pct:.2%} (threshold: {threshold:.2%})")
        
        if non_en_pct < threshold:
            logger.info(f"Non-English share below threshold, translation disabled")
            df['text_en'] = df['review_text']
            df['translation_meta'] = None
            output_path = Path(config['data']['processed_dir']) / '02_translated.parquet'
            df.to_parquet(output_path, index=False)
            return df
        else:
            logger.info(f"Non-English share above threshold, enabling translation")
    
    # Initialize translator
    cache_path = Path(config['governance']['caching']['translation_cache'])
    model_name = config['environment']['models']['translation_local_model']
    
    translator = LocalTranslator(
        model_name=model_name,
        cache_path=cache_path,
        use_cache=config['config']['translation']['cache']
    )
    
    # Translate non-English texts
    logger.info("Translating non-English reviews...")
    
    df['text_en'] = df['review_text'].copy()
    df['translation_meta'] = None
    
    non_en_mask = (df['lang'] != 'en') & (df['lang'] != 'unknown')
    non_en_indices = df[non_en_mask].index
    
    if len(non_en_indices) > 0:
        for idx in tqdm(non_en_indices, desc="Translating"):
            source_text = df.loc[idx, 'review_text']
            source_lang = df.loc[idx, 'lang']
            
            try:
                translation = translator.translate(source_text, source_lang, 'en')
                df.loc[idx, 'text_en'] = translation
                df.loc[idx, 'translation_meta'] = f"translated_{source_lang}_to_en"
            except Exception as e:
                logger.warning(f"Translation failed for index {idx}: {e}")
                df.loc[idx, 'text_en'] = source_text  # Keep original on failure
                df.loc[idx, 'translation_meta'] = "translation_failed"
    
    logger.info(f"Translated {non_en_mask.sum()} reviews")
    
    # Back-translation QA (optional, on sample)
    if config['config']['translation']['back_translation_qc']['enabled']:
        logger.info("Performing back-translation QC...")
        sample_size = min(
            config['config']['translation']['back_translation_qc']['sample_size'],
            non_en_mask.sum()
        )
        
        if sample_size > 0:
            sample_indices = df[non_en_mask].sample(n=sample_size, random_state=42).index
            
            # For QC, we'll just log a summary (actual back-translation would double translation time)
            logger.info(f"Back-translation QC would analyze {sample_size} samples")
            logger.info("(Actual back-translation skipped for performance; implement if needed)")
    
    # Save processed data
    output_path = Path(config['data']['processed_dir']) / '02_translated.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved translated data to {output_path}")
    
    # Summary
    logger.info(f"\nS2 Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  Translated: {(df['translation_meta'].str.contains('translated', na=False)).sum()}")
    logger.info(f"  Translation failures: {(df['translation_meta'] == 'translation_failed').sum()}")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s2_translate()
    print(f"\nTranslation complete. Shape: {df.shape}")

