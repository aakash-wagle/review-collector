"""
S5: Model Baselines (SiEBERT Overall + PyABSA Aspects)
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SiEBERTSentiment:
    """SiEBERT transformer-based sentiment analysis."""
    
    def __init__(self, model_name: str, label_mapping: Dict):
        self.model_name = model_name
        self.label_mapping = label_mapping
        self.pipe = None
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"SiEBERT device: {'cuda' if self.device == 0 else 'cpu'}")
    
    def load_model(self):
        """Lazy load model."""
        if self.pipe is None:
            logger.info(f"Loading SiEBERT model: {self.model_name}...")
            self.pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device
            )
            logger.info("SiEBERT model loaded successfully")
    
    def predict(self, texts: List[str], batch_size: int = 128) -> List[Dict]:
        """Predict sentiment for batch of texts using efficient GPU batching."""
        self.load_model()
        
        # Handle empty texts
        texts_processed = [text if text and text.strip() else "neutral" for text in texts]
        
        # Use pipeline with built-in batching for GPU efficiency
        # This processes the entire dataset in optimized batches
        try:
            logger.info(f"Processing {len(texts_processed)} texts with batch_size={batch_size}")
            predictions = list(self.pipe(
                texts_processed,
                batch_size=batch_size,  # Let pipeline handle batching
                truncation=True,
                max_length=512
            ))
            return predictions
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            # Fallback to default sentiment
            return [{'label': 'NEUTRAL', 'score': 1.0}] * len(texts)
    
    def process_results(self, predictions: List[Dict]) -> pd.DataFrame:
        """Process predictions to DataFrame."""
        df_results = pd.DataFrame(predictions)
        df_results['sentiment_numeric'] = df_results['label'].map(self.label_mapping)
        return df_results


class PyABSAAspectExtractor:
    """PyABSA-based aspect sentiment extraction."""
    
    def __init__(self, backend_model: str, aspect_catalog: List[str]):
        self.backend_model = backend_model
        self.aspect_catalog = aspect_catalog
        self.model = None
    
    def load_model(self):
        """Load PyABSA model."""
        if self.model is None:
            logger.info("Loading PyABSA model...")
            try:
                from pyabsa import ATEPCCheckpointManager
                import torch
                
                # Determine device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"PyABSA will use device: {device}")
                
                # Use a pretrained checkpoint with explicit device
                self.model = ATEPCCheckpointManager.get_aspect_extractor(
                    checkpoint='english',  # Use pretrained English model
                    auto_device=device  # Explicitly set device
                )
                logger.info(f"PyABSA model loaded successfully on {device}")
            except Exception as e:
                logger.error(f"Failed to load PyABSA: {e}")
                logger.warning("PyABSA will use fallback mode")
                self.model = None
    
    def extract_aspects(self, texts: List[str]) -> List[List[Dict]]:
        """Extract aspects and sentiments from texts."""
        self.load_model()
        
        if self.model is None:
            logger.warning("PyABSA model not available, using fallback heuristic")
            return self._fallback_extraction(texts)
        
        results = []
        batch_size = 128  # Process 128 texts at once on GPU
        
        # Process in batches for GPU efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="PyABSA extraction (batches)"):
            batch_texts = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch_texts:
                if not text or not text.strip():
                    batch_results.append([])
                    continue
                
                try:
                    prediction = self.model.predict(
                        text,
                        pred_sentiment=True,
                        print_result=False,
                        eval_batch_size=128,  # Internal batch size for PyABSA
                        ignore_error=True
                    )
                    
                    # Parse results
                    aspects = []
                    if prediction and hasattr(prediction, 'aspect'):
                        for aspect, sentiment, confidence in zip(
                            prediction.aspect,
                            prediction.sentiment,
                            prediction.confidence
                        ):
                            aspects.append({
                                'aspect': aspect,
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'evidence_span': aspect  # The aspect term itself
                            })
                    
                    batch_results.append(aspects)
                    
                except Exception as e:
                    batch_results.append([])
            
            results.extend(batch_results)
        
        return results
    
    def _fallback_extraction(self, texts: List[str]) -> List[List[Dict]]:
        """Fallback aspect extraction using dictionary matching + VADER."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        analyzer = SentimentIntensityAnalyzer()
        
        # Simple keyword-based extraction
        keywords = {
            'Sound/Audio': ['sound', 'audio', 'bass', 'clarity', 'volume'],
            'Comfort/Fit': ['comfort', 'fit', 'tight', 'hurt', 'ear'],
            'Battery/Longevity': ['battery', 'charge', 'life', 'hours'],
            'Connectivity/Bluetooth': ['bluetooth', 'connect', 'wireless', 'pair'],
            'Build/Durability': ['build', 'quality', 'durable', 'break'],
        }
        
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append([])
                continue
            
            text_lower = text.lower()
            aspects = []
            
            for aspect, terms in keywords.items():
                for term in terms:
                    if term in text_lower:
                        # Extract sentence containing term
                        sentences = text.split('.')
                        for sent in sentences:
                            if term in sent.lower():
                                sentiment_scores = analyzer.polarity_scores(sent)
                                sentiment = 'positive' if sentiment_scores['compound'] > 0.05 else \
                                           'negative' if sentiment_scores['compound'] < -0.05 else 'neutral'
                                
                                aspects.append({
                                    'aspect': aspect,
                                    'sentiment': sentiment,
                                    'confidence': abs(sentiment_scores['compound']),
                                    'evidence_span': sent.strip()
                                })
                                break
                        break
            
            results.append(aspects)
        
        return results


def aggregate_aspect_sentiments(aspect_extractions: List[List[Dict]]) -> pd.DataFrame:
    """Aggregate aspect sentiments per review."""
    aggregated = []
    
    for aspects in aspect_extractions:
        if not aspects:
            aggregated.append({})
            continue
        
        # Count sentiments per aspect
        aspect_dict = {}
        for asp in aspects:
            aspect_name = asp['aspect']
            sentiment = asp['sentiment']
            
            if aspect_name not in aspect_dict:
                aspect_dict[aspect_name] = {'positive': 0, 'negative': 0, 'neutral': 0, 'count': 0}
            
            aspect_dict[aspect_name][sentiment] += 1
            aspect_dict[aspect_name]['count'] += 1
        
        aggregated.append(aspect_dict)
    
    return aggregated


def run_s5_models(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S5: Model Baselines Pipeline
    - SiEBERT overall sentiment
    - PyABSA aspect extraction and sentiment
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load feature-engineered data (from S3, not S4)
    input_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} reviews for model processing")
    
    # === SiEBERT Overall Sentiment ===
    if config['config']['sentiment']['transformer_overall']['enabled']:
        logger.info("Running SiEBERT sentiment analysis...")
        
        model_name = config['config']['sentiment']['transformer_overall']['model']
        label_mapping = config['config']['sentiment']['transformer_overall']['label_mapping']
        
        siebert = SiEBERTSentiment(model_name, label_mapping)
        
        # Prepare texts
        texts = df['text_en_clean'].fillna('').tolist()
        
        # Predict with GPU-optimized batch size
        predictions = siebert.predict(texts, batch_size=256)
        
        # Process results
        sentiment_df = siebert.process_results(predictions)
        df['siebert_label'] = sentiment_df['label']
        df['siebert_score'] = sentiment_df['score']
        df['siebert_sentiment'] = sentiment_df['sentiment_numeric']
        
        logger.info(f"SiEBERT sentiment distribution: {df['siebert_label'].value_counts().to_dict()}")
    
    # === PyABSA Aspect Extraction ===
    if config['config']['aspects']['pyabsa']['enabled']:
        logger.info("Running PyABSA aspect extraction...")
        
        aspect_catalog = config['config']['aspects']['target_catalog']
        backend_model = config['config']['aspects']['pyabsa']['backend_model']
        
        pyabsa = PyABSAAspectExtractor(backend_model, aspect_catalog)
        
        # Extract aspects
        texts = df['text_en_clean'].fillna('').tolist()
        aspect_extractions = pyabsa.extract_aspects(texts)
        
        # Store raw extractions
        df['aspects_pyabsa_raw'] = aspect_extractions
        
        # Aggregate
        df['aspects_pyabsa_agg'] = aggregate_aspect_sentiments(aspect_extractions)
        
        # Count total aspects found
        total_aspects = sum(len(aspects) for aspects in aspect_extractions)
        logger.info(f"PyABSA extracted {total_aspects} aspect mentions")
    
    # Save processed data
    output_path = Path(config['data']['processed_dir']) / '05_models.parquet'
    
    # Convert complex columns to strings for parquet compatibility
    if 'aspects_pyabsa_raw' in df.columns:
        df['aspects_pyabsa_raw'] = df['aspects_pyabsa_raw'].apply(lambda x: str(x) if x else None)
    if 'aspects_pyabsa_agg' in df.columns:
        df['aspects_pyabsa_agg'] = df['aspects_pyabsa_agg'].apply(lambda x: str(x) if x else None)
    
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved model results to {output_path}")
    
    # Summary
    logger.info(f"\nS5 Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    if 'siebert_label' in df.columns:
        logger.info(f"  SiEBERT sentiment: {df['siebert_label'].value_counts().to_dict()}")
    if 'aspects_pyabsa_raw' in df.columns:
        logger.info(f"  PyABSA aspects extracted: {total_aspects} mentions")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s5_models()
    print(f"\nModel baselines complete. Shape: {df.shape}")

