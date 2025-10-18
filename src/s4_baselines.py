"""
S4: Deterministic Baselines (EDA, VADER, Topics, Dict Aspects)
"""
import pandas as pd
import numpy as np
import yaml
import logging
import re
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

logger = logging.getLogger(__name__)


class VADERSentiment:
    """VADER sentiment analysis."""
    
    def __init__(self, config: Dict):
        self.analyzer = SentimentIntensityAnalyzer()
        self.thresholds = config['sentiment']['vader']['thresholds']
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment and return scores + label."""
        if pd.isna(text) or not str(text).strip():
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'label': 'neutral'}
        
        scores = self.analyzer.polarity_scores(str(text))
        
        # Label based on compound score
        compound = scores['compound']
        if compound >= self.thresholds['pos']:
            label = 'positive'
        elif compound <= self.thresholds['neg']:
            label = 'negative'
        else:
            label = 'neutral'
        
        scores['label'] = label
        return scores


class TopicModeler:
    """Topic modeling using TF-IDF + KMeans."""
    
    def __init__(self, config: Dict):
        self.config = config['topics']
        self.vectorizer = None
        self.model = None
        self.best_k = None
        self.tfidf_matrix = None
    
    def fit(self, texts: pd.Series) -> Dict:
        """Fit topic model and select best k."""
        # TF-IDF vectorization
        logger.info("Vectorizing texts with TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(self.config['tfidf']['ngram_range']),
            min_df=self.config['tfidf']['min_df'],
            max_features=self.config['tfidf']['max_features'],
            stop_words='english'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Find best k using silhouette score
        k_min = self.config['kmeans']['k_min']
        k_max = self.config['kmeans']['k_max']
        
        logger.info(f"Finding best k in range [{k_min}, {k_max}]...")
        silhouette_scores = {}
        
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.tfidf_matrix)
            score = silhouette_score(self.tfidf_matrix, labels, sample_size=5000)
            silhouette_scores[k] = score
            logger.info(f"  k={k}: silhouette={score:.4f}")
        
        # Select best k
        self.best_k = max(silhouette_scores, key=silhouette_scores.get)
        logger.info(f"Best k selected: {self.best_k} (silhouette={silhouette_scores[self.best_k]:.4f})")
        
        # Fit final model
        self.model = KMeans(n_clusters=self.best_k, random_state=42, n_init=10)
        labels = self.model.fit_predict(self.tfidf_matrix)
        
        return {
            'best_k': self.best_k,
            'silhouette_scores': silhouette_scores,
            'best_silhouette': silhouette_scores[self.best_k],
            'labels': labels
        }
    
    def get_top_terms(self, cluster_id: int, n_terms: int = 10) -> List[str]:
        """Get top terms for a cluster."""
        center = self.model.cluster_centers_[cluster_id]
        terms = self.vectorizer.get_feature_names_out()
        top_indices = center.argsort()[-n_terms:][::-1]
        return [terms[i] for i in top_indices]
    
    def generate_labels(self, df: pd.DataFrame = None) -> Dict[int, str]:
        """Generate semantic labels from top terms and context."""
        labels = {}
        
        for cluster_id in range(self.best_k):
            top_terms = self.get_top_terms(cluster_id, n_terms=20)
            
            # Get avg stars for this cluster if available
            avg_stars = None
            if df is not None:
                cluster_mask = df['topic_cluster'] == cluster_id
                if cluster_mask.sum() > 0:
                    avg_stars = df.loc[cluster_mask, 'stars_num'].mean()
            
            # Generate semantic label based on keywords
            label = self._interpret_topic(top_terms, avg_stars)
            labels[cluster_id] = label
        
        return labels
    
    def _interpret_topic(self, terms: List[str], avg_stars: float = None) -> str:
        """Interpret topic from terms using semantic rules."""
        terms_str = ' '.join(terms).lower()
        
        # Problem indicators
        if any(word in terms_str for word in ['stop work', 'break', 'broken', 'month']):
            if avg_stars and avg_stars < 3.0:
                return "Product Failures/Breaking"
        
        if any(word in terms_str for word in ['hurt', 'ear hurt', 'tight', 'uncomfortable', 'pain']):
            return "Comfort Issues (Ear Pain/Fit)"
        
        # Positive aspects
        if 'battery life' in terms_str or ('battery' in terms_str and 'life' in terms_str):
            return "Battery Life Satisfaction"
        
        if any(word in terms_str for word in ['amazing sound', 'great sound', 'excellent']):
            return "Excellent Sound Quality"
        
        if 'noise cancel' in terms_str or 'noise cancellation' in terms_str:
            return "Noise Cancellation"
        
        if any(word in terms_str for word in ['christmas', 'gift', 'daughter', 'present']):
            return "Gift Purchases"
        
        if any(word in terms_str for word in ['color', 'rose gold', 'design', 'look']):
            return "Color & Design"
        
        # General categories
        if 'sound quality' in terms_str or 'sound' in terms[:3]:
            if avg_stars and avg_stars >= 4.5:
                return "Sound Quality Praise"
            else:
                return "Sound Quality Discussion"
        
        if 'charge' in terms_str and 'long' in terms_str:
            return "Battery Charging & Life"
        
        if 'improvement' in terms_str or 'point' in terms_str:
            return "Areas for Improvement"
        
        # Default: use top 3 non-bigram terms
        single_words = [t for t in terms if ' ' not in t][:3]
        return f"{' '.join(single_words).title()}"


class AspectDictionary:
    """Dictionary-based aspect extraction."""
    
    def __init__(self, config: Dict):
        self.catalog = config['aspects']['target_catalog']
        self.keywords = self._build_keyword_dict()
    
    def _build_keyword_dict(self) -> Dict[str, List[str]]:
        """Build keyword dictionary for each aspect."""
        keywords = {
            'Sound/Audio': [
                r'\bsound\b', r'\baudio\b', r'\bbass\b', r'\btreble\b', 
                r'\bclarity\b', r'\bclear\b', r'\bquality\b', r'\bvolume\b',
                r'\bloud\b', r'\bmusic\b', r'\bacoustic\b'
            ],
            'Comfort/Fit': [
                r'\bcomfort\b', r'\bfit\b', r'\btight\b', r'\bhurt\b', 
                r'\bear\b', r'\bhead\b', r'\bpadding\b', r'\bsore\b',
                r'\buncomfortable\b', r'\bpressure\b', r'\bpainful\b'
            ],
            'Battery/Longevity': [
                r'\bbattery\b', r'\bcharge\b', r'\bcharging\b', r'\blife\b',
                r'\bhours?\b', r'\blast\b', r'\bpower\b', r'\blongevity\b'
            ],
            'Connectivity/Bluetooth': [
                r'\bbluetooth\b', r'\bconnect\b', r'\bpair\b', r'\bconnection\b',
                r'\bwireless\b', r'\bw1\b', r'\bchip\b', r'\blatency\b'
            ],
            'Build/Durability': [
                r'\bbuild\b', r'\bquality\b', r'\bdurable\b', r'\bbreak\b',
                r'\bbroken\b', r'\bsturdy\b', r'\bmetal\b', r'\bplastic\b',
                r'\bfragile\b', r'\bconstruction\b'
            ],
            'Controls/Buttons': [
                r'\bbutton\b', r'\bcontrol\b', r'\bvolume\b', r'\bplay\b',
                r'\bpause\b', r'\bskip\b', r'\btrack\b'
            ],
            'Mic/Calls': [
                r'\bmic\b', r'\bmicrophone\b', r'\bcall\b', r'\bphone\b',
                r'\bvoice\b', r'\btalk\b', r'\bhear\b', r'\bspeak\b'
            ],
            'Price/Value': [
                r'\bprice\b', r'\bvalue\b', r'\bcost\b', r'\bcheap\b',
                r'\bexpensive\b', r'\bworth\b', r'\bdeal\b', r'\bsale\b',
                r'\bdollar\b', r'\bmoney\b'
            ],
            'Portability/Foldability': [
                r'\bportable\b', r'\bfold\b', r'\bcompact\b', r'\btravel\b',
                r'\bcarry\b', r'\bbackpack\b'
            ],
            'Charging/Port': [
                r'\bcharging\b', r'\bport\b', r'\busb\b', r'\bcable\b',
                r'\bcord\b', r'\bcharger\b'
            ],
            'Noise Isolation': [
                r'\bnoise\b', r'\bcancell?ation\b', r'\bisolation\b',
                r'\bblock\b', r'\bcancel\b', r'\bleaking\b', r'\bleak\b'
            ],
            'Accessories/Case': [
                r'\bcase\b', r'\baccessor\b', r'\bincluded\b', r'\bpackage\b',
                r'\bbox\b'
            ]
        }
        return keywords
    
    def extract_aspects(self, text: str) -> List[str]:
        """Extract aspects mentioned in text."""
        if pd.isna(text) or not str(text).strip():
            return []
        
        text_lower = str(text).lower()
        aspects = []
        
        for aspect, patterns in self.keywords.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    aspects.append(aspect)
                    break  # Found one keyword for this aspect
        
        if not aspects:
            aspects.append('Other')
        
        return aspects


def plot_ratings_histogram(df: pd.DataFrame, output_path: Path):
    """Plot star ratings distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    star_counts = df['stars_num'].value_counts().sort_index()
    ax.bar(star_counts.index, star_counts.values, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Star Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Star Ratings', fontsize=14, fontweight='bold')
    ax.set_xticks([1, 2, 3, 4, 5])
    
    # Add value labels on bars
    for x, y in zip(star_counts.index, star_counts.values):
        ax.text(x, y + max(star_counts.values) * 0.01, str(y), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Ratings histogram saved to {output_path}")


def plot_sentiment_confusion(vader_labels, stars, output_path: Path):
    """Plot confusion between VADER sentiment and stars."""
    # Convert stars to sentiment labels
    star_sentiment = pd.cut(stars, bins=[0, 2, 3, 5], labels=['negative', 'neutral', 'positive'])
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(star_sentiment, vader_labels, labels=['negative', 'neutral', 'positive'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'],
                ax=ax)
    
    ax.set_xlabel('VADER Sentiment', fontsize=12)
    ax.set_ylabel('Star-based Sentiment', fontsize=12)
    ax.set_title('VADER vs Stars Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sentiment confusion plot saved to {output_path}")


def run_s4_baselines(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S4: Baselines Pipeline
    - EDA visualizations
    - VADER sentiment
    - Topic modeling with heuristic labels
    - Dictionary-based aspect extraction
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load feature-engineered data
    input_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} reviews for baseline processing")
    
    outputs_dir = Path(config['data']['outputs_dir'])
    
    # === EDA ===
    logger.info("Generating EDA visualizations...")
    
    # Ratings histogram
    fig_path = outputs_dir / 'figures' / 'ratings_histogram.png'
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plot_ratings_histogram(df, fig_path)
    
    # === VADER Sentiment ===
    logger.info("Running VADER sentiment analysis...")
    vader = VADERSentiment(config['config'])
    
    vader_results = df['text_en_clean'].apply(vader.analyze)
    df['vader_compound'] = vader_results.apply(lambda x: x['compound'])
    df['vader_pos'] = vader_results.apply(lambda x: x['pos'])
    df['vader_neu'] = vader_results.apply(lambda x: x['neu'])
    df['vader_neg'] = vader_results.apply(lambda x: x['neg'])
    df['vader_label'] = vader_results.apply(lambda x: x['label'])
    
    # Confusion matrix
    fig_path = outputs_dir / 'figures' / 'sentiment_confusion_vader_vs_stars.png'
    plot_sentiment_confusion(df['vader_label'], df['stars_num'], fig_path)
    
    # Agreement metrics
    star_sentiment = pd.cut(df['stars_num'], bins=[0, 2, 3, 5], labels=['negative', 'neutral', 'positive'])
    agreement = (df['vader_label'] == star_sentiment).mean()
    logger.info(f"VADER-Stars agreement: {agreement:.2%}")
    
    # === Topic Modeling ===
    logger.info("Running topic modeling...")
    topic_modeler = TopicModeler(config['config'])
    
    # Filter out low-info texts
    texts_for_topics = df[~df['low_info']]['text_en_lemma']
    
    topic_results = topic_modeler.fit(texts_for_topics)
    
    # Assign topics to all reviews
    df['topic_cluster'] = -1
    df.loc[~df['low_info'], 'topic_cluster'] = topic_results['labels']
    
    # Generate semantic topic labels (with star ratings for context)
    topic_labels = topic_modeler.generate_labels(df)
    df['topic_label'] = df['topic_cluster'].map(lambda x: topic_labels.get(x, 'Unknown'))
    
    # Save topic metadata
    topic_stats = []
    for cluster_id in range(topic_results['best_k']):
        cluster_mask = df['topic_cluster'] == cluster_id
        topic_stats.append({
            'cluster_id': cluster_id,
            'label': topic_labels[cluster_id],
            'count': cluster_mask.sum(),
            'avg_stars': df.loc[cluster_mask, 'stars_num'].mean(),
            'top_terms': ', '.join(topic_modeler.get_top_terms(cluster_id, n_terms=10))
        })
    
    topic_df = pd.DataFrame(topic_stats)
    topic_table_path = outputs_dir / 'tables' / 'topics_summary.csv'
    topic_table_path.parent.mkdir(parents=True, exist_ok=True)
    topic_df.to_csv(topic_table_path, index=False)
    logger.info(f"Topic summary saved to {topic_table_path}")
    
    # === Dictionary-based Aspect Extraction ===
    logger.info("Extracting aspects using dictionary...")
    aspect_extractor = AspectDictionary(config['config'])
    
    df['aspects_dict'] = df['text_en_clean'].apply(aspect_extractor.extract_aspects)
    
    # Compute per-aspect statistics
    aspect_stats = []
    for aspect in config['config']['aspects']['target_catalog']:
        aspect_mask = df['aspects_dict'].apply(lambda x: aspect in x)
        aspect_reviews = df[aspect_mask]
        
        if len(aspect_reviews) > 0:
            # Negative share (stars <= 2)
            neg_share = (aspect_reviews['stars_num'] <= 2).mean()
            
            aspect_stats.append({
                'aspect': aspect,
                'count': len(aspect_reviews),
                'avg_stars': aspect_reviews['stars_num'].mean(),
                'neg_share': neg_share,
                'neg_count': (aspect_reviews['stars_num'] <= 2).sum()
            })
    
    aspect_df = pd.DataFrame(aspect_stats).sort_values('count', ascending=False)
    aspect_table_path = outputs_dir / 'tables' / 'aspects_summary_dict.csv'
    aspect_df.to_csv(aspect_table_path, index=False)
    logger.info(f"Aspect summary (dictionary) saved to {aspect_table_path}")
    
    # Save processed data
    output_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved baseline results to {output_path}")
    
    # Summary
    logger.info(f"\nS4 Summary:")
    logger.info(f"  Total reviews: {len(df)}")
    logger.info(f"  VADER sentiment distribution: {df['vader_label'].value_counts().to_dict()}")
    logger.info(f"  Best topic k: {topic_results['best_k']} (silhouette: {topic_results['best_silhouette']:.4f})")
    logger.info(f"  Aspects detected: {len(aspect_df)} aspects")
    
    # Topic insights
    topic_df_sorted = pd.DataFrame(topic_stats).sort_values('avg_stars')
    problem_topics = topic_df_sorted[topic_df_sorted['avg_stars'] < 3.5]
    strong_topics = topic_df_sorted[topic_df_sorted['avg_stars'] >= 4.7]
    
    if len(problem_topics) > 0:
        logger.info(f"\n  Problem Topics (low stars):")
        for _, row in problem_topics.iterrows():
            logger.info(f"    - {row['label']}: {row['count']} reviews, {row['avg_stars']:.2f} stars")
    
    if len(strong_topics) > 0:
        logger.info(f"\n  Strong Topics (high stars):")
        for _, row in strong_topics.head(3).iterrows():
            logger.info(f"    - {row['label']}: {row['count']} reviews, {row['avg_stars']:.2f} stars")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s4_baselines()
    print(f"\nBaselines complete. Shape: {df.shape}")

