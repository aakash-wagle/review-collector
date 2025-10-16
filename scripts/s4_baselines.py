"""
S4: Deterministic Baselines (EDA, Sentiment, Topics, Aspects)
Performs exploratory data analysis and baseline modeling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def analyze_ratings_over_time(df, config, output_dir):
    """Analyze rating trends over time with confidence intervals."""
    logger = setup_logging(config)
    
    # Monthly average stars
    monthly_stats = df.groupby('month_start').agg({
        'stars_num': ['mean', 'std', 'count']
    }).reset_index()
    monthly_stats.columns = ['month_start', 'mean_stars', 'std_stars', 'count']
    
    # Bootstrap confidence intervals (simplified)
    monthly_stats['ci_lower'] = monthly_stats['mean_stars'] - 1.96 * monthly_stats['std_stars'] / np.sqrt(monthly_stats['count'])
    monthly_stats['ci_upper'] = monthly_stats['mean_stars'] + 1.96 * monthly_stats['std_stars'] / np.sqrt(monthly_stats['count'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_stats['month_start'], monthly_stats['mean_stars'], 'o-', linewidth=2, markersize=8, label='Mean Rating')
    ax.fill_between(monthly_stats['month_start'], monthly_stats['ci_lower'], monthly_stats['ci_upper'], alpha=0.3, label='95% CI')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Stars', fontsize=12)
    ax.set_title('iPhone 17 Pro Max: Ratings Over Time with 95% Confidence Interval', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = output_dir / 'ratings_over_time_with_CI.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {output_path}")
    
    return monthly_stats

def analyze_sentiment_vader(df, config):
    """Perform VADER sentiment analysis."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    
    # Compute sentiment scores
    tqdm.pandas(desc="VADER sentiment")
    df['vader_scores'] = df['text_en_clean'].progress_apply(
        lambda x: analyzer.polarity_scores(x)
    )
    
    # Extract compound score
    df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
    
    # Classify sentiment
    thresholds = config['config']['sentiment']['vader']['thresholds']
    df['vader_sentiment'] = pd.cut(
        df['vader_compound'],
        bins=[-1, thresholds['neg'], thresholds['pos'], 1],
        labels=['negative', 'neutral', 'positive']
    )
    
    return df

def build_aspect_dictionary(config):
    """Build aspect detection dictionaries."""
    aspects_dict = {
        'Battery': [
            'battery', 'battery life', 'charge', 'charging', 'power', 'drain',
            'last', 'lasts', 'lasting', 'juice', 'mah', 'hours'
        ],
        'Performance/Overheating': [
            'performance', 'speed', 'fast', 'slow', 'lag', 'laggy', 'smooth',
            'hot', 'heat', 'heating', 'overheat', 'overheating', 'warm', 'burning',
            'throttle', 'throttling', 'processor', 'chip', 'a18'
        ],
        'Camera': [
            'camera', 'photo', 'photos', 'picture', 'pictures', 'image', 'images',
            'video', 'videos', 'lens', 'zoom', 'portrait', 'shot', 'shots'
        ],
        'Display': [
            'display', 'screen', 'brightness', 'bright', 'dim', 'resolution',
            'oled', 'refresh', 'hz', 'sunlight', 'outdoor'
        ],
        'Durability/Build': [
            'build', 'quality', 'durability', 'durable', 'scratch', 'scratches',
            'bend', 'bending', 'break', 'broke', 'broken', 'crack', 'cracked',
            'titanium', 'aluminum', 'aluminium', 'metal', 'glass', 'gorilla',
            'strong', 'weak', 'fragile', 'solid', 'premium'
        ],
        'Charging/USB-C': [
            'usb', 'usb-c', 'usbc', 'cable', 'charger', 'charging', 'port',
            'magsafe', 'wireless', 'fast charge', 'quick charge'
        ],
        'iOS/Software': [
            'ios', 'software', 'update', 'updates', 'bug', 'bugs', 'glitch',
            'glitches', 'app', 'apps', 'system', 'interface', 'ui', 'siri'
        ],
        'Price/Value': [
            'price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money',
            'afford', 'overpriced', 'underpriced', 'deal', 'sale', 'discount'
        ],
        'Weight/Ergonomics': [
            'weight', 'heavy', 'light', 'lighter', 'heavier', 'ergonomic',
            'comfortable', 'comfort', 'hold', 'holding', 'hand', 'grip',
            'bulky', 'size', 'compact'
        ]
    }
    
    return aspects_dict

def detect_aspects_dictionary(df, config):
    """Detect aspects using dictionary matching."""
    aspects_dict = build_aspect_dictionary(config)
    
    # Initialize aspect columns
    for aspect in aspects_dict.keys():
        df[f'aspect_{aspect}'] = False
    
    # Detect aspects
    for aspect, keywords in tqdm(aspects_dict.items(), desc="Aspect detection"):
        pattern = '|'.join([f'\\b{kw}\\b' for kw in keywords])
        df[f'aspect_{aspect}'] = df['text_en_clean'].str.contains(pattern, case=False, regex=True, na=False)
    
    return df

def topic_modeling_kmeans(df, config, output_dir):
    """Perform topic modeling using KMeans on TF-IDF vectors."""
    logger = setup_logging(config)
    
    # TF-IDF vectorization
    tfidf_config = config['config']['topics']['tfidf']
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(tfidf_config['ngram_range']),
        min_df=tfidf_config['min_df'],
        max_features=tfidf_config['max_features'],
        stop_words='english'
    )
    
    logger.info("Computing TF-IDF vectors...")
    tfidf_matrix = vectorizer.fit_transform(df['text_en_lemma'])
    
    # Find optimal k using silhouette score
    kmeans_config = config['config']['topics']['kmeans']
    k_range = range(kmeans_config['k_min'], kmeans_config['k_max'] + 1)
    silhouette_scores = []
    
    logger.info("Finding optimal number of clusters...")
    for k in tqdm(k_range, desc="Silhouette scores"):
        kmeans = KMeans(n_clusters=k, random_state=config['config']['seeds']['sklearn'], n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, labels, sample_size=min(5000, len(df)))
        silhouette_scores.append(score)
    
    # Choose best k
    best_idx = np.argmax(silhouette_scores)
    best_k = list(k_range)[best_idx]
    best_score = silhouette_scores[best_idx]
    
    logger.info(f"Optimal k: {best_k} (silhouette score: {best_score:.4f})")
    
    # Final clustering
    logger.info(f"Performing final clustering with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=config['config']['seeds']['sklearn'], n_init=20)
    df['topic_cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # Extract top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    cluster_terms = {}
    
    for cluster_id in range(best_k):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        cluster_terms[cluster_id] = top_terms
        logger.info(f"  Cluster {cluster_id}: {', '.join(top_terms[:5])}")
    
    # Save cluster info
    cluster_info = pd.DataFrame({
        'cluster_id': range(best_k),
        'top_terms': [', '.join(terms) for terms in cluster_terms.values()],
        'count': [sum(df['topic_cluster'] == i) for i in range(best_k)]
    })
    
    cluster_info.to_csv(output_dir.parent / 'tables' / 'topics_summary_labeled.csv', index=False)
    
    return df, best_k, best_score, cluster_terms

def deterministic_baselines(config):
    """
    Pipeline Stage 4: Deterministic Baselines
    
    Steps:
    1. Rating trends and EDA
    2. Sentiment analysis (VADER)
    3. Topic modeling (KMeans)
    4. Aspect detection (dictionary)
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S4: DETERMINISTIC BASELINES")
    logger.info("=" * 80)
    
    # Load input data
    input_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    output_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    
    logger.info(f"Loading data from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} reviews")
    
    # Create output directories
    figures_dir = Path(config['data']['outputs_dir']) / 'figures'
    tables_dir = Path(config['data']['outputs_dir']) / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ratings over time
    logger.info("\n1. Analyzing ratings over time...")
    monthly_stats = analyze_ratings_over_time(df, config, figures_dir)
    monthly_stats.to_csv(tables_dir / 'monthly_ratings_stats.csv', index=False)
    
    # Star distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    star_counts = df['stars_num'].value_counts().sort_index()
    ax.bar(star_counts.index, star_counts.values, color='steelblue', alpha=0.8)
    ax.set_xlabel('Star Rating', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Star Ratings', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'star_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Translation share over time
    if 'lang' in df.columns:
        trans_monthly = df.groupby('month_start').agg({
            'lang': lambda x: (x != 'en').sum() / len(x) * 100
        }).reset_index()
        trans_monthly.columns = ['month_start', 'pct_translated']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(trans_monthly['month_start'], trans_monthly['pct_translated'], 'o-', linewidth=2, markersize=8, color='coral')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('% Non-English Reviews', fontsize=12)
        ax.set_title('Share of Translated (Non-English) Reviews Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figures_dir / 'share_translated_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {figures_dir / 'share_translated_over_time.png'}")
    
    # 2. Sentiment analysis
    logger.info("\n2. Performing sentiment analysis (VADER)...")
    df = analyze_sentiment_vader(df, config)
    
    sentiment_dist = df['vader_sentiment'].value_counts()
    logger.info(f"Sentiment distribution:")
    for sent, count in sentiment_dist.items():
        logger.info(f"  {sent}: {count:,} ({count / len(df) * 100:.1f}%)")
    
    # Sentiment vs stars
    sentiment_stars = pd.crosstab(df['vader_sentiment'], df['stars_num'], normalize='columns') * 100
    sentiment_stars.to_csv(tables_dir / 'sentiment_vs_stars.csv')
    
    # 3. Topic modeling
    logger.info("\n3. Performing topic modeling (KMeans on TF-IDF)...")
    df, best_k, best_score, cluster_terms = topic_modeling_kmeans(df, config, figures_dir)
    
    logger.info(f"Topic modeling complete: {best_k} clusters, silhouette={best_score:.4f}")
    
    # 4. Aspect detection
    logger.info("\n4. Detecting aspects (dictionary-based)...")
    df = detect_aspects_dictionary(df, config)
    
    # Aspect summary
    aspect_cols = [col for col in df.columns if col.startswith('aspect_')]
    aspect_names = [col.replace('aspect_', '') for col in aspect_cols]
    
    aspect_summary = []
    for aspect_col, aspect_name in zip(aspect_cols, aspect_names):
        aspect_df = df[df[aspect_col]]
        if len(aspect_df) > 0:
            summary = {
                'aspect': aspect_name,
                'volume': len(aspect_df),
                'pct_total': len(aspect_df) / len(df) * 100,
                'avg_stars': aspect_df['stars_num'].mean(),
                'neg_share': (aspect_df['stars_num'] <= 3).sum() / len(aspect_df) * 100
            }
            aspect_summary.append(summary)
    
    aspect_summary_df = pd.DataFrame(aspect_summary).sort_values('volume', ascending=False)
    aspect_summary_df.to_csv(tables_dir / 'aspects_summary_dict.csv', index=False)
    
    logger.info("\nAspect Summary (Dictionary-based):")
    for _, row in aspect_summary_df.iterrows():
        logger.info(f"  {row['aspect']}: {row['volume']:,} mentions ({row['pct_total']:.1f}%), "
                   f"avg stars={row['avg_stars']:.2f}, neg%={row['neg_share']:.1f}%")
    
    # Save processed data
    logger.info(f"\nSaving to {output_path}")
    df.to_parquet(output_path, index=False)
    
    logger.info("✓ S4 completed successfully")
    return df

if __name__ == "__main__":
    config = load_config()
    df = deterministic_baselines(config)
    print(f"\n✓ Stage 4 complete. Analyzed {len(df):,} reviews.")

