"""
S7: Final EDA & Storytelling Assets
All visualization and plotting logic consolidated here.
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def plot_aspect_negative_share(dict_df: pd.DataFrame, pyabsa_df: pd.DataFrame, output_path: Path):
    """Plot negative share comparison for aspects."""
    # Merge on aspect
    merged = dict_df[['aspect', 'neg_share']].merge(
        pyabsa_df[['aspect', 'neg_share']], 
        on='aspect', 
        suffixes=('_dict', '_pyabsa'),
        how='outer'
    ).fillna(0)
    
    # Filter to top aspects by total count
    top_aspects = merged.nlargest(10, ['neg_share_dict', 'neg_share_pyabsa'].copy()).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(top_aspects))
    width = 0.35
    
    ax.bar(x - width/2, top_aspects['neg_share_dict'], width, label='Dictionary', color='steelblue')
    ax.bar(x + width/2, top_aspects['neg_share_pyabsa'], width, label='PyABSA', color='coral')
    
    ax.set_xlabel('Aspect', fontsize=12)
    ax.set_ylabel('Negative Share (%)', fontsize=12)
    ax.set_title('Negative Sentiment Share by Aspect: Dictionary vs PyABSA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top_aspects['aspect'], rotation=45, ha='right')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Aspect negative share plot saved to {output_path}")


def plot_topic_clusters(df: pd.DataFrame, output_path: Path):
    """Plot topic clusters with labels."""
    topic_counts = df['topic_label'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(topic_counts.index[:10], topic_counts.values[:10], color='steelblue')
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Topic', fontsize=12)
    ax.set_title('Top 10 Review Topics (Auto-Labeled)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (topic, count) in enumerate(zip(topic_counts.index[:10], topic_counts.values[:10])):
        ax.text(count + max(topic_counts.values) * 0.01, i, str(count), va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Topic clusters plot saved to {output_path}")


def plot_keyword_rates_by_rating(df: pd.DataFrame, output_path: Path):
    """Plot keyword mention rates by star rating."""
    keywords = {
        'Sound': ['sound', 'audio', 'bass'],
        'Comfort': ['comfort', 'fit', 'tight', 'hurt'],
        'Battery': ['battery', 'charge', 'life']
    }
    
    # For each star rating, compute keyword mention rate
    stars = [1, 2, 3, 4, 5]
    keyword_rates = {kw: [] for kw in keywords}
    
    for star in stars:
        star_reviews = df[df['stars_num'] == star]['text_en_clean']
        
        for kw_name, kw_list in keywords.items():
            mentions = star_reviews.apply(
                lambda text: any(kw in str(text).lower() for kw in kw_list)
            ).mean()
            keyword_rates[kw_name].append(mentions)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for kw_name, rates in keyword_rates.items():
        ax.plot(stars, rates, marker='o', label=kw_name, linewidth=2)
    
    ax.set_xlabel('Star Rating', fontsize=12)
    ax.set_ylabel('Mention Rate', fontsize=12)
    ax.set_title('Keyword Mention Rates by Star Rating', fontsize=14, fontweight='bold')
    ax.set_xticks(stars)
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Keyword rates plot saved to {output_path}")


def plot_text_length_distribution(df: pd.DataFrame, output_path: Path):
    """Plot text length distribution by star rating."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overall distribution
    ax1 = axes[0]
    ax1.hist(df['word_len'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(df['word_len'].median(), color='red', linestyle='--', linewidth=2, label=f'Median: {df["word_len"].median():.0f}')
    ax1.axvline(df['word_len'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {df["word_len"].mean():.0f}')
    ax1.set_xlabel('Word Count', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Review Length Distribution (All)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: By star rating
    ax2 = axes[1]
    star_ratings = [1, 2, 3, 4, 5]
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#90ee90', '#2ca02c']
    
    for star, color in zip(star_ratings, colors):
        star_data = df[df['stars_num'] == star]['word_len']
        ax2.hist(star_data, bins=30, alpha=0.5, label=f'{star}★ (μ={star_data.mean():.0f})', 
                color=color, edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Word Count', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Review Length by Star Rating', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Text length distribution plot saved to {output_path}")


def plot_top_ngrams_negative(df: pd.DataFrame, output_path: Path):
    """Plot top n-grams in negative reviews (1-2 stars)."""
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Filter negative reviews
    negative_reviews = df[df['stars_num'] <= 2]['text_en_clean'].dropna()
    
    if len(negative_reviews) < 10:
        logger.warning("Not enough negative reviews for n-gram analysis")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Unigrams
    ax1 = axes[0]
    vectorizer_1 = CountVectorizer(ngram_range=(1, 1), max_features=15, stop_words='english')
    unigrams = vectorizer_1.fit_transform(negative_reviews)
    unigram_counts = unigrams.sum(axis=0).A1
    unigram_names = vectorizer_1.get_feature_names_out()
    
    top_unigrams = sorted(zip(unigram_names, unigram_counts), key=lambda x: x[1], reverse=True)[:15]
    words, counts = zip(*top_unigrams)
    
    ax1.barh(range(len(words)), counts, color='coral')
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words)
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_title('Top 15 Words in Negative Reviews (1-2★)', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    for i, count in enumerate(counts):
        ax1.text(count + max(counts) * 0.01, i, str(count), va='center', fontsize=9)
    
    # Bigrams
    ax2 = axes[1]
    vectorizer_2 = CountVectorizer(ngram_range=(2, 2), max_features=15, stop_words='english')
    bigrams = vectorizer_2.fit_transform(negative_reviews)
    bigram_counts = bigrams.sum(axis=0).A1
    bigram_names = vectorizer_2.get_feature_names_out()
    
    top_bigrams = sorted(zip(bigram_names, bigram_counts), key=lambda x: x[1], reverse=True)[:15]
    phrases, counts2 = zip(*top_bigrams)
    
    ax2.barh(range(len(phrases)), counts2, color='steelblue')
    ax2.set_yticks(range(len(phrases)))
    ax2.set_yticklabels(phrases)
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_title('Top 15 Bigrams in Negative Reviews (1-2★)', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    
    for i, count in enumerate(counts2):
        ax2.text(count + max(counts2) * 0.01, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Top n-grams (negative) plot saved to {output_path}")


def plot_topic_clusters_2d(df: pd.DataFrame, topics_df: pd.DataFrame, output_dir: Path):
    """Create 2D scatter plots of topic clusters using t-SNE and PCA."""
    logger.info("Creating 2D cluster visualizations...")
    
    # Filter out low-info texts
    df_filtered = df[~df['low_info']].copy()
    
    # Recreate TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=10000,
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df_filtered['text_en_lemma'])
    clusters = df_filtered['topic_cluster'].values
    
    # Sample for speed
    sample_size = min(3000, len(df_filtered))
    
    # === t-SNE Visualization ===
    logger.info("  Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords_2d = tsne.fit_transform(tfidf_matrix.toarray()[:sample_size])
    
    plot_df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'cluster': clusters[:sample_size],
        'label': df_filtered['topic_label'].values[:sample_size]
    })
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    cluster_labels = topics_df.set_index('cluster_id')['label'].to_dict()
    n_clusters = len(plot_df['cluster'].unique())
    colors = sns.color_palette('husl', n_clusters)
    
    for idx, cluster_id in enumerate(sorted(plot_df['cluster'].unique())):
        if cluster_id == -1:
            continue
            
        mask = plot_df['cluster'] == cluster_id
        cluster_data = plot_df[mask]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        
        ax.scatter(
            cluster_data['x'], cluster_data['y'],
            c=[colors[idx]], label=label, alpha=0.6, s=50,
            edgecolors='white', linewidth=0.5
        )
        
        centroid_x = cluster_data['x'].mean()
        centroid_y = cluster_data['y'].mean()
        short_label = label[:30] + '...' if len(label) > 30 else label
        
        ax.annotate(
            short_label, (centroid_x, centroid_y),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[idx], alpha=0.7, edgecolor='white'),
            ha='center', va='center', color='white'
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title('Topic Clusters - 2D Projection (t-SNE)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_clusters_2d_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # === PCA Visualization ===
    logger.info("  Computing PCA projection...")
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(tfidf_matrix.toarray()[:sample_size])
    
    plot_df_pca = pd.DataFrame({
        'x': coords_pca[:, 0],
        'y': coords_pca[:, 1],
        'cluster': clusters[:sample_size],
        'label': df_filtered['topic_label'].values[:sample_size]
    })
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    for idx, cluster_id in enumerate(sorted(plot_df_pca['cluster'].unique())):
        if cluster_id == -1:
            continue
            
        mask = plot_df_pca['cluster'] == cluster_id
        cluster_data = plot_df_pca[mask]
        label = cluster_labels.get(cluster_id, f'Cluster {cluster_id}')
        
        ax.scatter(
            cluster_data['x'], cluster_data['y'],
            c=[colors[idx]], label=label, alpha=0.6, s=50,
            edgecolors='white', linewidth=0.5
        )
        
        centroid_x = cluster_data['x'].mean()
        centroid_y = cluster_data['y'].mean()
        short_label = label[:30] + '...' if len(label) > 30 else label
        
        ax.annotate(
            short_label, (centroid_x, centroid_y),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[idx], alpha=0.7, edgecolor='white'),
            ha='center', va='center', color='white'
        )
    
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({variance[0]*100:.1f}% variance)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'PC2 ({variance[1]*100:.1f}% variance)', fontsize=14, fontweight='bold')
    ax.set_title('Topic Clusters - 2D Projection (PCA)', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'topic_clusters_2d_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  2D cluster visualizations saved")


def plot_siebert_agreement(df: pd.DataFrame, output_dir: Path):
    """Create SiEBERT vs Stars agreement analysis with error examples."""
    logger.info("Creating SiEBERT agreement analysis...")
    
    # Map stars to sentiment
    def stars_to_sentiment(stars):
        if stars <= 2:
            return 'negative'
        elif stars == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df['stars_sentiment'] = df['stars_num'].apply(stars_to_sentiment)
    df['siebert_sentiment'] = df['siebert_label'].str.lower()
    df['agree'] = df['stars_sentiment'] == df['siebert_sentiment']
    
    overall_agreement = df['agree'].mean()
    
    # === Agreement Chart ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SiEBERT vs Stars: Agreement Analysis', fontsize=16, fontweight='bold')
    
    # Chart 1: Agreement by Star Rating
    ax1 = axes[0, 0]
    agreement_by_star = df.groupby('stars_num')['agree'].agg(['sum', 'count'])
    agreement_by_star['disagree'] = agreement_by_star['count'] - agreement_by_star['sum']
    agreement_by_star['agree'] = agreement_by_star['sum']
    
    x = agreement_by_star.index
    agree_counts = agreement_by_star['agree']
    disagree_counts = agreement_by_star['disagree']
    
    ax1.bar(x, agree_counts, label='Agree', color='#2ecc71', alpha=0.8)
    ax1.bar(x, disagree_counts, bottom=agree_counts, label='Disagree', color='#e74c3c', alpha=0.8)
    ax1.set_xlabel('Star Rating', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Agreement by Star Rating', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.set_xticks([1, 2, 3, 4, 5])
    
    for i, star in enumerate(x):
        total = agreement_by_star.loc[star, 'count']
        agree = agreement_by_star.loc[star, 'agree']
        pct = agree / total * 100
        ax1.text(star, total + 20, f'{pct:.1f}%', ha='center', fontweight='bold')
    
    # Chart 2: Agreement Rate by Star
    ax2 = axes[0, 1]
    agreement_rate = df.groupby('stars_num')['agree'].mean() * 100
    colors_rate = ['#e74c3c' if rate < 80 else '#f39c12' if rate < 90 else '#2ecc71' 
                   for rate in agreement_rate]
    
    bars = ax2.bar(agreement_rate.index, agreement_rate, color=colors_rate, alpha=0.8, edgecolor='black')
    ax2.axhline(y=overall_agreement*100, color='red', linestyle='--', linewidth=2, 
                label=f'Overall ({overall_agreement:.1%})')
    ax2.set_xlabel('Star Rating', fontsize=12)
    ax2.set_ylabel('Agreement Rate (%)', fontsize=12)
    ax2.set_title('Agreement Rate by Star Rating', fontsize=13, fontweight='bold')
    ax2.set_xticks([1, 2, 3, 4, 5])
    ax2.set_ylim(0, 105)
    ax2.legend()
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Chart 3: Confusion Breakdown
    ax3 = axes[1, 0]
    confusion_data = pd.crosstab(df['stars_sentiment'], df['siebert_sentiment'], normalize='index') * 100
    sns.heatmap(confusion_data, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                ax=ax3, cbar_kws={'label': 'Percentage (%)'})
    ax3.set_xlabel('SiEBERT Prediction', fontsize=12)
    ax3.set_ylabel('Star-based Sentiment', fontsize=12)
    ax3.set_title('Confusion Matrix (%)', fontsize=13, fontweight='bold')
    
    # Chart 4: Error Types
    ax4 = axes[1, 1]
    errors = df[~df['agree']].copy()
    
    def categorize_error(row):
        if row['stars_sentiment'] == 'positive' and row['siebert_sentiment'] == 'negative':
            return 'False Negative\n(5* → NEG)'
        elif row['stars_sentiment'] == 'negative' and row['siebert_sentiment'] == 'positive':
            return 'False Positive\n(1* → POS)'
        elif row['stars_sentiment'] == 'neutral':
            return 'Neutral Mismatch'
        else:
            return 'Other'
    
    errors['error_type'] = errors.apply(categorize_error, axis=1)
    error_counts = errors['error_type'].value_counts()
    
    colors_pie = ['#e74c3c', '#3498db', '#f39c12', '#9b59b6']
    ax4.pie(error_counts.values, labels=error_counts.index, autopct='%1.1f%%',
            colors=colors_pie, startangle=90)
    ax4.set_title('Error Types Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'siebert_vs_stars_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # === Error Examples CSV ===
    errors_detailed = []
    error_types = [
        ('False Negative', 'positive', 'negative', 5),
        ('False Positive', 'negative', 'positive', 10),
    ]
    
    for error_name, star_sent, siebert_sent, n_examples in error_types:
        subset = df[
            (df['stars_sentiment'] == star_sent) & 
            (df['siebert_sentiment'] == siebert_sent)
        ].copy()
        
        if len(subset) > 0:
            subset = subset.sort_values('siebert_score', ascending=False).head(n_examples)
            
            for _, row in subset.iterrows():
                review_text = row['text_en_clean'][:200] + '...' if len(row['text_en_clean']) > 200 else row['text_en_clean']
                
                errors_detailed.append({
                    'Error_Type': error_name,
                    'Stars': int(row['stars_num']),
                    'Star_Sentiment': star_sent.title(),
                    'SiEBERT_Prediction': siebert_sent.title(),
                    'SiEBERT_Confidence': f"{row['siebert_score']:.3f}",
                    'Review_Text': review_text
                })
    
    errors_df = pd.DataFrame(errors_detailed)
    errors_df.to_csv(output_dir.parent / 'tables' / 'siebert_error_examples.csv', index=False)
    
    logger.info(f"  SiEBERT agreement analysis saved")


def run_s7_story(config_path: str = "config.yaml"):
    """
    S7: Storytelling Assets Pipeline
    - Generate all required figures and tables
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load evaluation data
    input_path = Path(config['data']['processed_dir']) / '06_evaluation.parquet'
    df = pd.read_parquet(input_path)
    
    logger.info(f"Loaded {len(df)} reviews for storytelling")
    
    outputs_dir = Path(config['data']['outputs_dir'])
    
    # Load aspect summaries
    dict_aspects = pd.read_csv(outputs_dir / 'tables' / 'aspects_summary_dict.csv')
    pyabsa_aspects = pd.read_csv(outputs_dir / 'tables' / 'aspects_summary_pyabsa.csv')
    
    # === Generate Required Figures ===
    
    # 1. ratings_histogram.png - already created in S4, verify existence
    ratings_hist_path = outputs_dir / 'figures' / 'ratings_histogram.png'
    if ratings_hist_path.exists():
        logger.info(f"Ratings histogram already exists at {ratings_hist_path}")
    
    # 2. language_distribution.png - already created in S1, verify
    lang_dist_path = outputs_dir / 'figures' / 'language_distribution.png'
    if lang_dist_path.exists():
        logger.info(f"Language distribution already exists at {lang_dist_path}")
    
    # 3. sentiment_confusion_vader_vs_siebert.png - already created in S6
    vader_siebert_path = outputs_dir / 'figures' / 'sentiment_confusion_vader_vs_siebert.png'
    if vader_siebert_path.exists():
        logger.info(f"VADER vs SiEBERT confusion already exists at {vader_siebert_path}")
    
    # 4. aspect_negative_share_pyabsa_vs_dict.png - create now
    logger.info("Creating aspect negative share comparison plot...")
    aspect_neg_path = outputs_dir / 'figures' / 'aspect_negative_share_pyabsa_vs_dict.png'
    plot_aspect_negative_share(dict_aspects, pyabsa_aspects, aspect_neg_path)
    
    # 5. topic_clusters_with_auto_labels.png - create now
    logger.info("Creating topic clusters plot...")
    topic_path = outputs_dir / 'figures' / 'topic_clusters_with_auto_labels.png'
    plot_topic_clusters(df, topic_path)
    
    # 6. keyword_rates_by_rating_sound_comfort_battery.png - create now
    logger.info("Creating keyword rates plot...")
    keyword_path = outputs_dir / 'figures' / 'keyword_rates_by_rating_sound_comfort_battery.png'
    plot_keyword_rates_by_rating(df, keyword_path)
    
    # 7. text_length_distribution.png - create now
    logger.info("Creating text length distribution plot...")
    text_len_path = outputs_dir / 'figures' / 'text_length_distribution.png'
    plot_text_length_distribution(df, text_len_path)
    
    # 8. top_ngrams_negative.png - create now
    logger.info("Creating top n-grams in negative reviews plot...")
    ngrams_path = outputs_dir / 'figures' / 'top_ngrams_negative.png'
    plot_top_ngrams_negative(df, ngrams_path)
    
    # 9. 2D topic cluster visualizations (t-SNE and PCA)
    logger.info("Creating 2D topic cluster visualizations...")
    topics_summary = pd.read_csv(outputs_dir / 'tables' / 'topics_summary.csv')
    plot_topic_clusters_2d(df, topics_summary, outputs_dir / 'figures')
    
    # 10. SiEBERT agreement analysis
    logger.info("Creating SiEBERT agreement analysis...")
    plot_siebert_agreement(df, outputs_dir / 'figures')
    
    # === Verify Required Tables ===
    required_tables = [
        'aspects_summary_dict.csv',
        'aspects_summary_pyabsa.csv',
        'evaluation_metrics.csv',
        'language_distribution.csv'
    ]
    
    for table_name in required_tables:
        table_path = outputs_dir / 'tables' / table_name
        if table_path.exists():
            logger.info(f"Table exists: {table_name}")
        else:
            logger.warning(f"Table missing: {table_name}")
    
    # === Summary Statistics Table ===
    summary_stats = {
        'Total Reviews': len(df),
        'Avg Star Rating': df['stars_num'].mean(),
        'Positive Reviews (4-5*)': (df['stars_num'] >= 4).sum(),
        'Negative Reviews (1-2*)': (df['stars_num'] <= 2).sum(),
        'Neutral Reviews (3*)': (df['stars_num'] == 3).sum(),
        'Avg Word Count': df['word_len'].mean(),
        'Total Topics': df['topic_cluster'].nunique(),
        'Total Aspects (Dict)': dict_aspects['count'].sum(),
        'Total Aspects (PyABSA)': pyabsa_aspects['count'].sum() if not pyabsa_aspects.empty else 0
    }
    
    summary_df = pd.DataFrame([summary_stats]).T
    summary_df.columns = ['Value']
    summary_path = outputs_dir / 'tables' / 'summary_statistics.csv'
    summary_df.to_csv(summary_path)
    logger.info(f"Summary statistics saved to {summary_path}")
    
    logger.info("\nS7 Summary: All storytelling assets generated")
    
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    run_s7_story()
    print("\nStorytelling assets complete.")

