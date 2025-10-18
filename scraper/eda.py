import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_explore_data(file_path):
    df = pd.read_csv(file_path)
    
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names: {list(df.columns)}")
    print("\n" + "-" * 80)
    print("FIRST 5 ROWS")
    print("-" * 80)
    print(df.head())
    
    print("\n" + "-" * 80)
    print("DATA TYPES")
    print("-" * 80)
    print(df.dtypes)
    
    print("\n" + "-" * 80)
    print("MISSING VALUES")
    print("-" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df)
    
    print(df.describe(include='all'))
    
    return df


def extract_numeric_rating(stars_text):
    if pd.isna(stars_text):
        return np.nan
    match = re.search(r'(\d+\.?\d*)', str(stars_text))
    if match:
        return float(match.group(1))
    return np.nan


def extract_months_ago(date_text):
    if pd.isna(date_text):
        return np.nan
    match = re.search(r'(\d+)\s+months?\s+ago', str(date_text))
    if match:
        return int(match.group(1))
    return np.nan


def analyze_ratings(df):
    df['rating'] = df['stars'].apply(extract_numeric_rating)
    
    print("\nRating Statistics:")
    print(df['rating'].describe())
    
    print("\nRating Distribution:")
    rating_counts = df['rating'].value_counts().sort_index()
    print(rating_counts)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    rating_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    rating_counts.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Rating Distribution (Percentage)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('')
    
    axes[1, 0].hist(df['rating'].dropna(), bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Rating Histogram', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Rating')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    axes[1, 1].boxplot(df['rating'].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
    axes[1, 1].set_title('Rating Box Plot', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Rating')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rating_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: rating_analysis.png")
    plt.show()
    
    return df


def analyze_text_length(df):
    df['text_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    
    print("\nText Length Statistics:")
    print(df['text_length'].describe())
    
    print("\nWord Count Statistics:")
    print(df['word_count'].describe())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(df['text_length'], bins=50, color='purple', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Review Text Length Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    axes[0, 1].hist(df['word_count'], bins=50, color='teal', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    axes[1, 0].scatter(df['rating'], df['text_length'], alpha=0.5, c=df['rating'], 
                      cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[1, 0].set_title('Text Length vs Rating', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Rating')
    axes[1, 0].set_ylabel('Text Length')
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].scatter(df['rating'], df['word_count'], alpha=0.5, c=df['rating'], 
                      cmap='plasma', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Word Count vs Rating', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Rating')
    axes[1, 1].set_ylabel('Word Count')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('text_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: text_analysis.png")
    plt.show()
    
    return df


def analyze_temporal_data(df):    
    df['months_ago'] = df['date'].apply(extract_months_ago)
    
    print("\nMonths Ago Statistics:")
    print(df['months_ago'].describe())
    
    print("\nReviews by Time Period:")
    time_counts = df['months_ago'].value_counts().sort_index()
    print(time_counts.head(10))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    time_counts.plot(kind='line', ax=axes[0], marker='o', color='darkblue', linewidth=2)
    axes[0].set_title('Number of Reviews Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Months Ago')
    axes[0].set_ylabel('Number of Reviews')
    axes[0].grid(alpha=0.3)
    
    avg_rating_time = df.groupby('months_ago')['rating'].mean()
    avg_rating_time.plot(kind='line', ax=axes[1], marker='s', color='darkred', linewidth=2)
    axes[1].set_title('Average Rating Over Time', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Months Ago')
    axes[1].set_ylabel('Average Rating')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: temporal_analysis.png")
    plt.show()
    
    return df


def analyze_text_content(df):
    all_text = ' '.join(df['review_text'].astype(str))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    wordcloud_all = WordCloud(width=800, height=400, background_color='white',
                             colormap='viridis', max_words=100).generate(all_text)
    axes[0, 0].imshow(wordcloud_all, interpolation='bilinear')
    axes[0, 0].set_title('Word Cloud - All Reviews', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    positive_text = ' '.join(df[df['rating'] >= 4.0]['review_text'].astype(str))
    if positive_text.strip():
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white',
                                 colormap='Greens', max_words=100).generate(positive_text)
        axes[0, 1].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0, 1].set_title('Word Cloud - Positive Reviews (≥4 stars)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    negative_text = ' '.join(df[df['rating'] <= 2.0]['review_text'].astype(str))
    if negative_text.strip():
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white',
                                 colormap='Reds', max_words=100).generate(negative_text)
        axes[1, 0].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1, 0].set_title('Word Cloud - Negative Reviews (≤2 stars)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    from collections import Counter
    
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
                      'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                      'might', 'can', 'these', 'they', 'them', 'their', 'i', 'my', 'me', 'you',
                      'your', 'it', 'its', 'that', 'this', 'from', 'as', 'so', 'if', 'than'])
    
    words = all_text.lower().split()
    words = [w for w in words if w.isalpha() and len(w) > 3 and w not in stop_words]
    word_freq = Counter(words).most_common(20)
    
    keywords, counts = zip(*word_freq) if word_freq else ([], [])
    axes[1, 1].barh(range(len(keywords)), counts, color='steelblue')
    axes[1, 1].set_yticks(range(len(keywords)))
    axes[1, 1].set_yticklabels(keywords)
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].set_title('Top 20 Keywords', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('text_content_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: text_content_analysis.png")
    plt.show()
    
    print("\nTop 20 Keywords:")
    for i, (word, count) in enumerate(word_freq[:20], 1):
        print(f"{i:2d}. {word:15s} - {count:4d} occurrences")


def create_correlation_analysis(df):
    numeric_cols = ['rating', 'text_length', 'word_count', 'months_ago']
    corr_data = df[numeric_cols].dropna()
    
    correlation_matrix = corr_data.corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: correlation_analysis.png")
    plt.show()


def preprocess_data(df):
    df_processed = df.copy()
    
    if 'rating' not in df_processed.columns:
        df_processed['rating'] = df_processed['stars'].apply(extract_numeric_rating)
    
    if 'months_ago' not in df_processed.columns:
        df_processed['months_ago'] = df_processed['date'].apply(extract_months_ago)
    
    df_processed['review_text_clean'] = df_processed['review_text'].str.strip()
    
    if 'text_length' not in df_processed.columns:
        df_processed['text_length'] = df_processed['review_text'].str.len()
    if 'word_count' not in df_processed.columns:
        df_processed['word_count'] = df_processed['review_text'].str.split().str.len()
    
    def categorize_sentiment(rating):
        if pd.isna(rating):
            return 'Unknown'
        elif rating >= 4.0:
            return 'Positive'
        elif rating >= 3.0:
            return 'Neutral'
        else:
            return 'Negative'
    
    df_processed['sentiment'] = df_processed['rating'].apply(categorize_sentiment)
    
    df_processed['review_length_category'] = pd.cut(
        df_processed['word_count'],
        bins=[0, 50, 150, 300, float('inf')],
        labels=['Short', 'Medium', 'Long', 'Very Long']
    )
    
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=['review_text', 'rating'])
    removed_rows = initial_rows - len(df_processed)
    print(f"   Removed {removed_rows} rows with missing critical data")
    
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=['review_text'], keep='first')
    removed_duplicates = initial_rows - len(df_processed)
    print(f"   Removed {removed_duplicates} duplicate reviews")
    
    features = {
        'mentions_battery': df_processed['review_text'].str.lower().str.contains('battery', na=False),
        'mentions_sound': df_processed['review_text'].str.lower().str.contains('sound|audio|bass', na=False),
        'mentions_comfort': df_processed['review_text'].str.lower().str.contains('comfort|hurt|pain|ear', na=False),
        'mentions_price': df_processed['review_text'].str.lower().str.contains('price|\\$|dollar|sale|cheap|expensive', na=False),
        'mentions_bluetooth': df_processed['review_text'].str.lower().str.contains('bluetooth|wireless|connect', na=False),
        'mentions_quality': df_processed['review_text'].str.lower().str.contains('quality', na=False),
    }
    
    for feature_name, feature_values in features.items():
        df_processed[feature_name] = feature_values.astype(int)
    
    df_processed = df_processed.reset_index(drop=True)
    
    print(f"Original rows: {len(df)}")
    print(f"Processed rows: {len(df_processed)}")
    print(f"New columns added: {len(df_processed.columns) - len(df.columns)}")
    print(f"\nNew columns: {[col for col in df_processed.columns if col not in df.columns]}")
    
    print(df_processed[['rating', 'sentiment', 'word_count', 'review_length_category', 
                       'months_ago', 'mentions_sound', 'mentions_comfort']].head(10))
    
    return df_processed


def save_preprocessed_data(df_processed, output_path):
    df_processed.to_csv(output_path, index=False)
    print(f"\n✓ Preprocessed data saved to: {output_path}")
    print(f"  Rows: {len(df_processed)}")
    print(f"  Columns: {len(df_processed.columns)}")
    print(f"  File size: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB (in memory)")


def create_summary_statistics_table(df):
    summary_data = {
        'Metric': [
            'Total Reviews',
            'Average Rating',
            'Median Rating',
            'Mode Rating',
            'Std Dev Rating',
            'Average Word Count',
            'Median Word Count',
            'Average Text Length',
            'Positive Reviews (≥4)',
            'Neutral Reviews (3)',
            'Negative Reviews (≤2)',
            'Most Recent Review (months ago)',
            'Oldest Review (months ago)'
        ],
        'Value': [
            len(df),
            f"{df['rating'].mean():.2f}",
            f"{df['rating'].median():.2f}",
            f"{df['rating'].mode()[0]:.1f}" if not df['rating'].mode().empty else 'N/A',
            f"{df['rating'].std():.2f}",
            f"{df['word_count'].mean():.1f}",
            f"{df['word_count'].median():.1f}",
            f"{df['text_length'].mean():.1f}",
            len(df[df['rating'] >= 4.0]),
            len(df[df['rating'] == 3.0]),
            len(df[df['rating'] <= 2.0]),
            f"{df['months_ago'].min():.0f}" if df['months_ago'].notna().any() else 'N/A',
            f"{df['months_ago'].max():.0f}" if df['months_ago'].notna().any() else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='left', loc='center', colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary Statistics Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('summary_statistics.png', 
                dpi=300, bbox_inches='tight')
    print("\n✓ Saved: summary_statistics.png")
    plt.show()


def main():
    input_file = '../data/raw/reviews.csv'
    output_file = '../data/raw/reviews_preprocessed.csv'
    
    df = load_and_explore_data(input_file)
    
    df = analyze_ratings(df)
    
    df = analyze_text_length(df)
    
    df = analyze_temporal_data(df)
    
    analyze_text_content(df)
    create_correlation_analysis(df)
    
    create_summary_statistics_table(df)
    
    df_processed = preprocess_data(df)
    
    save_preprocessed_data(df_processed, output_file)
    
    print("ANALYSIS COMPLETE!")
    print("\nGenerated files:")
    print("  1. rating_analysis.png")
    print("  2. text_analysis.png")
    print("  3. temporal_analysis.png")
    print("  4. text_content_analysis.png")
    print("  5. correlation_analysis.png")
    print("  6. summary_statistics.png")
    print(f"  7. {output_file}")


if __name__ == "__main__":
    main()
