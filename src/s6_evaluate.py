"""
S6: Reconciliation & Evaluation (No LLM)
"""
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_sentiment_agreement(df: pd.DataFrame, method1: str, method2: str) -> Dict:
    """Compute agreement metrics between two sentiment methods."""
    # Convert stars to sentiment categories
    if method2 == 'stars':
        method2_labels = pd.cut(df['stars_num'], bins=[0, 2, 3, 5], labels=['negative', 'neutral', 'positive'])
    else:
        method2_labels = df[method2]
    
    method1_labels = df[method1]
    
    # Filter out any NaN values
    valid_mask = method1_labels.notna() & method2_labels.notna()
    method1_labels = method1_labels[valid_mask]
    method2_labels = method2_labels[valid_mask]
    
    # Accuracy
    accuracy = accuracy_score(method2_labels, method1_labels)
    
    # Confusion matrix
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(method2_labels, method1_labels, labels=labels)
    
    # Per-class metrics
    precision = precision_score(method2_labels, method1_labels, labels=labels, average='weighted', zero_division=0)
    recall = recall_score(method2_labels, method1_labels, labels=labels, average='weighted', zero_division=0)
    f1 = f1_score(method2_labels, method1_labels, labels=labels, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'labels': labels
    }


def compare_aspect_methods(df_baselines: pd.DataFrame, df_models: pd.DataFrame, aspect_catalog: list) -> pd.DataFrame:
    """Compare dictionary vs PyABSA aspect extraction."""
    comparison_results = []
    
    for aspect in aspect_catalog:
        # Dictionary: check if aspect in list
        dict_mentions = df_baselines['aspects_dict'].apply(lambda x: aspect in x if isinstance(x, list) else False)
        
        # PyABSA: check if aspect in raw extractions
        def check_pyabsa_aspect(x):
            # Handle string-serialized aspects
            if isinstance(x, str):
                import ast
                try:
                    x = ast.literal_eval(x)
                except:
                    x = []
            return any(asp.get('aspect') == aspect for asp in x if isinstance(asp, dict)) if isinstance(x, list) and x else False
        
        pyabsa_mentions = df_models['aspects_pyabsa_raw'].apply(check_pyabsa_aspect)
        
        # Compute overlap
        both = (dict_mentions & pyabsa_mentions).sum()
        dict_only = (dict_mentions & ~pyabsa_mentions).sum()
        pyabsa_only = (~dict_mentions & pyabsa_mentions).sum()
        neither = (~dict_mentions & ~pyabsa_mentions).sum()
        
        # Compute precision/recall treating dictionary as "ground truth" (imperfect but no manual labels)
        # PyABSA vs Dictionary
        if dict_mentions.sum() > 0:
            precision = both / (both + pyabsa_only) if (both + pyabsa_only) > 0 else 0
            recall = both / (both + dict_only) if (both + dict_only) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision, recall, f1 = 0, 0, 0
        
        comparison_results.append({
            'aspect': aspect,
            'dict_count': dict_mentions.sum(),
            'pyabsa_count': pyabsa_mentions.sum(),
            'both': both,
            'dict_only': dict_only,
            'pyabsa_only': pyabsa_only,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return pd.DataFrame(comparison_results)


def plot_sentiment_comparison(metrics: Dict, output_path: Path, title: str):
    """Plot confusion matrix for sentiment comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cm = metrics['confusion_matrix']
    labels = metrics['labels']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add metrics as text
    metrics_text = f"Accuracy: {metrics['accuracy']:.3f}\nF1: {metrics['f1']:.3f}"
    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sentiment comparison plot saved to {output_path}")


def run_s6_evaluate(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    S6: Evaluation Pipeline
    - Compare VADER vs Stars
    - Compare SiEBERT vs Stars
    - Compare PyABSA vs Dictionary aspects
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load processed data
    df_baselines = pd.read_parquet(Path(config['data']['processed_dir']) / '04_baselines.parquet')
    df_models = pd.read_parquet(Path(config['data']['processed_dir']) / '05_models.parquet')
    
    logger.info(f"Loaded {len(df_baselines)} baseline and {len(df_models)} model reviews")
    
    outputs_dir = Path(config['data']['outputs_dir'])
    
    # === VADER vs Stars ===
    logger.info("Evaluating VADER vs Stars...")
    vader_metrics = compute_sentiment_agreement(df_baselines, 'vader_label', 'stars')
    
    logger.info(f"VADER vs Stars - Accuracy: {vader_metrics['accuracy']:.3f}, F1: {vader_metrics['f1']:.3f}")
    
    # === SiEBERT vs Stars ===
    logger.info("Evaluating SiEBERT vs Stars...")
    
    # Map SiEBERT labels to standard format
    siebert_label_map = {'NEGATIVE': 'negative', 'NEUTRAL': 'neutral', 'POSITIVE': 'positive'}
    df_models['siebert_label_mapped'] = df_models['siebert_label'].map(siebert_label_map)
    
    siebert_metrics = compute_sentiment_agreement(df_models, 'siebert_label_mapped', 'stars')
    
    logger.info(f"SiEBERT vs Stars - Accuracy: {siebert_metrics['accuracy']:.3f}, F1: {siebert_metrics['f1']:.3f}")
    
    # Plot SiEBERT vs VADER confusion
    fig_path = outputs_dir / 'figures' / 'sentiment_confusion_vader_vs_siebert.png'
    
    # For this, we need to compare VADER and SiEBERT directly
    # Merge the dataframes on index
    df_merged = df_baselines[['vader_label']].join(df_models[['siebert_label_mapped']], how='inner')
    
    if len(df_merged) > 0:
        labels = ['negative', 'neutral', 'positive']
        cm = confusion_matrix(df_merged['vader_label'], df_merged['siebert_label_mapped'], labels=labels)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('SiEBERT Sentiment', fontsize=12)
        ax.set_ylabel('VADER Sentiment', fontsize=12)
        ax.set_title('VADER vs SiEBERT Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"VADER vs SiEBERT confusion plot saved to {fig_path}")
    
    # === PyABSA vs Dictionary Aspects ===
    logger.info("Evaluating PyABSA vs Dictionary aspects...")
    
    aspect_catalog = config['config']['aspects']['target_catalog']
    aspect_comparison = compare_aspect_methods(df_baselines, df_models, aspect_catalog)
    
    # Save comparison table
    aspect_table_path = outputs_dir / 'tables' / 'aspect_comparison_pyabsa_vs_dict.csv'
    aspect_table_path.parent.mkdir(parents=True, exist_ok=True)
    aspect_comparison.to_csv(aspect_table_path, index=False)
    logger.info(f"Aspect comparison saved to {aspect_table_path}")
    
    # Compute PyABSA aspect summary
    pyabsa_aspect_stats = []
    for aspect in aspect_catalog:
        aspect_reviews = []
        aspect_sentiments = []
        
        for idx, aspects in enumerate(df_models['aspects_pyabsa_raw']):
            # Handle string-serialized aspects (from parquet)
            if isinstance(aspects, str):
                import ast
                try:
                    aspects = ast.literal_eval(aspects)
                except:
                    aspects = []
            
            if isinstance(aspects, list):
                for asp in aspects:
                    if isinstance(asp, dict) and asp.get('aspect') == aspect:
                        aspect_reviews.append(idx)
                        aspect_sentiments.append(asp.get('sentiment', 'neutral'))
        
        if aspect_reviews:
            # Get stars for these reviews
            stars = df_models.loc[aspect_reviews, 'stars_num']
            neg_share = (stars <= 2).mean() if len(stars) > 0 else 0
            
            pyabsa_aspect_stats.append({
                'aspect': aspect,
                'count': len(aspect_reviews),
                'avg_stars': stars.mean() if len(stars) > 0 else 0,
                'neg_share': neg_share,
                'neg_count': (stars <= 2).sum() if len(stars) > 0 else 0,
                'sentiment_dist': pd.Series(aspect_sentiments).value_counts().to_dict() if aspect_sentiments else {}
            })
    
    if pyabsa_aspect_stats:
        pyabsa_aspect_df = pd.DataFrame(pyabsa_aspect_stats).sort_values('count', ascending=False)
    else:
        # Create empty DataFrame with expected columns
        pyabsa_aspect_df = pd.DataFrame(columns=['aspect', 'count', 'avg_stars', 'neg_share', 'neg_count', 'sentiment_dist'])
        logger.warning("No PyABSA aspects found, creating empty summary")
    
    pyabsa_table_path = outputs_dir / 'tables' / 'aspects_summary_pyabsa.csv'
    pyabsa_aspect_df.to_csv(pyabsa_table_path, index=False)
    logger.info(f"PyABSA aspect summary saved to {pyabsa_table_path}")
    
    # === Combine Results ===
    evaluation_metrics = {
        'VADER_vs_Stars': {
            'accuracy': vader_metrics['accuracy'],
            'precision': vader_metrics['precision'],
            'recall': vader_metrics['recall'],
            'f1': vader_metrics['f1']
        },
        'SiEBERT_vs_Stars': {
            'accuracy': siebert_metrics['accuracy'],
            'precision': siebert_metrics['precision'],
            'recall': siebert_metrics['recall'],
            'f1': siebert_metrics['f1']
        }
    }
    
    metrics_df = pd.DataFrame(evaluation_metrics).T
    metrics_table_path = outputs_dir / 'tables' / 'evaluation_metrics.csv'
    metrics_df.to_csv(metrics_table_path)
    logger.info(f"Evaluation metrics saved to {metrics_table_path}")
    
    # Merge all data for final output
    df_merged = df_baselines.copy()
    df_merged['siebert_label'] = df_models['siebert_label']
    df_merged['siebert_score'] = df_models['siebert_score']
    df_merged['aspects_pyabsa_raw'] = df_models['aspects_pyabsa_raw']
    df_merged['aspects_pyabsa_agg'] = df_models['aspects_pyabsa_agg']
    
    # Save evaluation data
    output_path = Path(config['data']['processed_dir']) / '06_evaluation.parquet'
    df_merged.to_parquet(output_path, index=False)
    logger.info(f"Saved evaluation data to {output_path}")
    
    # Summary
    logger.info(f"\nS6 Summary:")
    logger.info(f"  VADER vs Stars F1: {vader_metrics['f1']:.3f}")
    logger.info(f"  SiEBERT vs Stars F1: {siebert_metrics['f1']:.3f}")
    logger.info(f"  Aspect comparison: {len(aspect_comparison)} aspects analyzed")
    
    return df_merged


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    df = run_s6_evaluate()
    print(f"\nEvaluation complete. Shape: {df.shape}")

