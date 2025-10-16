"""
S7: Final EDA & Storytelling Assets
Generates comprehensive visualizations and tables for storytelling.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def create_aspect_comparison_chart(config):
    """Create comparison chart of aspect negative shares (LLM vs Dictionary)."""
    logger = setup_logging(config)
    
    # Load data
    baselines_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    df = pd.read_parquet(baselines_path)
    
    # Compute negative share for each aspect (dictionary)
    aspect_cols = [col for col in df.columns if col.startswith('aspect_')]
    aspect_names = [col.replace('aspect_', '') for col in aspect_cols]
    
    dict_neg_shares = []
    for aspect_col, aspect_name in zip(aspect_cols, aspect_names):
        aspect_df = df[df[aspect_col]]
        if len(aspect_df) > 0:
            neg_share = (aspect_df['stars_num'] <= 3).sum() / len(aspect_df) * 100
            dict_neg_shares.append({
                'aspect': aspect_name,
                'neg_share_dict': neg_share,
                'method': 'Dictionary'
            })
    
    dict_df = pd.DataFrame(dict_neg_shares)
    
    # Try to load LLM data
    llm_aspects_path = Path(config['data']['processed_dir']) / '05_llm_aspects.parquet'
    
    if llm_aspects_path.exists():
        df_llm = pd.read_parquet(llm_aspects_path)
        
        # Merge with baseline to get stars
        common_indices = df_llm['review_idx'].values
        baseline_subset = df.loc[common_indices]
        
        llm_neg_shares = []
        for aspect in aspect_names:
            llm_col = f'llm_aspect_{aspect}'
            
            # Create LLM aspect columns if needed
            if llm_col not in df_llm.columns:
                df_llm[llm_col] = False
                for idx, row in df_llm.iterrows():
                    try:
                        aspects_data = json.loads(row['aspects_json'])
                        for aspect_entry in aspects_data.get('aspects', []):
                            if aspect_entry.get('name', '') == aspect:
                                df_llm.loc[idx, llm_col] = True
                    except:
                        pass
            
            llm_subset = df_llm[df_llm[llm_col]]
            if len(llm_subset) > 0:
                # Get corresponding stars
                llm_indices = llm_subset['review_idx'].values
                stars = baseline_subset.loc[llm_indices, 'stars_num']
                neg_share = (stars <= 3).sum() / len(stars) * 100
                
                llm_neg_shares.append({
                    'aspect': aspect,
                    'neg_share_llm': neg_share,
                    'method': 'LLM'
                })
        
        llm_df = pd.DataFrame(llm_neg_shares)
        
        # Merge
        comparison_df = dict_df.merge(llm_df, on='aspect', how='outer')
    else:
        comparison_df = dict_df
        comparison_df['neg_share_llm'] = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    dict_vals = comparison_df['neg_share_dict'].fillna(0)
    llm_vals = comparison_df['neg_share_llm'].fillna(0)
    
    ax.bar(x - width/2, dict_vals, width, label='Dictionary', alpha=0.8, color='steelblue')
    if not comparison_df['neg_share_llm'].isna().all():
        ax.bar(x + width/2, llm_vals, width, label='LLM', alpha=0.8, color='coral')
    
    ax.set_xlabel('Aspect', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Negative Reviews (≤3 stars)', fontsize=12, fontweight='bold')
    ax.set_title('Aspect Negative Share: LLM vs Dictionary Detection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['aspect'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = Path(config['data']['outputs_dir']) / 'figures'
    output_path = figures_dir / 'aspect_negative_share_llm_vs_dict.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")
    
    # Save comparison table
    tables_dir = Path(config['data']['outputs_dir']) / 'tables'
    comparison_df.to_csv(tables_dir / 'aspects_summary_llm_vs_dict.csv', index=False)

def create_topic_visualization(config):
    """Create visualization of topic clusters with labels."""
    logger = setup_logging(config)
    
    # Load data
    baselines_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    df = pd.read_parquet(baselines_path)
    
    if 'topic_cluster' not in df.columns:
        logger.warning("No topic clusters found. Skipping topic visualization.")
        return
    
    # Load topic labels
    tables_dir = Path(config['data']['outputs_dir']) / 'tables'
    topics_path = tables_dir / 'topics_summary_labeled.csv'
    
    if topics_path.exists():
        topics_df = pd.read_csv(topics_path)
    else:
        logger.warning("Topic labels not found. Skipping topic visualization.")
        return
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    
    cluster_counts = df['topic_cluster'].value_counts().sort_index()
    
    # Get labels
    labels = []
    for cluster_id in cluster_counts.index:
        topic_row = topics_df[topics_df['cluster_id'] == cluster_id]
        if len(topic_row) > 0 and 'label' in topic_row.columns:
            label = topic_row.iloc[0]['label']
        else:
            label = f"Topic {cluster_id}"
        labels.append(f"{label}\n(n={cluster_counts[cluster_id]})")
    
    ax.barh(range(len(cluster_counts)), cluster_counts.values, color='teal', alpha=0.7)
    ax.set_yticks(range(len(cluster_counts)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Reviews', fontsize=12, fontweight='bold')
    ax.set_title('Topic Clusters with LLM-Generated Labels', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    figures_dir = Path(config['data']['outputs_dir']) / 'figures'
    output_path = figures_dir / 'topic_clusters_with_llm_labels.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")

def create_keyword_spikes_chart(config):
    """Create chart showing keyword mention spikes over time."""
    logger = setup_logging(config)
    
    # Load data
    baselines_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    df = pd.read_parquet(baselines_path)
    
    # Track key aspects over time
    key_aspects = ['Battery', 'Performance/Overheating', 'Camera']
    
    monthly_aspect_data = []
    
    for aspect in key_aspects:
        aspect_col = f'aspect_{aspect}'
        if aspect_col in df.columns:
            monthly_counts = df.groupby('month_start')[aspect_col].sum()
            monthly_total = df.groupby('month_start').size()
            monthly_pct = (monthly_counts / monthly_total * 100).reset_index()
            monthly_pct.columns = ['month_start', 'pct']
            monthly_pct['aspect'] = aspect
            monthly_aspect_data.append(monthly_pct)
    
    if not monthly_aspect_data:
        logger.warning("No aspect data for keyword spikes. Skipping.")
        return
    
    aspect_trend_df = pd.concat(monthly_aspect_data)
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for aspect in key_aspects:
        aspect_data = aspect_trend_df[aspect_trend_df['aspect'] == aspect]
        ax.plot(aspect_data['month_start'], aspect_data['pct'], 'o-', 
                label=aspect, linewidth=2, markersize=8)
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of Reviews Mentioning Aspect', fontsize=12, fontweight='bold')
    ax.set_title('Keyword Mention Spikes: Battery, Overheating, Camera', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    figures_dir = Path(config['data']['outputs_dir']) / 'figures'
    output_path = figures_dir / 'keyword_spikes_battery_overheat_camera.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path}")

def final_storytelling_assets(config):
    """
    Pipeline Stage 7: Final EDA & Storytelling Assets
    
    Generates all required figures and tables for the report.
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S7: FINAL EDA & STORYTELLING ASSETS")
    logger.info("=" * 80)
    
    # Create all required visualizations
    logger.info("\n1. Creating aspect comparison chart...")
    create_aspect_comparison_chart(config)
    
    logger.info("\n2. Creating topic visualization...")
    create_topic_visualization(config)
    
    logger.info("\n3. Creating keyword spikes chart...")
    create_keyword_spikes_chart(config)
    
    logger.info("\n✓ S7 completed successfully")
    logger.info("All required figures and tables have been generated.")

if __name__ == "__main__":
    config = load_config()
    final_storytelling_assets(config)
    print("\n✓ Stage 7 complete. Generated all storytelling assets.")

