"""
S5: LangChain Agentic Layer
Uses LLM for advanced aspect tagging and insights generation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import sqlite3
from tqdm import tqdm
import os

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging, text_hash

def setup_llm(config):
    """Set up LangChain LLM."""
    try:
        from langchain_openai import ChatOpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.0,
            api_key=api_key
        )
        
        return llm
    except ImportError:
        raise ImportError("LangChain not installed. Run: pip install langchain langchain-openai")

def aspect_tagger_llm(review_text, llm, cache_conn=None, text_hash_val=None):
    """
    Use LLM to tag aspects in a review with sentiment.
    Returns structured JSON with aspects, sentiments, and evidence spans.
    """
    # Check cache
    if cache_conn and text_hash_val:
        cursor = cache_conn.cursor()
        cursor.execute(
            "SELECT aspects_json FROM llm_aspects WHERE text_hash = ?",
            (text_hash_val,)
        )
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
    
    prompt = f"""Analyze this iPhone review and identify mentioned aspects with their sentiment.

Review: "{review_text}"

Aspects to consider: Battery, Performance/Overheating, Camera, Display, Durability/Build, Charging/USB-C, iOS/Software, Price/Value, Weight/Ergonomics, Other

For each aspect mentioned, provide:
1. Aspect name (from list above)
2. Sentiment (positive, neutral, or negative)
3. Evidence span (short quote from review)

Return a JSON array of aspects. Example:
{{"aspects": [{{"name": "Battery", "sentiment": "positive", "evidence_span": "battery life is fantastic"}}]}}

If no aspects found, return: {{"aspects": []}}

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        aspects_data = json.loads(content)
        
        # Validate schema
        if 'aspects' not in aspects_data:
            aspects_data = {'aspects': []}
        
        # Cache result
        if cache_conn and text_hash_val:
            cursor.execute(
                "INSERT OR REPLACE INTO llm_aspects (text_hash, aspects_json) VALUES (?, ?)",
                (text_hash_val, json.dumps(aspects_data))
            )
            cache_conn.commit()
        
        return aspects_data
        
    except Exception as e:
        # Return empty aspects on error
        return {'aspects': []}

def topic_labeler_llm(cluster_id, top_terms, sample_texts, llm):
    """Use LLM to generate human-readable labels for topic clusters."""
    
    sample_text_str = "\n".join([f"- {text[:150]}..." for text in sample_texts[:5]])
    
    prompt = f"""A topic cluster was created from iPhone 17 Pro Max reviews.

Top terms: {', '.join(top_terms[:10])}

Sample reviews:
{sample_text_str}

Generate:
1. A 2-4 word label for this topic
2. A one-line description

Return JSON: {{"label": "...", "description": "..."}}

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Extract JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        label_data = json.loads(content)
        
        return label_data
        
    except Exception as e:
        return {
            'label': f"Topic {cluster_id}",
            'description': f"Topics related to: {', '.join(top_terms[:5])}"
        }

def insight_writer_llm(summary_stats, llm):
    """Use LLM to generate insights and draft recommendations."""
    
    prompt = f"""You are analyzing iPhone 17 Pro Max user reviews. Below is a summary of key findings:

{summary_stats}

Generate:
1. 6-8 bullet points of key insights
2. 3-5 evidence-backed design recommendations

Format:
## Key Insights
- [insight 1]
- [insight 2]
...

## Draft Recommendations
1. **[Finding]**: [evidence] → **Recommendation**: [actionable change]
2. ...

Markdown:"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"# LLM Insights\n\nError generating insights: {str(e)}"

def agentic_layer(config):
    """
    Pipeline Stage 5: LangChain Agentic Layer
    
    Steps:
    1. Set up LLM and caching
    2. Aspect tagging with LLM
    3. Topic labeling with LLM
    4. Insight generation with LLM
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S5: LANGCHAIN AGENTIC LAYER")
    logger.info("=" * 80)
    
    if not config['config']['agentic']['enabled']:
        logger.info("Agentic layer disabled in config. Skipping S5.")
        return None
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not found. Skipping agentic layer.")
        logger.warning("To enable: Set OPENAI_API_KEY environment variable")
        return None
    
    # Load input data
    features_path = Path(config['data']['processed_dir']) / '03_features.parquet'
    baselines_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    
    logger.info(f"Loading data from {baselines_path}")
    df = pd.read_parquet(baselines_path)
    logger.info(f"Loaded {len(df):,} reviews")
    
    # Set up LLM
    logger.info("Setting up LLM...")
    try:
        llm = setup_llm(config)
        logger.info("✓ LLM initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None
    
    # Set up cache
    cache_dir = Path(config['data']['cache_dir'])
    cache_path = cache_dir / 'llm.sqlite'
    
    cache_conn = sqlite3.connect(cache_path)
    cache_conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_aspects (
            text_hash TEXT PRIMARY KEY,
            aspects_json TEXT
        )
    """)
    cache_conn.commit()
    
    # 1. Aspect Tagging with LLM
    logger.info("\n1. Aspect tagging with LLM...")
    
    # Sample reviews for LLM (cost control)
    non_en_mask = df['lang'] != 'en'
    non_en_sample = df[non_en_mask].copy()
    
    # Sample English reviews (stratified by month and stars)
    en_sample = df[~non_en_mask].groupby(['month_start', 'stars_num']).apply(
        lambda x: x.sample(min(len(x), 50), random_state=42)
    ).reset_index(drop=True)
    
    sample_df = pd.concat([non_en_sample, en_sample]).reset_index(drop=True)
    logger.info(f"Processing {len(sample_df):,} reviews with LLM (cost control)")
    
    # Apply LLM aspect tagging
    llm_aspects_list = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="LLM aspect tagging"):
        if idx >= config['config']['cost_controls']['max_llm_calls']:
            logger.warning(f"Reached max LLM calls limit: {config['config']['cost_controls']['max_llm_calls']}")
            break
        
        aspects_data = aspect_tagger_llm(
            row['text_en_clean'],
            llm,
            cache_conn,
            text_hash(row['text_en_clean'])
        )
        
        llm_aspects_list.append({
            'review_idx': row.name if hasattr(row, 'name') else idx,
            'aspects_json': json.dumps(aspects_data)
        })
    
    llm_aspects_df = pd.DataFrame(llm_aspects_list)
    
    # Save LLM aspects
    output_path = Path(config['data']['processed_dir']) / '05_llm_aspects.parquet'
    llm_aspects_df.to_parquet(output_path, index=False)
    logger.info(f"Saved LLM aspects to {output_path}")
    
    # 2. Topic Labeling with LLM
    logger.info("\n2. Topic labeling with LLM...")
    
    if 'topic_cluster' in df.columns:
        # Read cluster terms from baselines
        tables_dir = Path(config['data']['outputs_dir']) / 'tables'
        cluster_info = pd.read_csv(tables_dir / 'topics_summary_labeled.csv')
        
        topic_labels = []
        for _, row in tqdm(cluster_info.iterrows(), total=len(cluster_info), desc="Topic labeling"):
            cluster_id = row['cluster_id']
            top_terms = row['top_terms'].split(', ')
            
            # Get sample texts from this cluster
            cluster_df = df[df['topic_cluster'] == cluster_id]
            sample_texts = cluster_df['text_en_clean'].sample(min(5, len(cluster_df)), random_state=42).tolist()
            
            label_data = topic_labeler_llm(cluster_id, top_terms, sample_texts, llm)
            
            topic_labels.append({
                'cluster_id': cluster_id,
                'label': label_data.get('label', f'Topic {cluster_id}'),
                'description': label_data.get('description', ''),
                'top_terms': row['top_terms']
            })
        
        topic_labels_df = pd.DataFrame(topic_labels)
        
        # Save topic labels
        output_path = Path(config['data']['processed_dir']) / '05_llm_topics.parquet'
        topic_labels_df.to_parquet(output_path, index=False)
        
        # Also update the CSV
        topic_labels_df.to_csv(tables_dir / 'topics_summary_labeled.csv', index=False)
        logger.info(f"Saved LLM topic labels to {output_path}")
    
    # 3. Insight Generation with LLM
    logger.info("\n3. Generating insights with LLM...")
    
    # Prepare summary stats
    summary_stats = f"""
Total Reviews: {len(df):,}
Date Range: {df['date_parsed'].min()} to {df['date_parsed'].max()}

Star Distribution:
{df['stars_num'].value_counts().sort_index().to_string()}

Top 5 Most Discussed Aspects (Dictionary):
{df[[col for col in df.columns if col.startswith('aspect_')]].sum().sort_values(ascending=False).head().to_string()}

Average Sentiment (VADER): {df['vader_compound'].mean():.3f}

Sample Negative Comments (< 3 stars):
{df[df['stars_num'] < 3]['text_en_clean'].sample(min(5, len(df[df['stars_num'] < 3]))).tolist()}
"""
    
    insights_md = insight_writer_llm(summary_stats, llm)
    
    # Save insights
    output_dir = Path(config['data']['outputs_dir']) / 'llm'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    insights_path = output_dir / 'insights.md'
    with open(insights_path, 'w', encoding='utf-8') as f:
        f.write(insights_md)
    
    logger.info(f"Saved insights to {insights_path}")
    
    # Close cache
    cache_conn.close()
    
    logger.info("✓ S5 completed successfully")
    return llm_aspects_df

if __name__ == "__main__":
    config = load_config()
    result = agentic_layer(config)
    if result is not None:
        print(f"\n✓ Stage 5 complete. Processed LLM analysis.")
    else:
        print("\n⚠ Stage 5 skipped (agentic layer disabled or API key missing).")

