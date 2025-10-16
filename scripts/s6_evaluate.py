"""
S6: Reconciliation & Evaluation (LLM vs Baselines)
Compares LLM-based and dictionary-based aspect detection.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import load_config, setup_logging

def compare_aspect_detection(config):
    """
    Compare LLM vs dictionary aspect detection.
    
    Returns:
        DataFrame with evaluation metrics
    """
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("S6: RECONCILIATION & EVALUATION")
    logger.info("=" * 80)
    
    # Load data
    baselines_path = Path(config['data']['processed_dir']) / '04_baselines.parquet'
    llm_aspects_path = Path(config['data']['processed_dir']) / '05_llm_aspects.parquet'
    
    logger.info(f"Loading baseline data from {baselines_path}")
    df_baseline = pd.read_parquet(baselines_path)
    
    # Check if LLM aspects exist
    if not llm_aspects_path.exists():
        logger.warning("LLM aspects not found. Skipping evaluation.")
        return None
    
    logger.info(f"Loading LLM aspects from {llm_aspects_path}")
    df_llm = pd.read_parquet(llm_aspects_path)
    
    # Parse LLM aspects
    logger.info("Parsing LLM aspect detections...")
    
    aspect_catalog = config['config']['aspects']['catalog']
    
    # Create binary columns for LLM aspects
    for aspect in aspect_catalog:
        df_llm[f'llm_aspect_{aspect}'] = False
    
    for idx, row in df_llm.iterrows():
        try:
            aspects_data = json.loads(row['aspects_json'])
            for aspect_entry in aspects_data.get('aspects', []):
                aspect_name = aspect_entry.get('name', '')
                if aspect_name in aspect_catalog:
                    df_llm.loc[idx, f'llm_aspect_{aspect_name}'] = True
        except:
            pass
    
    # Merge with baseline data (on review index)
    # For simplicity, we'll compare on the subset where we have both
    logger.info("Computing agreement metrics...")
    
    evaluation_results = []
    
    for aspect in aspect_catalog:
        dict_col = f'aspect_{aspect}'
        llm_col = f'llm_aspect_{aspect}'
        
        if dict_col not in df_baseline.columns:
            continue
        
        # Get overlapping indices
        common_indices = df_llm['review_idx'].values
        if len(common_indices) == 0:
            continue
        
        # Filter to common reviews
        baseline_subset = df_baseline.loc[common_indices]
        llm_subset = df_llm.set_index('review_idx').loc[common_indices]
        
        dict_labels = baseline_subset[dict_col].values
        llm_labels = llm_subset[llm_col].values
        
        # Compute metrics (treating dictionary as "ground truth" for comparison)
        if dict_labels.sum() > 0:  # Only if dictionary found this aspect
            precision = precision_score(dict_labels, llm_labels, zero_division=0)
            recall = recall_score(dict_labels, llm_labels, zero_division=0)
            f1 = f1_score(dict_labels, llm_labels, zero_division=0)
            
            # Agreement rate
            agreement = (dict_labels == llm_labels).mean()
            
            evaluation_results.append({
                'aspect': aspect,
                'dict_count': dict_labels.sum(),
                'llm_count': llm_labels.sum(),
                'agreement_rate': agreement,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'comparison': 'LLM vs Dictionary'
            })
    
    eval_df = pd.DataFrame(evaluation_results)
    
    # Display results
    logger.info("\nEvaluation Results (LLM vs Dictionary):")
    logger.info("=" * 80)
    for _, row in eval_df.iterrows():
        logger.info(f"\n{row['aspect']}:")
        logger.info(f"  Dictionary detections: {row['dict_count']:.0f}")
        logger.info(f"  LLM detections: {row['llm_count']:.0f}")
        logger.info(f"  Agreement: {row['agreement_rate']:.3f}")
        logger.info(f"  Precision: {row['precision']:.3f}")
        logger.info(f"  Recall: {row['recall']:.3f}")
        logger.info(f"  F1: {row['f1']:.3f}")
    
    # Save evaluation results
    output_path = Path(config['data']['processed_dir']) / '06_evaluation.parquet'
    eval_df.to_parquet(output_path, index=False)
    
    tables_dir = Path(config['data']['outputs_dir']) / 'tables'
    eval_df.to_csv(tables_dir / 'evaluation_metrics.csv', index=False)
    
    logger.info(f"\nSaved evaluation to {output_path}")
    logger.info("✓ S6 completed successfully")
    
    return eval_df

if __name__ == "__main__":
    config = load_config()
    result = compare_aspect_detection(config)
    if result is not None:
        print(f"\n✓ Stage 6 complete. Evaluated {len(result)} aspects.")
    else:
        print("\n⚠ Stage 6 skipped (LLM data not available).")

