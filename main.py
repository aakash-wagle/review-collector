"""
Main Orchestrator for Beats Solo3 Wireless Review Analysis Pipeline
"""
import sys
import logging
import yaml
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from s1_ingest import run_s1_ingest
from s2_translate import run_s2_translate
from s3_features import run_s3_features
from s4_baselines import run_s4_baselines
from s5_models import run_s5_models
from s6_evaluate import run_s6_evaluate
from s7_story import run_s7_story
from s8_recommendations import run_s8_recommendations


def setup_logging(config_path: str = "config.yaml"):
    """Setup logging configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_file = Path(config['governance']['logging']['save_to'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['governance']['logging']['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_pipeline(config_path: str = "config.yaml", start_from: str = None, end_at: str = None):
    """
    Run the complete analysis pipeline.
    
    Args:
        config_path: Path to configuration file
        start_from: Stage to start from (e.g., 's3' to skip s1 and s2)
        end_at: Stage to end at (e.g., 's6' to stop before s7)
    """
    logger = setup_logging(config_path)
    
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("Starting Beats Solo3 Wireless Review Analysis Pipeline")
    logger.info("="*80)
    
    stages = {
        's1': ('S1: Ingest', run_s1_ingest),
        's2': ('S2: Translation', run_s2_translate),
        's3': ('S3: Features', run_s3_features),
        's4': ('S4: Baselines', run_s4_baselines),
        's5': ('S5: Models', run_s5_models),
        's6': ('S6: Evaluation', run_s6_evaluate),
        's7': ('S7: Storytelling', run_s7_story),
        's8': ('S8: Recommendations', run_s8_recommendations)
    }
    
    stage_names = list(stages.keys())
    
    # Determine which stages to run
    start_idx = stage_names.index(start_from) if start_from else 0
    end_idx = stage_names.index(end_at) if end_at else len(stage_names) - 1
    
    stages_to_run = stage_names[start_idx:end_idx + 1]
    
    logger.info(f"Running stages: {', '.join(stages_to_run)}")
    logger.info("-"*80)
    
    # Execute pipeline
    try:
        for stage_id in stages_to_run:
            stage_name, stage_func = stages[stage_id]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Starting {stage_name}")
            logger.info(f"{'='*80}\n")
            
            stage_start = datetime.now()
            
            # Run stage
            stage_func(config_path)
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            logger.info(f"\n{stage_name} completed in {stage_duration:.2f}s")
        
        # Pipeline complete
        total_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Total execution time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
        logger.info(f"Recommendations: outputs/recommendations.md")
        logger.info(f"Figures: outputs/figures/")
        logger.info(f"Tables: outputs/tables/")
        logger.info(f"All outputs saved to: outputs/")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"PIPELINE FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Beats Solo3 Review Analysis Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--start-from', help='Stage to start from (s1-s8)')
    parser.add_argument('--end-at', help='Stage to end at (s1-s8)')
    
    args = parser.parse_args()
    
    success = run_pipeline(
        config_path=args.config,
        start_from=args.start_from,
        end_at=args.end_at
    )
    
    sys.exit(0 if success else 1)

