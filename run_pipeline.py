"""
Main Pipeline Runner
Orchestrates all stages of the iPhone 17 Pro Max review analysis pipeline.
"""
import sys
from pathlib import Path
import time
from datetime import datetime

# Add scripts to path
sys.path.append(str(Path(__file__).parent))

from scripts.utils import load_config, setup_logging

def run_full_pipeline():
    """Execute all pipeline stages in sequence."""
    
    config = load_config()
    logger = setup_logging(config)
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("IPHONE 17 PRO MAX REVIEWS - MULTILINGUAL ANALYSIS PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    stages = [
        ("S1", "Ingest & Canonicalize", "scripts.s1_ingest", "ingest_and_canonicalize"),
        ("S2", "Language Detection & Translation", "scripts.s2_translate", "language_detection_and_translation"),
        ("S3", "Text Normalization & Features", "scripts.s3_features", "text_normalization_and_features"),
        ("S4", "Deterministic Baselines", "scripts.s4_baselines", "deterministic_baselines"),
        ("S5", "LangChain Agentic Layer", "scripts.s5_agentic", "agentic_layer"),
        ("S6", "Evaluation & Reconciliation", "scripts.s6_evaluate", "compare_aspect_detection"),
        ("S7", "Final EDA & Storytelling", "scripts.s7_storytelling", "final_storytelling_assets"),
        ("S8", "Evidence-Backed Recommendations", "scripts.s8_recommendations", "generate_recommendations"),
        ("S9", "Two-Page Report", "scripts.s9_report", "generate_pdf_report"),
    ]
    
    results = {}
    
    for stage_id, stage_name, module_name, function_name in stages:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Starting {stage_id}: {stage_name}")
        logger.info(f"{'=' * 80}")
        
        try:
            # Import module dynamically
            module = __import__(module_name, fromlist=[function_name])
            stage_function = getattr(module, function_name)
            
            # Execute stage
            stage_start = time.time()
            result = stage_function(config)
            stage_duration = time.time() - stage_start
            
            results[stage_id] = {
                'status': 'success',
                'duration': stage_duration,
                'result': result
            }
            
            logger.info(f"✓ {stage_id} completed in {stage_duration:.2f}s")
            
        except Exception as e:
            stage_duration = time.time() - stage_start if 'stage_start' in locals() else 0
            results[stage_id] = {
                'status': 'failed',
                'duration': stage_duration,
                'error': str(e)
            }
            
            logger.error(f"✗ {stage_id} failed: {e}")
            
            # Continue with next stage if this is optional (S5, S6)
            if stage_id in ['S5', 'S6']:
                logger.warning(f"Continuing pipeline despite {stage_id} failure (optional stage)")
            else:
                logger.error("Critical stage failed. Stopping pipeline.")
                break
    
    # Summary
    total_duration = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    for stage_id, result in results.items():
        status_icon = "✓" if result['status'] == 'success' else "✗"
        logger.info(f"{status_icon} {stage_id}: {result['status']} ({result['duration']:.2f}s)")
        if result['status'] == 'failed':
            logger.info(f"    Error: {result['error']}")
    
    logger.info(f"\nTotal pipeline duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check completion
    critical_stages = ['S1', 'S2', 'S3', 'S4', 'S7', 'S8', 'S9']
    critical_success = all(
        results.get(stage, {}).get('status') == 'success' 
        for stage in critical_stages
    )
    
    if critical_success:
        logger.info("\n✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("All critical stages passed. Report available at: REPORT_2PAGES.pdf")
        return True
    else:
        logger.warning("\n⚠ PIPELINE COMPLETED WITH WARNINGS")
        logger.warning("Some stages failed. Check logs for details.")
        return False

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n" + "=" * 80)
    print("iPhone 17 Pro Max Reviews - Multilingual Analysis Pipeline")
    print("=" * 80)
    print("\nStarting pipeline execution...\n")
    
    success = run_full_pipeline()
    
    if success:
        print("\n✓ Pipeline completed successfully!")
        print("\nOutputs:")
        print("  - Report: REPORT_2PAGES.pdf")
        print("  - Figures: outputs/figures/")
        print("  - Tables: outputs/tables/")
        print("  - Recommendations: outputs/recommendations.md")
        print("  - Logs: outputs/run_logs.txt")
    else:
        print("\n⚠ Pipeline completed with some failures. Check outputs/run_logs.txt for details.")
    
    sys.exit(0 if success else 1)

