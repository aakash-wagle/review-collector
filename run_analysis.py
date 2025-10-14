#!/usr/bin/env python3
"""
iPhone 17 Pro Max Review Analysis - Main Execution Script
Complete LangChain-based analytical pipeline
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_orchestrator import ReviewAnalysisPipeline


def main():
    """Main execution function"""
    
    # Configuration
    csv_path = 'iphone_17_pro_max_reviews.csv'
    output_dir = './analysis_output'
    use_llm = False  # Set to True to use OpenAI for advanced insights
    
    print("\n" + "="*80)
    print(" 🍎 iPhone 17 Pro Max Review Analysis Pipeline")
    print(" LangChain-Based Automated Analysis System")
    print("="*80 + "\n")
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found: {csv_path}")
        print(f"\nPlease ensure '{csv_path}' is in the current directory.")
        return 1
    
    # Initialize pipeline
    print("📋 Initializing pipeline with 8 LangChain agents...")
    print("   1. Data Ingestion Agent")
    print("   2. Data Cleaning Agent")
    print("   3. EDA Agent")
    print("   4. Sentiment Analysis Agent")
    print("   5. Topic Extraction Agent")
    print("   6. Insight Generation Agent")
    print("   7. Visualization Agent")
    print("   8. Report Generation Agent")
    print()
    
    pipeline = ReviewAnalysisPipeline(use_llm=use_llm)
    
    # Run pipeline
    result = pipeline.run_full_pipeline(csv_path, output_dir)
    
    if result['status'] == 'success':
        print("\n" + "="*80)
        print(" ✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 All outputs saved to: {output_dir}/")
        print("\n📄 Generated Files:")
        print("   - analysis_report.md (comprehensive report)")
        print("   - rating_distribution.png")
        print("   - sentiment_distribution.png")
        print("   - topic_sentiment_heatmap.png")
        print("   - wordcloud_all.png (if WordCloud available)")
        print("   - wordcloud_positive.png (if WordCloud available)")
        print("   - wordcloud_negative.png (if WordCloud available)")
        print("\n💡 Next Steps:")
        print("   1. Read the comprehensive report: analysis_report.md")
        print("   2. Review visualizations for insights")
        print("   3. Consider implementing design recommendations")
        print()
        return 0
    else:
        print("\n" + "="*80)
        print(" ❌ PIPELINE FAILED")
        print("="*80)
        print(f"\n   Stage: {result.get('stage', 'unknown')}")
        print(f"   Error: {result.get('message', 'Unknown error')}")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())

