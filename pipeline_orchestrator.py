"""
Main Pipeline Orchestrator
Coordinates all agents in the LangChain review analysis pipeline
"""

import os
import sys
from typing import Dict, Any, Optional
import pandas as pd

# Import all agents
from agents.data_ingestion_agent import DataIngestionAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.eda_agent import EDAAgent
from agents.sentiment_agent import SentimentAgent
from agents.topic_agent import TopicAgent
from agents.insight_agent import InsightAgent
from agents.visualization_agent import VisualizationAgent
from agents.report_agent import ReportAgent


class ReviewAnalysisPipeline:
    """
    Orchestrates the complete review analysis pipeline using LangChain agents
    """
    
    def __init__(self, use_llm: bool = False, llm=None):
        """
        Initialize the pipeline with all agents
        
        Args:
            use_llm: Whether to use LLM for advanced reasoning
            llm: Optional LLM instance
        """
        self.use_llm = use_llm
        self.llm = llm
        
        # Initialize agents
        self.ingestion_agent = DataIngestionAgent(llm=llm)
        self.cleaning_agent = DataCleaningAgent()
        self.eda_agent = EDAAgent()
        self.sentiment_agent = SentimentAgent()
        self.topic_agent = TopicAgent()
        self.insight_agent = InsightAgent(llm=llm, use_llm=use_llm)
        self.visualization_agent = VisualizationAgent()
        self.report_agent = ReportAgent()
        
        # Store results from each stage
        self.results = {
            'ingestion': None,
            'cleaning': None,
            'eda': None,
            'sentiment': None,
            'topics': None,
            'insights': None,
            'visualizations': None,
            'report': None
        }
        
        self.data = None
        
    def run_full_pipeline(self, csv_path: str, output_dir: str = './output') -> Dict[str, Any]:
        """
        Execute the complete analysis pipeline
        
        Args:
            csv_path: Path to the review CSV file
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("iPhone 17 Pro Max Review Analysis Pipeline")
        print("="*80)
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: Data Ingestion
        print("📥 Stage 1: Data Ingestion")
        print("-" * 40)
        ingestion_result = self.ingestion_agent.load_data(csv_path)
        
        if ingestion_result['status'] != 'success':
            print(f"❌ Error: {ingestion_result['message']}")
            return {'status': 'error', 'stage': 'ingestion', 'message': ingestion_result['message']}
        
        print(f"✓ {ingestion_result['message']}")
        self.results['ingestion'] = ingestion_result
        print()
        
        # Stage 2: Data Cleaning
        print("🧹 Stage 2: Data Cleaning")
        print("-" * 40)
        cleaning_result = self.cleaning_agent.clean_data(ingestion_result['data'])
        
        if cleaning_result['status'] != 'success':
            print(f"❌ Error: {cleaning_result['message']}")
            return {'status': 'error', 'stage': 'cleaning', 'message': cleaning_result['message']}
        
        print(f"✓ {cleaning_result['message']}")
        print(f"  Average text length: {cleaning_result['stats']['avg_text_length']:.0f} characters")
        self.results['cleaning'] = cleaning_result
        self.data = cleaning_result['data']
        print()
        
        # Stage 3: Exploratory Data Analysis
        print("📊 Stage 3: Exploratory Data Analysis")
        print("-" * 40)
        eda_result = self.eda_agent.perform_eda(self.data)
        
        if eda_result['status'] != 'success':
            print(f"❌ Error: {eda_result['message']}")
            return {'status': 'error', 'stage': 'eda', 'message': eda_result['message']}
        
        stats = eda_result['results']['statistics']
        print(f"✓ {eda_result['message']}")
        print(f"  Average rating: {stats['rating_mean']:.2f}/5.0 (±{stats['rating_std']:.2f})")
        print(f"  Most common rating: {stats['rating_mode']}★")
        print(f"  Rating distribution: {stats['rating_distribution']}")
        self.results['eda'] = eda_result['results']
        print()
        
        # Stage 4: Sentiment Analysis
        print("😊 Stage 4: Sentiment Analysis")
        print("-" * 40)
        sentiment_result = self.sentiment_agent.analyze_sentiments(self.data)
        
        if sentiment_result['status'] != 'success':
            print(f"❌ Error: {sentiment_result['message']}")
            return {'status': 'error', 'stage': 'sentiment', 'message': sentiment_result['message']}
        
        print(f"✓ {sentiment_result['message']}")
        sentiment_dist = sentiment_result['results']['sentiment_distribution']
        total = sum(sentiment_dist.values())
        for sentiment_label, count in sentiment_dist.items():
            print(f"  {sentiment_label.capitalize()}: {count} ({count/total*100:.1f}%)")
        
        self.results['sentiment'] = sentiment_result['results']
        self.data = sentiment_result['data']  # Update with sentiment columns
        print()
        
        # Stage 5: Topic Modeling
        print("🔍 Stage 5: Topic Extraction")
        print("-" * 40)
        topic_result = self.topic_agent.extract_topics(self.data)
        
        if topic_result['status'] != 'success':
            print(f"❌ Error: {topic_result['message']}")
            return {'status': 'error', 'stage': 'topics', 'message': topic_result['message']}
        
        print(f"✓ {topic_result['message']}")
        print(f"  Identified {topic_result['results']['total_aspects_identified']} key aspects")
        print(f"  Found {len(topic_result['results']['pain_points'])} pain points")
        print(f"  Found {len(topic_result['results']['opportunities'])} strengths")
        
        if topic_result['results']['pain_points']:
            print("\n  Top pain points:")
            for pp in topic_result['results']['pain_points'][:3]:
                print(f"    - {pp['aspect'].replace('_', ' ').title()}: "
                      f"{pp['mentions']} mentions, {pp['negative_ratio']*100:.1f}% negative")
        
        if topic_result['results']['opportunities']:
            print("\n  Top strengths:")
            for opp in topic_result['results']['opportunities'][:3]:
                print(f"    - {opp['aspect'].replace('_', ' ').title()}: "
                      f"{opp['mentions']} mentions, {opp['positive_ratio']*100:.1f}% positive")
        
        self.results['topics'] = topic_result['results']
        self.data = topic_result['data']  # Update with topic columns
        print()
        
        # Stage 6: Insight Generation
        print("💡 Stage 6: Insight Generation")
        print("-" * 40)
        insight_result = self.insight_agent.generate_insights(
            self.results['topics'],
            self.results['sentiment'],
            self.results['eda']
        )
        
        if insight_result['status'] != 'success':
            print(f"❌ Error: {insight_result['message']}")
            return {'status': 'error', 'stage': 'insights', 'message': insight_result['message']}
        
        print(f"✓ {insight_result['message']}")
        print("\n  Key Recommendations:")
        for i, rec in enumerate(insight_result['results']['recommendations'][:3], 1):
            print(f"    {i}. {rec['title']} [{rec.get('priority', 'MEDIUM')} Priority]")
        
        self.results['insights'] = insight_result['results']
        print()
        
        # Stage 7: Visualization
        print("📈 Stage 7: Creating Visualizations")
        print("-" * 40)
        temporal_data = self.results['eda']['temporal_analysis'].get('temporal_df')
        
        viz_result = self.visualization_agent.create_all_visualizations(
            self.data,
            self.results['topics'],
            temporal_data
        )
        
        if viz_result['status'] != 'success':
            print(f"❌ Error: {viz_result['message']}")
            return {'status': 'error', 'stage': 'visualization', 'message': viz_result['message']}
        
        print(f"✓ {viz_result['message']}")
        
        # Save visualizations
        for name, fig in viz_result['figures'].items():
            save_path = os.path.join(output_dir, f'{name}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {name}.png")
        
        self.results['visualizations'] = viz_result
        print()
        
        # Stage 8: Report Generation
        print("📝 Stage 8: Generating Report")
        print("-" * 40)
        report_result = self.report_agent.generate_report(
            self.results['eda'],
            self.results['sentiment'],
            self.results['topics'],
            self.results['insights']
        )
        
        if report_result['status'] != 'success':
            print(f"❌ Error: {report_result['message']}")
            return {'status': 'error', 'stage': 'report', 'message': report_result['message']}
        
        print(f"✓ {report_result['message']}")
        
        # Save report
        report_path = os.path.join(output_dir, 'analysis_report.md')
        self.report_agent.save_report(report_path)
        print(f"  Saved: analysis_report.md")
        
        self.results['report'] = report_result
        print()
        
        # Pipeline Complete
        print("="*80)
        print("✅ Pipeline Execution Complete!")
        print("="*80)
        print(f"\nOutputs saved to: {output_dir}/")
        print(f"  - Analysis report: analysis_report.md")
        print(f"  - Visualizations: {len(viz_result['figures'])} PNG files")
        print()
        
        return {
            'status': 'success',
            'message': 'Pipeline completed successfully',
            'results': self.results,
            'output_directory': output_dir
        }
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the processed dataframe"""
        return self.data
    
    def get_results(self) -> Dict[str, Any]:
        """Return all results"""
        return self.results


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='iPhone Review Analysis Pipeline')
    parser.add_argument('csv_path', help='Path to the review CSV file')
    parser.add_argument('--output-dir', default='./output', help='Output directory for results')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for advanced insights')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = ReviewAnalysisPipeline(use_llm=args.use_llm)
    result = pipeline.run_full_pipeline(args.csv_path, args.output_dir)
    
    if result['status'] == 'success':
        print("\n✓ Analysis complete! Check the output directory for results.")
        sys.exit(0)
    else:
        print(f"\n❌ Pipeline failed at stage: {result.get('stage', 'unknown')}")
        print(f"   Error: {result.get('message', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()

