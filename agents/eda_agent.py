"""
Exploratory Data Analysis Agent
Generates statistical summaries and basic visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, Any


class EDAAgent:
    """Agent responsible for exploratory data analysis"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate descriptive statistics"""
        stats = {
            'total_reviews': len(df),
            'rating_mean': df['rating'].mean(),
            'rating_median': df['rating'].median(),
            'rating_mode': df['rating'].mode()[0] if len(df['rating'].mode()) > 0 else None,
            'rating_std': df['rating'].std(),
            'avg_text_length': df['text_length'].mean(),
            'min_text_length': df['text_length'].min(),
            'max_text_length': df['text_length'].max(),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict()
        }
        return stats
    
    def extract_word_frequencies(self, df: pd.DataFrame, top_n: int = 30) -> Dict[str, int]:
        """Extract most common words from reviews"""
        # Combine all review text
        all_text = ' '.join(df['clean_text'].values).lower()
        
        # Simple word tokenization (remove common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'is', 'it', 'that', 'this', 'with', 'was', 'as', 'be', 'by',
                     'from', 'has', 'have', 'had', 'not', 'are', 'i', 'my', 'me', 'you',
                     'your', 'so', 'if', 'out', 'up', 'no', 'just', 'about', 'can', 'get',
                     'all', 'when', 'there', 'been', 'more', 'very', 'them', 'than', 'much'}
        
        words = [word.strip('.,!?";:()[]{}') for word in all_text.split() 
                if len(word) > 3 and word.lower() not in stop_words]
        
        word_counts = Counter(words)
        return dict(word_counts.most_common(top_n))
    
    def analyze_temporal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze review volume over time"""
        if 'parsed_date' in df.columns:
            df_temp = df.copy()
            df_temp['parsed_date'] = pd.to_datetime(df_temp['parsed_date'])
            
            # Group by date
            temporal = df_temp.groupby('parsed_date').agg({
                'rating': ['count', 'mean']
            }).reset_index()
            
            temporal.columns = ['date', 'review_count', 'avg_rating']
            
            return {
                'has_temporal_data': True,
                'temporal_df': temporal,
                'date_range': {
                    'start': df_temp['parsed_date'].min().strftime('%Y-%m-%d'),
                    'end': df_temp['parsed_date'].max().strftime('%Y-%m-%d')
                }
            }
        else:
            return {'has_temporal_data': False}
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Generate statistics
            stats = self.generate_statistics(df)
            
            # Extract word frequencies
            word_freq = self.extract_word_frequencies(df)
            
            # Temporal analysis
            temporal = self.analyze_temporal_trends(df)
            
            # Store results
            self.analysis_results = {
                'statistics': stats,
                'word_frequencies': word_freq,
                'temporal_analysis': temporal
            }
            
            return {
                'status': 'success',
                'message': f'EDA completed on {len(df)} reviews',
                'results': self.analysis_results
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error performing EDA: {str(e)}',
                'results': None
            }
    
    def visualize_rating_distribution(self, df: pd.DataFrame, save_path: str = None):
        """Create rating distribution visualization"""
        plt.figure(figsize=(10, 6))
        
        rating_counts = df['rating'].value_counts().sort_index()
        
        plt.subplot(1, 2, 1)
        plt.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='navy')
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Number of Reviews', fontsize=12)
        plt.title('Rating Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('viridis', len(rating_counts)))
        plt.title('Rating Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def get_tool(self):
        """Return tool interface for EDA"""
        return {
            'name': 'perform_exploratory_analysis',
            'func': self.perform_eda,
            'description': 'Performs exploratory data analysis on review data.'
        }


def create_eda_tool(df: pd.DataFrame = None):
    """
    Factory function to create EDA tool
    
    Args:
        df: Optional dataframe to analyze
        
    Returns:
        EDAAgent instance
    """
    agent = EDAAgent()
    
    if df is not None:
        result = agent.perform_eda(df)
        if result['status'] == 'success':
            print(f"EDA: {result['message']}")
            stats = result['results']['statistics']
            print(f"Average rating: {stats['rating_mean']:.2f} (±{stats['rating_std']:.2f})")
            print(f"Most common rating: {stats['rating_mode']}")
    
    return agent

