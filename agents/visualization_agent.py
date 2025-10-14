"""
Visualization Agent
Creates comprehensive visualizations for review analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

# Try to import wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


class VisualizationAgent:
    """Agent responsible for creating visualizations"""
    
    def __init__(self):
        self.figures = {}
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
    def plot_rating_distribution(self, df: pd.DataFrame, save_path: str = None):
        """Create rating distribution chart"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        rating_counts = df['rating'].value_counts().sort_index()
        axes[0].bar(rating_counts.index, rating_counts.values, 
                   color='steelblue', edgecolor='navy', alpha=0.7)
        axes[0].set_xlabel('Rating (Stars)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        axes[0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for idx, count in zip(rating_counts.index, rating_counts.values):
            axes[0].text(idx, count + max(rating_counts.values)*0.01, 
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        colors = ['#d62728', '#ff7f0e', '#ffdd57', '#90ee90', '#2ca02c']
        axes[1].pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Rating Proportions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        self.figures['rating_distribution'] = fig
        return fig
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, save_path: str = None):
        """Create sentiment distribution chart"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sentiment label distribution
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = {'positive': '#2ca02c', 'neutral': '#ffdd57', 'negative': '#d62728'}
        color_list = [colors.get(label, 'gray') for label in sentiment_counts.index]
        
        axes[0].bar(sentiment_counts.index, sentiment_counts.values, 
                   color=color_list, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add percentages
        total = sum(sentiment_counts.values)
        for idx, (label, count) in enumerate(sentiment_counts.items()):
            axes[0].text(idx, count + max(sentiment_counts.values)*0.01,
                        f'{count/total*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Sentiment compound score distribution
        axes[1].hist(df['sentiment_compound'], bins=30, color='skyblue', 
                    edgecolor='navy', alpha=0.7)
        axes[1].axvline(df['sentiment_compound'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f"Mean: {df['sentiment_compound'].mean():.3f}")
        axes[1].set_xlabel('Sentiment Compound Score', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Sentiment Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        self.figures['sentiment_distribution'] = fig
        return fig
    
    def plot_topic_sentiment_heatmap(self, topic_results: Dict, save_path: str = None):
        """Create topic vs sentiment heatmap"""
        aspect_analysis = topic_results['aspect_analysis']
        
        # Prepare data
        aspects = []
        sentiments = []
        counts = []
        
        for aspect, data in aspect_analysis.items():
            if data['count'] > 0:
                aspects.append(aspect.replace('_', ' ').title())
                sentiments.append(data['avg_sentiment'])
                counts.append(data['count'])
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1][:10]  # Top 10
        aspects = [aspects[i] for i in sorted_indices]
        sentiments = [sentiments[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create horizontal bar chart with sentiment coloring
        colors_map = plt.cm.RdYlGn([(s + 1) / 2 for s in sentiments])  # Normalize -1 to 1 -> 0 to 1
        bars = ax.barh(aspects, counts, color=colors_map, edgecolor='black', alpha=0.8)
        
        ax.set_xlabel('Number of Mentions', fontsize=12, fontweight='bold')
        ax.set_title('Product Aspects: Mentions and Sentiment', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add sentiment scores as text
        for i, (count, sentiment) in enumerate(zip(counts, sentiments)):
            ax.text(count + max(counts)*0.02, i, f'{sentiment:.2f}', 
                   va='center', fontweight='bold', fontsize=10)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Sentiment Score', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        self.figures['topic_sentiment_heatmap'] = fig
        return fig
    
    def create_word_cloud(self, df: pd.DataFrame, sentiment_filter: str = None, 
                         save_path: str = None):
        """Create word cloud from reviews"""
        if not WORDCLOUD_AVAILABLE:
            print("WordCloud library not available")
            return None
        
        # Filter by sentiment if specified
        if sentiment_filter:
            df_filtered = df[df['sentiment_label'] == sentiment_filter]
        else:
            df_filtered = df
        
        # Combine all text
        text = ' '.join(df_filtered['clean_text'].values)
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100,
                             relative_scaling=0.5,
                             min_font_size=10).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        title = 'Word Cloud'
        if sentiment_filter:
            title += f' ({sentiment_filter.capitalize()} Reviews)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        self.figures[f'wordcloud_{sentiment_filter or "all"}'] = fig
        return fig
    
    def plot_temporal_trends(self, temporal_data: pd.DataFrame, save_path: str = None):
        """Create temporal trend visualization"""
        if temporal_data is None or len(temporal_data) == 0:
            print("No temporal data available")
            return None
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Review volume over time
        axes[0].plot(temporal_data['date'], temporal_data['review_count'], 
                    marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[0].fill_between(temporal_data['date'], temporal_data['review_count'], 
                            alpha=0.3, color='steelblue')
        axes[0].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        axes[0].set_title('Review Volume Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Average rating over time
        axes[1].plot(temporal_data['date'], temporal_data['avg_rating'], 
                    marker='s', linewidth=2, markersize=6, color='green')
        axes[1].axhline(temporal_data['avg_rating'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f"Overall Avg: {temporal_data['avg_rating'].mean():.2f}")
        axes[1].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Average Rating', fontsize=12, fontweight='bold')
        axes[1].set_title('Average Rating Over Time', fontsize=14, fontweight='bold')
        axes[1].set_ylim([temporal_data['avg_rating'].min() - 0.2, 5.1])
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        self.figures['temporal_trends'] = fig
        return fig
    
    def create_all_visualizations(self, df: pd.DataFrame, topic_results: Dict, 
                                  temporal_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Create all visualizations
        
        Args:
            df: Dataframe with sentiment data
            topic_results: Topic analysis results
            temporal_data: Optional temporal analysis data
            
        Returns:
            Dictionary with visualization results
        """
        try:
            # Create visualizations
            print("Creating visualizations...")
            
            fig1 = self.plot_rating_distribution(df)
            print("  ✓ Rating distribution")
            
            fig2 = self.plot_sentiment_distribution(df)
            print("  ✓ Sentiment distribution")
            
            fig3 = self.plot_topic_sentiment_heatmap(topic_results)
            print("  ✓ Topic-sentiment heatmap")
            
            if WORDCLOUD_AVAILABLE:
                fig4 = self.create_word_cloud(df)
                print("  ✓ Word cloud")
                
                fig5 = self.create_word_cloud(df, sentiment_filter='positive')
                print("  ✓ Positive word cloud")
                
                fig6 = self.create_word_cloud(df, sentiment_filter='negative')
                print("  ✓ Negative word cloud")
            
            if temporal_data is not None and len(temporal_data) > 0:
                fig7 = self.plot_temporal_trends(temporal_data)
                print("  ✓ Temporal trends")
            
            return {
                'status': 'success',
                'message': f'Created {len(self.figures)} visualizations',
                'figures': self.figures
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error creating visualizations: {str(e)}',
                'figures': {}
            }
    
    def get_tool(self):
        """Return tool interface for visualization"""
        return {
            'name': 'create_visualizations',
            'func': lambda x: self.create_all_visualizations(x.get('df'), x.get('topics'), x.get('temporal')),
            'description': 'Creates comprehensive visualizations.'
        }


def create_visualization_tool(df=None, topic_results=None, temporal_data=None):
    """
    Factory function to create visualization tool
    
    Args:
        df: Dataframe with sentiment data
        topic_results: Topic analysis results
        temporal_data: Optional temporal data
        
    Returns:
        VisualizationAgent instance
    """
    agent = VisualizationAgent()
    
    if df is not None and topic_results is not None:
        result = agent.create_all_visualizations(df, topic_results, temporal_data)
        if result['status'] == 'success':
            print(f"\nVisualization: {result['message']}")
    
    return agent

