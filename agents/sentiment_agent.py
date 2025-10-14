"""
Sentiment Analysis Agent
Analyzes sentiment polarity and emotions in reviews
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Will use VADER for sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False


class SentimentAgent:
    """Agent responsible for sentiment analysis"""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
        self.sentiment_data = None
        
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        if not self.vader:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 0.0, 'neg': 0.0}
        
        scores = self.vader.polarity_scores(text)
        return scores
    
    def classify_sentiment(self, compound_score: float) -> str:
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_sentiments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform sentiment analysis on all reviews
        
        Args:
            df: Cleaned dataframe with review text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            df_sentiment = df.copy()
            
            # Analyze each review
            sentiment_scores = []
            for text in df_sentiment['clean_text']:
                scores = self.analyze_sentiment_vader(text)
                sentiment_scores.append(scores)
            
            # Add sentiment columns
            df_sentiment['sentiment_compound'] = [s['compound'] for s in sentiment_scores]
            df_sentiment['sentiment_pos'] = [s['pos'] for s in sentiment_scores]
            df_sentiment['sentiment_neu'] = [s['neu'] for s in sentiment_scores]
            df_sentiment['sentiment_neg'] = [s['neg'] for s in sentiment_scores]
            
            # Classify sentiment
            df_sentiment['sentiment_label'] = df_sentiment['sentiment_compound'].apply(
                self.classify_sentiment
            )
            
            # Calculate aggregate metrics
            sentiment_distribution = df_sentiment['sentiment_label'].value_counts().to_dict()
            
            avg_sentiment = {
                'compound': df_sentiment['sentiment_compound'].mean(),
                'positive': df_sentiment['sentiment_pos'].mean(),
                'neutral': df_sentiment['sentiment_neu'].mean(),
                'negative': df_sentiment['sentiment_neg'].mean()
            }
            
            # Sentiment by rating
            sentiment_by_rating = df_sentiment.groupby('rating')['sentiment_compound'].agg([
                'mean', 'std', 'count'
            ]).to_dict('index')
            
            self.sentiment_data = df_sentiment
            
            return {
                'status': 'success',
                'message': f'Sentiment analysis completed on {len(df_sentiment)} reviews',
                'data': df_sentiment,
                'results': {
                    'sentiment_distribution': sentiment_distribution,
                    'average_sentiment': avg_sentiment,
                    'sentiment_by_rating': sentiment_by_rating,
                    'most_positive_reviews': df_sentiment.nlargest(5, 'sentiment_compound')[
                        ['clean_text', 'rating', 'sentiment_compound']
                    ].to_dict('records'),
                    'most_negative_reviews': df_sentiment.nsmallest(5, 'sentiment_compound')[
                        ['clean_text', 'rating', 'sentiment_compound']
                    ].to_dict('records')
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error analyzing sentiment: {str(e)}',
                'data': None
            }
    
    def get_tool(self):
        """Return tool interface for sentiment analysis"""
        return {
            'name': 'analyze_sentiment',
            'func': self.analyze_sentiments,
            'description': 'Analyzes sentiment of reviews using VADER.'
        }


def create_sentiment_tool(df: pd.DataFrame = None):
    """
    Factory function to create sentiment analysis tool
    
    Args:
        df: Optional dataframe to analyze
        
    Returns:
        SentimentAgent instance
    """
    agent = SentimentAgent()
    
    if df is not None:
        result = agent.analyze_sentiments(df)
        if result['status'] == 'success':
            print(f"Sentiment Analysis: {result['message']}")
            dist = result['results']['sentiment_distribution']
            total = sum(dist.values())
            for sentiment, count in dist.items():
                print(f"  {sentiment.capitalize()}: {count} ({count/total*100:.1f}%)")
    
    return agent

