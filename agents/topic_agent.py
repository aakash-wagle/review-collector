"""
Topic Modeling and Keyword Extraction Agent
Extracts key themes and topics from reviews
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
from typing import Dict, Any, List


class TopicAgent:
    """Agent responsible for topic extraction and keyword analysis"""
    
    def __init__(self):
        self.topic_data = None
        self.topics = {}
        
        # Define product aspect keywords
        self.aspect_keywords = {
            'battery': ['battery', 'charge', 'charging', 'power', 'drain', 'last', 'life'],
            'camera': ['camera', 'photo', 'picture', 'image', 'video', 'zoom', 'lens'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processor', 'gaming', 'game'],
            'design': ['design', 'look', 'color', 'orange', 'titanium', 'feel', 'aesthetic', 'beautiful'],
            'display': ['screen', 'display', 'bright', 'sunlight', 'resolution'],
            'build_quality': ['quality', 'build', 'solid', 'premium', 'durable', 'scratch', 'bend'],
            'price': ['price', 'expensive', 'worth', 'value', 'cost', 'money'],
            'features': ['feature', 'ios', 'software', 'update', 'app', 'apps'],
            'size': ['size', 'weight', 'heavy', 'light', 'lighter', 'compact', 'big', 'large'],
            'setup': ['setup', 'install', 'transfer', 'backup', 'restore'],
            'heating': ['heat', 'hot', 'warm', 'cool', 'temperature', 'vapor', 'chamber'],
            'upgrade': ['upgrade', 'upgrading', 'upgraded', 'previous', 'older']
        }
    
    def extract_aspect_mentions(self, text: str) -> List[str]:
        """Extract product aspects mentioned in text"""
        text_lower = text.lower()
        mentioned_aspects = []
        
        for aspect, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + keyword + r'\b', text_lower):
                    mentioned_aspects.append(aspect)
                    break
        
        return mentioned_aspects
    
    def calculate_aspect_sentiment(self, df: pd.DataFrame, aspect: str, keywords: List[str]) -> Dict[str, Any]:
        """Calculate sentiment for a specific aspect"""
        # Find reviews mentioning this aspect
        mask = df['clean_text'].str.lower().str.contains('|'.join(keywords), regex=True, na=False)
        aspect_reviews = df[mask]
        
        if len(aspect_reviews) == 0:
            return {
                'count': 0,
                'avg_sentiment': 0.0,
                'avg_rating': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiment_dist = aspect_reviews['sentiment_label'].value_counts().to_dict()
        
        return {
            'count': len(aspect_reviews),
            'avg_sentiment': aspect_reviews['sentiment_compound'].mean(),
            'avg_rating': aspect_reviews['rating'].mean(),
            'positive_count': sentiment_dist.get('positive', 0),
            'negative_count': sentiment_dist.get('negative', 0),
            'neutral_count': sentiment_dist.get('neutral', 0),
            'sample_reviews': aspect_reviews.nlargest(3, 'sentiment_compound')[
                ['clean_text', 'rating', 'sentiment_compound']
            ].to_dict('records')
        }
    
    def extract_key_phrases(self, df: pd.DataFrame, aspect: str, keywords: List[str], top_n: int = 10) -> List[tuple]:
        """Extract common phrases for an aspect"""
        # Filter reviews mentioning the aspect
        mask = df['clean_text'].str.lower().str.contains('|'.join(keywords), regex=True, na=False)
        aspect_texts = df[mask]['clean_text'].values
        
        # Extract bigrams and trigrams
        phrases = []
        for text in aspect_texts:
            words = text.lower().split()
            # Bigrams
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if any(kw in phrase for kw in keywords):
                    phrases.append(phrase)
            # Trigrams
            for i in range(len(words) - 2):
                phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
                if any(kw in phrase for kw in keywords):
                    phrases.append(phrase)
        
        # Count phrase frequency
        phrase_counts = Counter(phrases)
        return phrase_counts.most_common(top_n)
    
    def extract_topics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract topics and aspects from reviews
        
        Args:
            df: Dataframe with sentiment data
            
        Returns:
            Dictionary with topic analysis results
        """
        try:
            df_topics = df.copy()
            
            # Extract aspects for each review
            df_topics['aspects'] = df_topics['clean_text'].apply(self.extract_aspect_mentions)
            
            # Calculate statistics for each aspect
            aspect_analysis = {}
            for aspect, keywords in self.aspect_keywords.items():
                aspect_stats = self.calculate_aspect_sentiment(df_topics, aspect, keywords)
                key_phrases = self.extract_key_phrases(df_topics, aspect, keywords, top_n=5)
                
                aspect_analysis[aspect] = {
                    **aspect_stats,
                    'key_phrases': key_phrases
                }
            
            # Sort aspects by frequency
            sorted_aspects = sorted(aspect_analysis.items(), 
                                  key=lambda x: x[1]['count'], 
                                  reverse=True)
            
            # Identify pain points (high mention, low sentiment)
            pain_points = []
            opportunities = []
            
            for aspect, stats in sorted_aspects:
                if stats['count'] > len(df) * 0.05:  # Mentioned in >5% of reviews
                    if stats['avg_sentiment'] < -0.1:
                        pain_points.append({
                            'aspect': aspect,
                            'sentiment': stats['avg_sentiment'],
                            'mentions': stats['count'],
                            'negative_ratio': stats['negative_count'] / stats['count'] if stats['count'] > 0 else 0
                        })
                    elif stats['avg_sentiment'] > 0.3:
                        opportunities.append({
                            'aspect': aspect,
                            'sentiment': stats['avg_sentiment'],
                            'mentions': stats['count'],
                            'positive_ratio': stats['positive_count'] / stats['count'] if stats['count'] > 0 else 0
                        })
            
            self.topic_data = df_topics
            self.topics = aspect_analysis
            
            return {
                'status': 'success',
                'message': f'Topic extraction completed on {len(df_topics)} reviews',
                'data': df_topics,
                'results': {
                    'aspect_analysis': dict(sorted_aspects),
                    'pain_points': pain_points,
                    'opportunities': opportunities,
                    'total_aspects_identified': len([a for a, s in sorted_aspects if s['count'] > 0])
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error extracting topics: {str(e)}',
                'data': None
            }
    
    def get_tool(self):
        """Return tool interface for topic extraction"""
        return {
            'name': 'extract_topics',
            'func': self.extract_topics,
            'description': 'Extracts topics and product aspects from reviews.'
        }


def create_topic_tool(df: pd.DataFrame = None):
    """
    Factory function to create topic extraction tool
    
    Args:
        df: Optional dataframe to analyze
        
    Returns:
        TopicAgent instance
    """
    agent = TopicAgent()
    
    if df is not None:
        result = agent.extract_topics(df)
        if result['status'] == 'success':
            print(f"Topic Extraction: {result['message']}")
            print(f"Identified {result['results']['total_aspects_identified']} key aspects")
            print(f"Found {len(result['results']['pain_points'])} pain points")
    
    return agent

