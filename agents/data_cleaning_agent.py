"""
Data Cleaning Agent
Normalizes, cleans, and preprocesses review data
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Dict, Any


class DataCleaningAgent:
    """Agent responsible for cleaning and preprocessing review data"""
    
    def __init__(self):
        self.cleaned_data = None
        
    def extract_rating(self, stars_str: str) -> float:
        """Extract numeric rating from 'Rated X.X out of 5' format"""
        try:
            if pd.isna(stars_str):
                return None
            match = re.search(r'(\d+\.?\d*)', str(stars_str))
            if match:
                rating = float(match.group(1))
                # Ensure rating is between 1 and 5
                return max(1.0, min(5.0, rating))
            return None
        except:
            return None
    
    def parse_date(self, date_str: str) -> str:
        """Parse relative date strings to ISO format"""
        try:
            if pd.isna(date_str):
                return None
                
            date_str = str(date_str).lower()
            today = datetime.now()
            
            # Extract number and unit
            if 'week' in date_str:
                match = re.search(r'(\d+)\s*week', date_str)
                if match:
                    weeks = int(match.group(1))
                    date = today - timedelta(weeks=weeks)
                else:
                    date = today - timedelta(weeks=1)
            elif 'day' in date_str:
                match = re.search(r'(\d+)\s*day', date_str)
                if match:
                    days = int(match.group(1))
                    date = today - timedelta(days=days)
                else:
                    date = today - timedelta(days=1)
            elif 'month' in date_str:
                match = re.search(r'(\d+)\s*month', date_str)
                if match:
                    months = int(match.group(1))
                    date = today - timedelta(days=months*30)
                else:
                    date = today - timedelta(days=30)
            elif 'year' in date_str:
                match = re.search(r'(\d+)\s*year', date_str)
                if match:
                    years = int(match.group(1))
                    date = today - timedelta(days=years*365)
                else:
                    date = today - timedelta(days=365)
            else:
                date = today
                
            return date.strftime('%Y-%m-%d')
        except:
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize review text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){3,}', r'\1\1', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Clean and preprocess the review dataframe
        
        Args:
            df: Input dataframe with raw review data
            
        Returns:
            Dictionary with status and cleaned dataframe
        """
        try:
            df_clean = df.copy()
            
            # Extract numeric ratings
            df_clean['rating'] = df_clean['stars'].apply(self.extract_rating)
            
            # Parse dates
            df_clean['parsed_date'] = df_clean['date'].apply(self.parse_date)
            
            # Clean review text
            df_clean['clean_text'] = df_clean['review_text'].apply(self.clean_text)
            
            # Remove duplicates based on review text
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=['clean_text'], keep='first')
            duplicates_removed = initial_count - len(df_clean)
            
            # Remove rows with missing ratings
            df_clean = df_clean[df_clean['rating'].notna()]
            
            # Remove empty reviews
            df_clean = df_clean[df_clean['clean_text'] != '']
            
            # Calculate text length
            df_clean['text_length'] = df_clean['clean_text'].str.len()
            
            # Remove very short reviews (less than 10 characters)
            df_clean = df_clean[df_clean['text_length'] >= 10]
            
            final_count = len(df_clean)
            
            self.cleaned_data = df_clean
            
            return {
                'status': 'success',
                'message': f'Cleaned {final_count} reviews (removed {initial_count - final_count} entries)',
                'data': df_clean,
                'stats': {
                    'total_cleaned': final_count,
                    'duplicates_removed': duplicates_removed,
                    'avg_text_length': df_clean['text_length'].mean(),
                    'rating_distribution': df_clean['rating'].value_counts().to_dict()
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error cleaning data: {str(e)}',
                'data': None
            }
    
    def get_tool(self):
        """Return tool interface for data cleaning"""
        return {
            'name': 'clean_review_data',
            'func': lambda x: self.clean_data(x),
            'description': 'Cleans and preprocesses review data.'
        }


def create_data_cleaning_tool(df: pd.DataFrame = None):
    """
    Factory function to create data cleaning tool
    
    Args:
        df: Optional dataframe to clean
        
    Returns:
        DataCleaningAgent instance
    """
    agent = DataCleaningAgent()
    
    if df is not None:
        result = agent.clean_data(df)
        if result['status'] == 'success':
            print(f"Data Cleaning: {result['message']}")
            print(f"Average text length: {result['stats']['avg_text_length']:.0f} characters")
    
    return agent

