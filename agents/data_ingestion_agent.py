"""
Data Ingestion Agent
Loads and validates review data from CSV files
"""

import pandas as pd
from typing import Dict, Any


class DataIngestionAgent:
    """Agent responsible for loading and validating review data"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.data = None
        
    def load_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load review data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary with status and loaded dataframe
        """
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Validate schema
            required_columns = ['review_text', 'stars', 'date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    'status': 'error',
                    'message': f'Missing required columns: {missing_columns}',
                    'data': None
                }
            
            # Remove completely empty rows
            df = df.dropna(how='all')
            
            # Basic validation
            initial_rows = len(df)
            
            # Remove rows with missing review_text
            df = df[df['review_text'].notna()]
            df = df[df['review_text'].str.strip() != '']
            
            final_rows = len(df)
            removed_rows = initial_rows - final_rows
            
            self.data = df
            
            return {
                'status': 'success',
                'message': f'Successfully loaded {final_rows} valid reviews (removed {removed_rows} invalid entries)',
                'data': df,
                'stats': {
                    'total_reviews': final_rows,
                    'columns': list(df.columns),
                    'removed_entries': removed_rows
                }
            }
            
        except FileNotFoundError:
            return {
                'status': 'error',
                'message': f'File not found: {file_path}',
                'data': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error loading data: {str(e)}',
                'data': None
            }
    
    def get_tool(self):
        """Return tool interface for data ingestion"""
        return {
            'name': 'load_review_data',
            'func': self.load_data,
            'description': 'Loads review data from a CSV file and validates the schema.'
        }


def create_data_ingestion_tool(file_path: str = None):
    """
    Factory function to create data ingestion tool
    
    Args:
        file_path: Optional file path to load
        
    Returns:
        DataIngestionAgent instance
    """
    agent = DataIngestionAgent()
    
    if file_path:
        result = agent.load_data(file_path)
        print(f"Data Ingestion: {result['message']}")
        if result['status'] == 'success':
            print(f"Loaded {result['stats']['total_reviews']} reviews")
    
    return agent

