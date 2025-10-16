"""
Utility functions for iPhone 17 Pro Max Reviews Analysis
"""
import yaml
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up logging configuration."""
    log_file = config['governance']['logging']['save_to']
    log_level = config['governance']['logging']['level']
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def parse_stars(stars_str: str, patterns: List[str]) -> float:
    """
    Parse star rating from various string formats.
    
    Args:
        stars_str: String containing star rating (e.g., "Rated 4.0 out of 5")
        patterns: List of regex patterns to try
        
    Returns:
        Normalized star rating (1-5 scale), or NaN if parsing fails
    """
    import numpy as np
    
    if not isinstance(stars_str, str):
        return np.nan
    
    for pattern in patterns:
        match = re.search(pattern, stars_str)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            # Normalize to 5-star scale
            normalized = (numerator / denominator) * 5.0
            return np.clip(normalized, 1.0, 5.0)
    
    return np.nan

def parse_date(date_str: str) -> Tuple[Any, Any]:
    """
    Parse date string from various formats.
    
    Args:
        date_str: String containing date (e.g., "Review submitted 3 weeks ago")
        
    Returns:
        Tuple of (parsed_datetime, month_start)
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    if not isinstance(date_str, str):
        return pd.NaT, pd.NaT
    
    # Reference date (approximate current date for "ago" calculations)
    reference_date = datetime(2024, 10, 15)
    
    # Pattern for "X weeks/days ago"
    weeks_ago = re.search(r'(\d+)\s+weeks?\s+ago', date_str, re.IGNORECASE)
    days_ago = re.search(r'(\d+)\s+days?\s+ago', date_str, re.IGNORECASE)
    
    if weeks_ago:
        weeks = int(weeks_ago.group(1))
        date = reference_date - timedelta(weeks=weeks)
    elif days_ago:
        days = int(days_ago.group(1))
        date = reference_date - timedelta(days=days)
    elif 'a week ago' in date_str.lower():
        date = reference_date - timedelta(weeks=1)
    elif 'yesterday' in date_str.lower():
        date = reference_date - timedelta(days=1)
    elif 'today' in date_str.lower():
        date = reference_date
    else:
        # Try to parse as standard date
        try:
            date = pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT, pd.NaT
    
    if pd.isna(date):
        return pd.NaT, pd.NaT
    
    # Convert to timestamp
    date = pd.Timestamp(date)
    # Get month start
    month_start = date.to_period('M').to_timestamp()
    
    return date, month_start

def text_hash(text: str) -> str:
    """Generate MD5 hash of text for caching."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def expand_contractions(text: str) -> str:
    """Expand common English contractions."""
    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "here's": "here is",
        "what's": "what is",
        "where's": "where is",
        "who's": "who is",
        "how's": "how is",
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
        text = text.replace(contraction.capitalize(), expansion.capitalize())
    
    return text

def clean_text(text: str, config: Dict[str, Any]) -> str:
    """
    Clean and normalize text according to configuration.
    
    Args:
        text: Input text
        config: Configuration dictionary
        
    Returns:
        Cleaned text
    """
    import emoji
    
    norm_config = config['config']['normalization']
    
    # Strip URLs and HTML
    if norm_config['strip_urls_html_handles']:
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'@\w+', '', text)
    
    # Demojize
    if norm_config['demojize']:
        text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Expand contractions
    if norm_config['expand_contractions']:
        text = expand_contractions(text)
    
    # Lowercase
    if norm_config['lowercase']:
        text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def compute_text_features(text: str) -> Dict[str, Any]:
    """
    Compute various text features.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary of features
    """
    return {
        'word_len': len(text.split()),
        'char_len': len(text),
        'exclaim_count': text.count('!'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
    }

