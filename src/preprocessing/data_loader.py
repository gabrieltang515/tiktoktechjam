#!/usr/bin/env python3
"""
Data loader for Google Maps Restaurant Reviews dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewDataLoader:
    """Load and examine the Google Maps restaurant reviews dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.reviews_df = None
        self.restaurant_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load the main reviews dataset"""
        
        # Load main reviews dataset
        reviews_path = self.data_dir / "reviews.csv"
        if reviews_path.exists():
            self.reviews_df = pd.read_csv(reviews_path)
            logger.info(f"Loaded {len(self.reviews_df)} reviews from {reviews_path}")
        else:
            raise FileNotFoundError(f"Reviews file not found: {reviews_path}")
        
        # Initialize empty restaurant dataframe (no longer needed)
        self.restaurant_df = pd.DataFrame()
        
        return self.reviews_df, self.restaurant_df
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the dataset"""
        
        if self.reviews_df is None:
            self.load_data()
        
        info = {
            'total_reviews': len(self.reviews_df),
            'columns': list(self.reviews_df.columns),
            'missing_values': self.reviews_df.isnull().sum().to_dict(),
            'rating_distribution': self.reviews_df['rating'].value_counts().sort_index().to_dict(),
            'rating_category_distribution': self.reviews_df['rating_category'].value_counts().to_dict(),
            'unique_businesses': self.reviews_df['business_name'].nunique(),
            'unique_authors': self.reviews_df['author_name'].nunique(),
            'avg_review_length': self.reviews_df['text'].str.len().mean(),
            'sample_reviews': self.reviews_df['text'].head(3).tolist()
        }
        
        return info
    
    def display_sample_reviews(self, n: int = 5) -> None:
        """Display sample reviews for inspection"""
        
        if self.reviews_df is None:
            self.load_data()
        
        print(f"\n{'='*80}")
        print(f"SAMPLE REVIEWS (showing {n} reviews)")
        print(f"{'='*80}")
        
        for i, (_, row) in enumerate(self.reviews_df.head(n).iterrows()):
            print(f"\nReview {i+1}:")
            print(f"Business: {row['business_name']}")
            print(f"Author: {row['author_name']}")
            print(f"Rating: {row['rating']} ({row['rating_category']})")
            print(f"Text: {row['text'][:200]}...")
            print(f"Photo: {row['photo']}")
            print("-" * 80)
    
    def get_rating_analysis(self) -> Dict[str, Any]:
        """Analyze rating patterns and distributions"""
        
        if self.reviews_df is None:
            self.load_data()
        
        analysis = {
            'rating_stats': {
                'mean': self.reviews_df['rating'].mean(),
                'median': self.reviews_df['rating'].median(),
                'std': self.reviews_df['rating'].std(),
                'min': self.reviews_df['rating'].min(),
                'max': self.reviews_df['rating'].max()
            },
            'rating_distribution': self.reviews_df['rating'].value_counts().sort_index().to_dict(),
            'category_distribution': self.reviews_df['rating_category'].value_counts().to_dict(),
            'business_rating_stats': self.reviews_df.groupby('business_name')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10).to_dict('index')
        }
        
        return analysis

if __name__ == "__main__":
    # Test the data loader
    loader = ReviewDataLoader()
    reviews_df, restaurant_df = loader.load_data()
    
    # Display dataset information
    info = loader.get_data_info()
    print("\nDATASET INFORMATION:")
    print(f"Total reviews: {info['total_reviews']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing values: {info['missing_values']}")
    print(f"Rating distribution: {info['rating_distribution']}")
    print(f"Rating categories: {info['rating_category_distribution']}")
    print(f"Unique businesses: {info['unique_businesses']}")
    print(f"Unique authors: {info['unique_authors']}")
    print(f"Average review length: {info['avg_review_length']:.1f} characters")
    
    # Display sample reviews
    loader.display_sample_reviews(3)
    
    # Rating analysis
    rating_analysis = loader.get_rating_analysis()
    print(f"\nRATING ANALYSIS:")
    print(f"Mean rating: {rating_analysis['rating_stats']['mean']:.2f}")
    print(f"Rating distribution: {rating_analysis['rating_distribution']}")
