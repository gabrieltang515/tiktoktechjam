#!/usr/bin/env python3
"""
Data Selection Guide for Review Quality Model Training

This script demonstrates various ways to cherry-pick data from reviews.csv
to train your review quality detection model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import List, Dict, Any

class DataSelector:
    """Class to help select specific data for model training"""
    
    def __init__(self, csv_path: str = "webscrapper/reviews.csv"):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the reviews data"""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.df)} reviews")
        print(f"✓ Columns: {list(self.df.columns)}")
        print(f"✓ Rating distribution: {self.df['rating'].value_counts().sort_index().to_dict()}")
    
    def select_by_rating(self, ratings: List[int], max_per_rating: int = None) -> pd.DataFrame:
        """Select reviews by specific ratings"""
        print(f"\n🔍 Selecting reviews with ratings: {ratings}")
        
        selected = self.df[self.df['rating'].isin(ratings)].copy()
        
        if max_per_rating:
            balanced_selection = []
            for rating in ratings:
                rating_data = selected[selected['rating'] == rating]
                if len(rating_data) > max_per_rating:
                    rating_data = rating_data.sample(n=max_per_rating, random_state=42)
                balanced_selection.append(rating_data)
            selected = pd.concat(balanced_selection, ignore_index=True)
        
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Rating distribution: {selected['rating'].value_counts().sort_index().to_dict()}")
        return selected
    
    def select_by_business(self, business_names: List[str]) -> pd.DataFrame:
        """Select reviews from specific businesses"""
        print(f"\n🏪 Selecting reviews from businesses: {business_names}")
        
        selected = self.df[self.df['business_name'].isin(business_names)].copy()
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Businesses found: {selected['business_name'].unique()}")
        return selected
    
    def select_by_text_length(self, min_length: int = 50, max_length: int = 500) -> pd.DataFrame:
        """Select reviews by text length range"""
        print(f"\n📝 Selecting reviews with text length between {min_length}-{max_length} characters")
        
        text_lengths = self.df['text'].str.len()
        selected = self.df[(text_lengths >= min_length) & (text_lengths <= max_length)].copy()
        
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Average text length: {selected['text'].str.len().mean():.1f} characters")
        return selected
    
    def select_by_keywords(self, keywords: List[str], include: bool = True) -> pd.DataFrame:
        """Select reviews containing specific keywords"""
        print(f"\n🔍 Selecting reviews {'containing' if include else 'not containing'} keywords: {keywords}")
        
        mask = self.df['text'].str.lower().str.contains('|'.join(keywords), na=False)
        if not include:
            mask = ~mask
        
        selected = self.df[mask].copy()
        print(f"✓ Selected {len(selected)} reviews")
        return selected
    
    def select_random_sample(self, n_samples: int, random_state: int = 42) -> pd.DataFrame:
        """Select a random sample of reviews"""
        print(f"\n🎲 Selecting {n_samples} random reviews")
        
        selected = self.df.sample(n=min(n_samples, len(self.df)), random_state=random_state).copy()
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Rating distribution: {selected['rating'].value_counts().sort_index().to_dict()}")
        return selected
    
    def select_balanced_sample(self, n_per_rating: int = 100, random_state: int = 42) -> pd.DataFrame:
        """Select balanced sample with equal number of reviews per rating"""
        print(f"\n⚖️ Selecting balanced sample with {n_per_rating} reviews per rating")
        
        balanced_selection = []
        for rating in sorted(self.df['rating'].unique()):
            rating_data = self.df[self.df['rating'] == rating]
            if len(rating_data) >= n_per_rating:
                selected = rating_data.sample(n=n_per_rating, random_state=random_state)
            else:
                selected = rating_data  # Use all available
            balanced_selection.append(selected)
        
        selected = pd.concat(balanced_selection, ignore_index=True)
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Rating distribution: {selected['rating'].value_counts().sort_index().to_dict()}")
        return selected
    
    def select_by_quality_indicators(self, 
                                   min_length: int = 30,
                                   max_exclamation: int = 3,
                                   exclude_keywords: List[str] = None) -> pd.DataFrame:
        """Select reviews based on quality indicators"""
        print(f"\n✨ Selecting reviews based on quality indicators")
        
        if exclude_keywords is None:
            exclude_keywords = ['buy now', 'special offer', 'call', 'website', 'www', 'http']
        
        # Filter by length
        selected = self.df[self.df['text'].str.len() >= min_length].copy()
        
        # Filter by excessive exclamation marks
        exclamation_count = selected['text'].str.count('!')
        selected = selected[exclamation_count <= max_exclamation]
        
        # Filter out promotional keywords
        for keyword in exclude_keywords:
            mask = ~selected['text'].str.lower().str.contains(keyword, na=False)
            selected = selected[mask]
        
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Average text length: {selected['text'].str.len().mean():.1f} characters")
        return selected
    
    def select_by_author(self, author_names: List[str]) -> pd.DataFrame:
        """Select reviews by specific authors"""
        print(f"\n👤 Selecting reviews by authors: {author_names}")
        
        selected = self.df[self.df['author_name'].isin(author_names)].copy()
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Authors found: {selected['author_name'].unique()}")
        return selected
    
    def select_by_rating_category(self, categories: List[str]) -> pd.DataFrame:
        """Select reviews by rating category (positive, negative, etc.)"""
        print(f"\n📊 Selecting reviews by rating category: {categories}")
        
        selected = self.df[self.df['rating_category'].isin(categories)].copy()
        print(f"✓ Selected {len(selected)} reviews")
        print(f"✓ Category distribution: {selected['rating_category'].value_counts().to_dict()}")
        return selected
    
    def combine_selections(self, selections: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple selections and remove duplicates"""
        print(f"\n🔗 Combining {len(selections)} selections")
        
        combined = pd.concat(selections, ignore_index=True)
        combined = combined.drop_duplicates(subset=['text', 'author_name', 'business_name'])
        
        print(f"✓ Combined {len(combined)} unique reviews")
        return combined
    
    def save_selection(self, selected_df: pd.DataFrame, output_path: str):
        """Save selected data to a new CSV file"""
        print(f"\n💾 Saving selection to {output_path}")
        
        selected_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(selected_df)} reviews to {output_path}")
    
    def preview_selection(self, selected_df: pd.DataFrame, n_samples: int = 5):
        """Preview a sample of selected reviews"""
        print(f"\n👀 Preview of {n_samples} selected reviews:")
        print("=" * 80)
        
        for i, (_, row) in enumerate(selected_df.head(n_samples).iterrows(), 1):
            print(f"\nReview {i}:")
            print(f"Business: {row['business_name']}")
            print(f"Author: {row['author_name']}")
            print(f"Rating: {row['rating']} stars")
            print(f"Text: {row['text'][:200]}{'...' if len(row['text']) > 200 else ''}")
            print("-" * 40)

def main():
    """Main function demonstrating different data selection strategies"""
    
    print("=" * 80)
    print("DATA SELECTION GUIDE FOR REVIEW QUALITY MODEL TRAINING")
    print("=" * 80)
    
    # Initialize data selector
    selector = DataSelector()
    
    # Example 1: Select balanced sample for training
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Balanced Training Sample")
    print("=" * 80)
    
    balanced_sample = selector.select_balanced_sample(n_per_rating=200)
    selector.preview_selection(balanced_sample, n_samples=3)
    selector.save_selection(balanced_sample, "data/balanced_training_sample.csv")
    
    # Example 2: Select high-quality reviews only
    print("\n" + "=" * 80)
    print("EXAMPLE 2: High-Quality Reviews Only")
    print("=" * 80)
    
    quality_reviews = selector.select_by_quality_indicators(
        min_length=50,
        max_exclamation=2,
        exclude_keywords=['buy now', 'special offer', 'call', 'website', 'www', 'http', 'earn money']
    )
    selector.preview_selection(quality_reviews, n_samples=3)
    selector.save_selection(quality_reviews, "data/high_quality_reviews.csv")
    
    # Example 3: Select problematic reviews for training
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Problematic Reviews (Spam, Ads, etc.)")
    print("=" * 80)
    
    problematic_reviews = selector.select_by_keywords([
        'buy now', 'special offer', 'call', 'website', 'www', 'http', 
        'earn money', 'limited time', 'free', 'discount'
    ])
    selector.preview_selection(problematic_reviews, n_samples=3)
    selector.save_selection(problematic_reviews, "data/problematic_reviews.csv")
    
    # Example 4: Select by specific businesses
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Specific Business Reviews")
    print("=" * 80)
    
    # Get some business names from the data
    business_names = selector.df['business_name'].value_counts().head(3).index.tolist()
    business_reviews = selector.select_by_business(business_names)
    selector.preview_selection(business_reviews, n_samples=3)
    selector.save_selection(business_reviews, "data/specific_business_reviews.csv")
    
    # Example 5: Custom combination
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Custom Combination")
    print("=" * 80)
    
    # Combine high-quality reviews with some problematic ones for balanced training
    high_quality = selector.select_by_quality_indicators(min_length=30, max_exclamation=2)
    problematic = selector.select_by_keywords(['buy now', 'special offer'], include=True)
    
    # Take samples from each
    high_quality_sample = high_quality.sample(n=min(500, len(high_quality)), random_state=42)
    problematic_sample = problematic.sample(n=min(100, len(problematic)), random_state=42)
    
    custom_combination = selector.combine_selections([high_quality_sample, problematic_sample])
    selector.preview_selection(custom_combination, n_samples=3)
    selector.save_selection(custom_combination, "data/custom_training_set.csv")
    
    print("\n" + "=" * 80)
    print("DATA SELECTION COMPLETE!")
    print("=" * 80)
    print("Generated files:")
    print("- data/balanced_training_sample.csv")
    print("- data/high_quality_reviews.csv") 
    print("- data/problematic_reviews.csv")
    print("- data/specific_business_reviews.csv")
    print("- data/custom_training_set.csv")
    print("\nYou can now use any of these files to train your model!")

if __name__ == "__main__":
    main()
