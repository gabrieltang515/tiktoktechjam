#!/usr/bin/env python3
"""
Download Google Maps Restaurant Reviews dataset from Kaggle
"""

import os
import zipfile
import pandas as pd
import requests
from pathlib import Path

def download_dataset():
    """Download the Google Maps Restaurant Reviews dataset"""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Check if dataset already exists
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print("Dataset already exists in data directory:")
        for file in csv_files:
            print(f"  - {file.name}")
        return
    
    # Try Kaggle CLI first
    try:
        print("Attempting to download using Kaggle CLI...")
        dataset_name = "denizbilginn/google-maps-restaurant-reviews"
        result = os.system(f"kaggle datasets download -d {dataset_name} -p {data_dir}")
        
        if result == 0:
            # Extract the zip file
            zip_path = data_dir / "google-maps-restaurant-reviews.zip"
            if zip_path.exists():
                print("Extracting dataset...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                
                # Remove the zip file
                zip_path.unlink()
                print("Dataset downloaded and extracted successfully!")
                
                # List the files
                print("\nFiles in data directory:")
                for file in data_dir.iterdir():
                    if file.is_file():
                        print(f"  - {file.name}")
                return
                
    except Exception as e:
        print(f"Kaggle CLI failed: {e}")
    
    # If Kaggle CLI fails, provide instructions
    print("\nKaggle CLI download failed. Please manually download the dataset:")
    print("1. Go to: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews")
    print("2. Click 'Download' button")
    print("3. Extract the zip file to the 'data' directory")
    print("4. Ensure the CSV files are directly in the 'data' folder")
    
    # Create a sample dataset for testing if needed
    create_sample_dataset(data_dir)

def create_sample_dataset(data_dir):
    """Create a sample dataset for testing purposes"""
    print("\nCreating a sample dataset for testing...")
    
    sample_data = {
        'reviewer_name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'review_text': [
            'Great food and excellent service! The pasta was delicious and the staff was very friendly.',
            'Terrible experience. Food was cold and service was slow. Would not recommend.',
            'Amazing restaurant! Best pizza I have ever had. The atmosphere is perfect for a date night.',
            'This place is a scam. They advertise fresh ingredients but everything tastes frozen.',
            'Wonderful dining experience. The chef really knows how to cook authentic Italian cuisine.'
        ],
        'rating': [5, 1, 5, 1, 5],
        'review_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'restaurant_name': ['Italian Delight', 'Italian Delight', 'Italian Delight', 'Italian Delight', 'Italian Delight'],
        'latitude': [40.7128, 40.7128, 40.7128, 40.7128, 40.7128],
        'longitude': [-74.0060, -74.0060, -74.0060, -74.0060, -74.0060]
    }
    
    df = pd.DataFrame(sample_data)
    sample_file = data_dir / "sample_reviews.csv"
    df.to_csv(sample_file, index=False)
    print(f"Sample dataset created: {sample_file}")

if __name__ == "__main__":
    download_dataset()
