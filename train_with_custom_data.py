#!/usr/bin/env python3
"""
Train Review Quality Model with Custom Selected Data

This script shows how to train the model using different custom data selections.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scripts.training.train_review_quality_model import main as train_model

def train_with_custom_data(data_file: str, output_suffix: str = ""):
    """Train model with custom data file"""
    
    print(f"🚀 Training model with custom data: {data_file}")
    print("=" * 80)
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file {data_file} not found!")
        return
    
    # Load and verify the data
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} reviews from {data_file}")
    print(f"✓ Rating distribution: {df['rating'].value_counts().sort_index().to_dict()}")
    
    # Create a backup of the original data
    original_data = "data/reviews.csv"
    if os.path.exists(original_data):
        backup_data = f"data/reviews_backup.csv"
        if not os.path.exists(backup_data):
            print(f"📋 Creating backup of original data: {backup_data}")
            os.system(f"cp {original_data} {backup_data}")
    
    # Replace the original data with custom data
    print(f"📝 Replacing training data with custom selection...")
    os.system(f"cp {data_file} {original_data}")
    
    try:
        # Train the model
        print(f"🎯 Starting model training...")
        train_model()
        print(f"✅ Training completed successfully!")
        
        # Rename the output models with custom suffix
        if output_suffix:
            models_dir = Path("models/review_quality")
            for model_file in models_dir.glob("*.pkl"):
                new_name = model_file.stem + f"_{output_suffix}" + model_file.suffix
                new_path = model_file.parent / new_name
                os.rename(model_file, new_path)
                print(f"📁 Renamed model: {new_name}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
    
    finally:
        # Restore original data
        if os.path.exists(backup_data):
            print(f"🔄 Restoring original data...")
            os.system(f"cp {backup_data} {original_data}")
            print(f"✅ Original data restored")

def main():
    """Main function to demonstrate training with different data selections"""
    
    print("=" * 80)
    print("TRAIN REVIEW QUALITY MODEL WITH CUSTOM DATA")
    print("=" * 80)
    
    # Available custom data files
    custom_data_files = {
        "1": ("data/balanced_training_sample.csv", "balanced"),
        "2": ("data/high_quality_reviews.csv", "high_quality"),
        "3": ("data/problematic_reviews.csv", "problematic"),
        "4": ("data/specific_business_reviews.csv", "specific_business"),
        "5": ("data/custom_training_set.csv", "custom_combination")
    }
    
    print("\nAvailable custom data files:")
    for key, (file_path, description) in custom_data_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"  {key}. {file_path} ({len(df)} reviews) - {description}")
        else:
            print(f"  {key}. {file_path} - NOT FOUND")
    
    print("\nChoose which data to use for training:")
    print("  1. Balanced sample (equal reviews per rating)")
    print("  2. High-quality reviews only")
    print("  3. Problematic reviews (spam, ads)")
    print("  4. Specific business reviews")
    print("  5. Custom combination")
    print("  6. Use original data (restore backup)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "6":
        # Restore original data
        backup_data = "data/reviews_backup.csv"
        if os.path.exists(backup_data):
            print("🔄 Restoring original data...")
            os.system(f"cp {backup_data} data/reviews.csv")
            print("✅ Original data restored")
        else:
            print("❌ No backup found!")
        return
    
    if choice in custom_data_files:
        file_path, suffix = custom_data_files[choice]
        train_with_custom_data(file_path, suffix)
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
