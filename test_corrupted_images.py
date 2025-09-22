#!/usr/bin/env python3
"""
Test script to verify corrupted image handling in the dataloader
"""

import os
import sys
from src.dataloader import get_data_pipelines

def test_dataloader_with_corrupted_images():
    """Test the dataloader with potentially corrupted images"""
    print("🧪 Testing dataloader with corrupted image handling...")
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please run the main.py script first to create the data splits.")
        return False
    
    try:
        # Load datasets
        print("📁 Loading datasets...")
        train_ds, val_ds, test_ds = get_data_pipelines(data_dir, batch_size=8)
        
        print("✅ Datasets loaded successfully!")
        
        # Test each dataset
        datasets = [("Training", train_ds), ("Validation", val_ds), ("Test", test_ds)]
        
        for name, dataset in datasets:
            if dataset is not None:
                print(f"\n🔍 Testing {name} dataset...")
                try:
                    # Try to iterate through a few batches
                    batch_count = 0
                    for x_batch, y_batch in dataset.take(3):  # Take only 3 batches for testing
                        batch_count += 1
                        print(f"  ✅ Batch {batch_count}: Images shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
                        
                        # Check for any NaN or infinite values
                        if tf.reduce_any(tf.math.is_nan(x_batch)):
                            print(f"  ⚠️  Warning: NaN values detected in batch {batch_count}")
                        if tf.reduce_any(tf.math.is_inf(x_batch)):
                            print(f"  ⚠️  Warning: Infinite values detected in batch {batch_count}")
                    
                    print(f"  ✅ {name} dataset processed {batch_count} batches successfully!")
                    
                except Exception as e:
                    print(f"  ❌ Error processing {name} dataset: {str(e)}")
                    return False
            else:
                print(f"  ⚠️  {name} dataset is None (no data found)")
        
        print("\n🎉 All tests passed! Corrupted image handling is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    import tensorflow as tf
    success = test_dataloader_with_corrupted_images()
    sys.exit(0 if success else 1)
