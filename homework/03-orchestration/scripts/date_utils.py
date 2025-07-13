#!/usr/bin/env python
"""
Utility functions for date calculations in the taxi prediction pipeline.
"""

def calculate_training_months(target_year, target_month):
    """
    Calculate training and validation months based on target month.
    
    Training data: 2 months before target
    Validation data: 1 month before target
    
    Returns:
        tuple: (train_year, train_month, val_year, val_month)
    """
    # Calculate training month (2 months ago)
    train_month = target_month - 2
    train_year = target_year
    
    if train_month <= 0:
        train_month += 12
        train_year -= 1
    
    # Calculate validation month (1 month ago)
    val_month = target_month - 1
    val_year = target_year
    
    if val_month <= 0:
        val_month += 12
        val_year -= 1
    
    return train_year, train_month, val_year, val_month

if __name__ == "__main__":
    # Test the function
    import argparse
    
    parser = argparse.ArgumentParser(description='Test date calculations')
    parser.add_argument('--target-year', type=int, required=True)
    parser.add_argument('--target-month', type=int, required=True)
    
    args = parser.parse_args()
    
    train_year, train_month, val_year, val_month = calculate_training_months(
        args.target_year, args.target_month
    )
    
    print(f"Target: {args.target_year}-{args.target_month:02d}")
    print(f"Training: {train_year}-{train_month:02d}")
    print(f"Validation: {val_year}-{val_month:02d}")