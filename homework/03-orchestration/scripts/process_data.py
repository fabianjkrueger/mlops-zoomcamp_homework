#!/usr/bin/env python
"""
Script to download and process NYC taxi data.
Updated to use target month with proper training/validation split.
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
import os

# Add scripts directory to path to import date_utils
sys.path.append(os.path.dirname(__file__))
from date_utils import calculate_training_months

def read_dataframe(year, month):
    """Same as before - no changes needed"""
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process taxi data')
    parser.add_argument('--target-year', type=int, required=True)
    parser.add_argument('--target-month', type=int, required=True)
    parser.add_argument('--output-train', type=str, required=True)
    parser.add_argument('--output-val', type=str, required=True)
    
    args = parser.parse_args()
    
    # Calculate actual training and validation months
    train_year, train_month, val_year, val_month = calculate_training_months(
        args.target_year, args.target_month
    )
    
    print(f"Target month: {args.target_year}-{args.target_month:02d}")
    print(f"Training data: {train_year}-{train_month:02d}")
    print(f"Validation data: {val_year}-{val_month:02d}")
    
    # Process training data (2 months ago)
    df_train = read_dataframe(train_year, train_month)
    
    # Process validation data (1 month ago)
    df_val = read_dataframe(val_year, val_month)
    
    # Save processed data
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(args.output_train)
    df_val.to_parquet(args.output_val)
    
    print(f"Saved training data: {args.output_train}")
    print(f"Saved validation data: {args.output_val}")