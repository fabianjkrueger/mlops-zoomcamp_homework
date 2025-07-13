#!/usr/bin/env python
"""
Script to download and process NYC taxi data.
Extracted from the original duration-prediction.py for use in Snakemake pipeline.
"""

import pandas as pd
import argparse
from pathlib import Path

def read_dataframe(year, month):
    """
    Download and process NYC green taxi data.
    Same logic as original script.
    """
    # url to data
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    # read parquet data to pandas data frame directly from url
    df = pd.read_parquet(url)

    # compute trip duration in seconds and save to a new column
    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    # convert trip duration to minutes
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # filter trip duration for trips minimum one minute, but maximum one hour
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    # convert categorical features to string type
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # create a new column joining pick up and drop off IDs -> new feature
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process taxi data')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    parser.add_argument('--output-train', type=str, required=True)
    parser.add_argument('--output-val', type=str, required=True)
    
    args = parser.parse_args()
    
    # Process training data
    print(f"Processing training data for {args.year}-{args.month:02d}")
    df_train = read_dataframe(args.year, args.month)
    
    # Process validation data (next month)
    next_year = args.year if args.month < 12 else args.year + 1
    next_month = args.month + 1 if args.month < 12 else 1
    print(f"Processing validation data for {next_year}-{next_month:02d}")
    df_val = read_dataframe(next_year, next_month)
    
    # Save processed data
    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_val).parent.mkdir(parents=True, exist_ok=True)
    
    df_train.to_parquet(args.output_train)
    df_val.to_parquet(args.output_val)
    
    print(f"Saved training data: {args.output_train}")
    print(f"Saved validation data: {args.output_val}")