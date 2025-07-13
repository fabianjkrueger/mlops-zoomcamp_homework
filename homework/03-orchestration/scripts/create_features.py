#!/usr/bin/env python
"""
Script to create features from processed taxi data.
Extracted from the original duration-prediction.py for use in Snakemake pipeline.
"""

import pandas as pd
import pickle
import argparse
from pathlib import Path
from sklearn.feature_extraction import DictVectorizer

def create_X(df, dv=None):
    """
    Create feature set from input data frame.
    Same logic as original script.
    """
    # define categorical and numerical features to keep
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    # convert the remaining df to a dictionary
    dicts = df[categorical + numerical].to_dict(orient='records')

    # if no dv was passed, a new one is initialized and fit
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    # if a dv was passed, just transform the data
    else:
        X = dv.transform(dicts)

    return X, dv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create features from processed data')
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--val-data', type=str, required=True)
    parser.add_argument('--output-features', type=str, required=True)
    parser.add_argument('--output-vectorizer', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load processed data
    print(f"Loading training data: {args.train_data}")
    df_train = pd.read_parquet(args.train_data)
    
    print(f"Loading validation data: {args.val_data}")
    df_val = pd.read_parquet(args.val_data)
    
    # Create features - fit vectorizer on training data
    print("Creating training features...")
    X_train, dv = create_X(df_train)
    
    # Transform validation data with fitted vectorizer
    print("Creating validation features...")
    X_val, _ = create_X(df_val, dv)
    
    # Get labels
    y_train = df_train['duration'].values
    y_val = df_val['duration'].values
    
    # Save features and vectorizer
    Path(args.output_features).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_vectorizer).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle files
    features_data = {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val
    }
    
    with open(args.output_features, 'wb') as f:
        pickle.dump(features_data, f)
    
    with open(args.output_vectorizer, 'wb') as f:
        pickle.dump(dv, f)
    
    print(f"Saved features: {args.output_features}")
    print(f"Saved vectorizer: {args.output_vectorizer}")
