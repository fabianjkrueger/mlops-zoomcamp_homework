#!/usr/bin/env python
# coding: utf-8

"""
Original script from the MLOps Zoomcamp GitHub repository.
The exercise is to convert this into a pipeline orchestrated by a workflow
manager. I chose to use Snakemake for this.

I added comments to the code to make it easier to understand.
"""

# Dependencies
# ------------

import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

# MLFlow configuration
# --------------------

# FIXME I may have to adapt this part to work with my particular MLFlow setup

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

# Functions
# ---------

def read_dataframe(year, month):
    """
    Download the NYC green taxi trips data in parquet format.
    Year and month must be passed as arguments.
    Read the data and compute trip duration in minutes.
    Get a new feature column by joining pick up and drop off IDs.
    Return the processed data as pandas data frame.
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
    # these are location ID for pick up and drop off, they are numbers
    # but of course they don't represent a numerical value, but a class
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    # create a new column joining pick up and drop off IDs -> new feature
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


def create_X(df, dv=None):
    """
    Create feature set from input data frame.
    Keep only two features:
    - Pick up + drop off joined ID
    - trip distance
    Drop all other features or rather columns.
    The trip duration that was just computed will serve as label later on.
    One hot encode numerical features using a dictionary vectorizer (dv).
    dv can optionally be passed as argument.
    If you pass a dv, it is not fit to the data again, just used to transform.
    If no dv is passed, a new one is initialized and fit.
    For train set, you should initialize a new one.
    For valid or test, you should pass the one returned when processing the
    train set.
    Finally, both the feature set (X) and the dv are returned.
    """
    # define categorical and numerical features to keep
    # just keep two features here, drop all others
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    
    # convert the remaining df to a dictionary
    dicts = df[categorical + numerical].to_dict(orient='records')

    # if no dv was passed, a new one is initialized
    # it's fit to the data and used to transform it
    # should be done like this for the train set
    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    # if a dv was passed, it is not fit again, but just transforms the data
    # this should be done for the valid or test set
    else:
        X = dv.transform(dicts)

    # return the feature set AND the dv
    return X, dv


def train_model(X_train, y_train, X_val, y_val, dv):
    """
    Train an XGBoost model using the previously determined best hyperparameters.
    Evaluate it on the validation set using RMSE as metric.
    Save and log model and dv, return run ID.
    """
    # start an MLFlow run for tracking the model
    with mlflow.start_run() as run:
        # cast train and validation data set to XGBoost input data structure
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        # save best XGBoost hyper parameters to a dictionary use in training
        # best hyperparameters were determined in a previous HPO experiment
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        # log the params using MLFlow
        mlflow.log_params(best_params)

        # train an XGBoost model
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        # evaluate the model on the val set using RMSE as metric, log metric
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        
        # save both dv and model
        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
            
        # log dv and model
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

        # return run ID
        return run.info.run_id


def run(year, month):
    """
    Main function wrapping around the previous ones to run the full pipeline.
    """
    # get the training data for a particular year and month
    df_train = read_dataframe(year=year, month=month)

    # get the validation data
    # determine next year and month from training, then use that for validation
    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1
    df_val = read_dataframe(year=next_year, month=next_month)

    # process training data, get a fit dv
    X_train, dv = create_X(df_train)
    # process validation data, use the train dv
    X_val, _ = create_X(df_val, dv)

    # get labels - use trip duration as target variable
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    # train and evaluate an XGBoost model, get run ID
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    
    # print and return run ID
    print(f"MLflow run_id: {run_id}")
    return run_id

# Run main wrapper function
# -------------------------
# only run this if the script is executed directly, not when it's imported
if __name__ == "__main__":
    # import argparse only when it is needed
    import argparse

    # parse arguments for year and month for training data
    # validation data will be the next year and month
    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    # run the main wrapper function for the entire pipeline
    run_id = run(year=args.year, month=args.month)

    # write the run ID to a text file `run_id.txt`
    with open("run_id.txt", "w") as f:
        f.write(run_id)