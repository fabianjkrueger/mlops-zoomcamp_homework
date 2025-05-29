"""
Log training a machine learning model with MLflow.

The model really is not too important in this script, as this is about properly
logging the training process.

This script is meant to be run from the command line.

```bash
python homework/02-experiment-tracking/train_noserver.py \
    --data_path homework/02-experiment-tracking/output/
```

Environment: `conda activate mlops-zoomcamp`
"""

# dependencies
import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path

# constants
EXPERIMENT_NAME = "nyc-taxi-experiment-server"

# set tracking uri -> local mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5002")

# set experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# print tracking uri and experiment artifact location
print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(mlflow.search_experiments())

# define helper function for loading pickle data
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# set command line options
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

# define main function for training and logging
def run_train(data_path: str):
    
    # enable MLflow autologging for scikit-learn
    mlflow.sklearn.autolog()
    
    # start a new run
    with mlflow.start_run():

        # load data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # train model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # make predictions
        y_pred = rf.predict(X_val)

        # calculate RMSE
        rmse = root_mean_squared_error(y_val, y_pred)
        
        # log the RMSE manually (autolog might not catch this)
        # this is a workaround to ensure the metric is logged
        mlflow.log_metric("rmse", rmse)

# run the main function for training and logging
if __name__ == '__main__':
    run_train()
