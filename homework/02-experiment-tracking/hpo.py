"""
Logging the results of a hyper parameter optimization (HPO).

In this script, hyperopt is used for HPO.
Again, this is not really about the model training itself though.
The focus of this script is to log the results of the HPO.

Before running this, start the mlflow server according to the README.
"""

# dependencies
import os
import pickle
import click
import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# set tracking -> use locally hosted server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# set experiment
mlflow.set_experiment("random-forest-hyperopt")

# helper function for loading pickle data
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# set arguments or rather flags for CLI calls
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_trials",
    default=15,
    help="The number of parameter evaluations for the optimizer to explore"
)

# define main function for hyper parameter optimization
def run_optimization(data_path: str, num_trials: int):

    # load data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    # define objective function to optimize for hyperopt: the model
    def objective(params):
        
        # start mlflow context manager
        with mlflow.start_run():
            # log resources
            mlflow.log_params(params)
            
            # initialize and train the model
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            
            # validate the model
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)
            
            # log metrics
            mlflow.log_metric("RMSE", rmse)
        
        return {'loss': rmse, 'status': STATUS_OK}

    # define search space for hyperopt
    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    # set random state for reproducible results
    rstate = np.random.default_rng(42)

    # run the optimization
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

# run main function
if __name__ == '__main__':
    run_optimization()
