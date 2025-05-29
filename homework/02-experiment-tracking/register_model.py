"""
"""

# dependencies
import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# constants
# experiment in which HPO was done: many models, just validated, not tested
# get best runs from this later, then re-train models with params and test
HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
# this experiement: new logs and so on are written to this one
EXPERIMENT_NAME = "random-forest-best-models"
# used to iterate through the available RF parameters
# to enable passing from previous runs
RF_PARAMS = [
    'max_depth',
    'n_estimators',
    'min_samples_split',
    'min_samples_leaf',
    'random_state'
]

# set tracking and experiment
# locally hosted tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# experiment specified above
mlflow.set_experiment(EXPERIMENT_NAME)
# activate automatic logging
mlflow.sklearn.autolog()

# helper function for loading pickle data
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# function for training, validating and testing the model and logging
def train_and_log_model(data_path, params):
    
    # load train, val and test data from pickle files
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # start mlflow context manager
    with mlflow.start_run():
        
        # convert the passed RF params to int and store in in dictionary
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        # unpack dictionary using ** and pass parameters to RF
        rf = RandomForestRegressor(**new_params)
        # train RF model
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets and log to mlflow
        val_rmse = root_mean_squared_error(y_val, rf.predict(X_val))
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = root_mean_squared_error(y_test, rf.predict(X_test))
        mlflow.log_metric("test_rmse", test_rmse)

# define CLI flags
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)


# this function calls the train_and_log function
def run_register_model(data_path: str, top_n: int):

    # create instance of mlflow client
    client = MlflowClient()

    # get runs of the experiment specified in the constants section
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    
    # Retrieve the top_n model runs and log the models
    # basically get the top 5 models from previous run, take over to this run
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n, # number can be set, here default from click is used
        order_by=["metrics.rmse ASC"]
    )
    # then train again and log, this time also evaluate on test set
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    
    # FIXME
    # I THINK THIS IS PRECISELY WHERE I NEED TO DO STUFF
    
    # best_run = client.search_runs( ...  )[0]

    # Register the best model
    # mlflow.register_model( ... )



# run main function
if __name__ == '__main__':
    run_register_model()
