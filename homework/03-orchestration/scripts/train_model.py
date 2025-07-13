#!/usr/bin/env python
"""
Script to train XGBoost model and log to MLflow.
Extracted from the original duration-prediction.py for use in Snakemake pipeline.
"""

import pickle
import argparse
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import mlflow

def train_model(X_train, y_train, X_val, y_val, dv, mlflow_uri, experiment_name):
    """
    Train XGBoost model and log to MLflow.
    Same logic as original script.
    """
    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Cast to XGBoost data structures
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)

        # Best hyperparameters from previous HPO
        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_param("num_boost_round", 30)
        mlflow.log_param("early_stopping_rounds", 50)

        # Train model
        print("Training XGBoost model...")
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=30,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        # Evaluate and log metrics
        y_pred = booster.predict(valid)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
        print(f"Validation RMSE: {rmse}")

        # Log artifacts
        preprocessor_path = "dict_vectorizer.pkl"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
        
        # Log model
        mlflow.xgboost.log_model(
            booster, 
            artifact_path="model",
            registered_model_name=None
        )
        
        # Clean up temporary file
        Path(preprocessor_path).unlink()

        return run.info.run_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--vectorizer', type=str, required=True)
    parser.add_argument('--output-run-id', type=str, required=True)
    parser.add_argument('--mlflow-uri', type=str, required=True)
    parser.add_argument('--experiment-name', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load features and vectorizer
    print(f"Loading features: {args.features}")
    with open(args.features, 'rb') as f:
        features_data = pickle.load(f)
    
    print(f"Loading vectorizer: {args.vectorizer}")
    with open(args.vectorizer, 'rb') as f:
        dv = pickle.load(f)
    
    # Extract data
    X_train = features_data['X_train']
    X_val = features_data['X_val']
    y_train = features_data['y_train']
    y_val = features_data['y_val']
    
    # Train model
    run_id = train_model(
        X_train, y_train, X_val, y_val, dv,
        args.mlflow_uri, args.experiment_name
    )
    
    # Save run ID
    Path(args.output_run_id).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_run_id, 'w') as f:
        f.write(run_id)
    
    print(f"MLflow run_id: {run_id}")
    print(f"Saved run ID to: {args.output_run_id}")