# MLOps Zoomcamp

## MLflow Setup for Local File System Only

```bash
# Activate environment (mlops needed)
conda deactivate
conda activate mlops-zoomcamp

# Set up directory and make a db file
mkdir mlruns
touch mlruns/mlflow.db

# Check available experiments
mlflow experiments search

# If needed, make new experiment
# First export the tracking uri, it is necessary during creation
# Without this, data base cannot be assigned
export MLFLOW_TRACKING_URI="sqlite:///mlruns/mlflow.db"
mlflow experiments create -n nyc-taxi-experiment
# In script make sure to have right experiment set
# Passing the experiment to the script as flag could be better in some situations
# But also not always, because the experiment won't change that often
# So I will just set it in the script

# Run training script
python homework/02-experiment-tracking/train_noserver.py \
    --data_path homework/02-experiment-tracking/output/

# View results with UI when needed
# Cautious: This blocks the terminal
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Then access the UI at: http://localhost:5000
```
