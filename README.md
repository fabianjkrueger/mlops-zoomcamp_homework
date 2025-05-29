# MLOps Zoomcamp

Be careful not to simultaneously run the server and the local file system.
This will mess up the database.
If you need to change between the two, make sure to clear all processes of the
other before starting the new one.
This was challenging during the homework, because I had to switch frequently.

You can use the following commands to kill the process by port.
In the example, I am killing the process using port 5000, but this works
for any port.

```bash
# Find the process using port 5002
lsof -ti:5000

# Kill the process (replace PID with the number from above)
kill $(lsof -ti:5000)

# Or force kill if needed
kill -9 $(lsof -ti:5000)
```

## MLflow Setup for Local File System Only

I think for me, this is the preferred way to use mlflow.
Using a server is kind of overkill, and it always has to be running.
I could use my RaspberryPi to run the server, but then the results would be
located somewhere else, and in general it's just overkill.
I will use the local file system for now, but I will keep the server setup
for future reference in case it ever becomes necessary.
If I have to collaborate with others, the server may become necessary.
Beyond that, the server is necessary for the homework, so here I have to use it.

```bash
# Activate environment (mlflow needed)
conda deactivate
conda activate mlops-zoomcamp

# Set up directory and make a db file
mkdir -p mlruns
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
python homework/02-experiment-tracking/train.py \
    --data_path homework/02-experiment-tracking/output/

# View results with UI when needed
# Cautious: This blocks the terminal
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5001

# Then access the UI at: http://localhost:5000
```

## MLflow Setup for Using Server

```bash
# Activate environment (mlflow needed)
conda deactivate
conda activate mlops-zoomcamp

# Create directory for mlflow server
# Must not mix up server with local, because that will mess up the database
mkdir -p mlflow_server/artifacts
touch mlflow_server/mlflow.db

# Launch mlflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow_server/mlflow.db \
    --default-artifact-root ./mlflow_server/artifacts \
    --host 127.0.0.1 \
    --port 5000

# When the server is used, it seems not to be necessary to create the experiment
# first. Apparently, you can just run the script with the experiment set

# Run training script
python homework/02-experiment-tracking/train_server.py \
    --data_path homework/02-experiment-tracking/output/

# Then access the UI at: http://localhost:5002
```