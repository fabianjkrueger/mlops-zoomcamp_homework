# MLOps Zoomcamp - Homework

## Environment Setup

This project uses a conda environment to manage dependencies.
The environment is defined in `environment.yaml`.

### Installation

1. Ensure you have conda installed (Anaconda or Miniconda)
2. Create the environment:
   ```bash
   conda env create -f environment.yaml -y
   ```
3. Activate the environment:
   ```bash
   conda activate mlops-zoomcamp
   ```

### Deactivating

To deactivate the environment when done:
```bash
conda deactivate
```

### Updating

If you made changes to the environment, please update it and push it to GitHub
using the script `update_and_share_environment_file.sh`:
```bash
bash scripts/update_and_share_environment_file.sh
```
