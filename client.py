"""
The mlflow. client module provides a Python CRUD interface to MLflow Experiments, Runs, Model Versions, and Registered Models.
This is a lower level API that directly translates to MLflow REST API calls. For a higher level API for managing an “active run”, use the mlflow module.
"""

from pathlib import Path
from mlflow import MlflowClient

client = MlflowClient()

experiment_id = client.create_experiment(

)