import os

import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression


def setup_mlflow():
    # Print the current user

    # Set the MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8081")
    mlflow.set_tracking_uri(tracking_uri)

    # Set the artifact root
    artifact_root = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow")
    os.environ["MLFLOW_ARTIFACT_ROOT"] = artifact_root

    # Set S3 endpoint URL
    s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint_url

    # Set AWS credentials
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id

    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Set the experiment name
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "test_minio")
    mlflow.set_experiment(experiment_name)


setup_mlflow()

# Create a simple linear regression model
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression()
model.fit(X, y)

with mlflow.start_run():
    # Log the model
    mlflow.sklearn.log_model(model, "model")

    # Log an artifact
    with open("test.txt", "w") as f:
        f.write("This is a test artifact")
    mlflow.log_artifact("test.txt")

print("Model and artifact logged. Check MLflow UI and MinIO console.")
