import mlflow

mlflow.set_tracking_uri("localhost:8081")

with mlflow.start_run():
    mlflow.log_metric("test_metric", 1.0)

print("Successfully logged metric to remote server")
