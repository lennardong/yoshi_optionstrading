import os
from functools import wraps

import mlflow
from dotenv import load_dotenv


def setup_mlflow():
    """
    Sets up the MLflow tracking URI and experiment name, and configures the S3 endpoint URL and AWS credentials if provided.

    This function should be called before any other MLflow operations to ensure the tracking URI and experiment are properly configured.
    """
    # Load environment variables from .env file
    load_dotenv()

    # MLflow configuration
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8081")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "StockPrediction")

    # S3 configuration for artifact storage (if using S3)
    MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

    if MLFLOW_S3_ENDPOINT_URL:
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
        os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_experiment(run_name, params=None, metrics=None, artifacts=None, tags=None):
    """
    A decorator that logs the experiment details to MLflow.

    This decorator can be used to wrap a function that performs an experiment or model training. It will automatically log the experiment details, including the run name, parameters, metrics, artifacts, and tags, to MLflow.

    Args:
        run_name (str): The name of the MLflow run.
        params (list[str], optional): A list of parameter names to log from the function's keyword arguments.
        metrics (list[str], optional): A list of metric names to log from the function's return value.
        artifacts (list[str], optional): A list of file paths to log as artifacts.
        tags (dict[str, str], optional): A dictionary of tags to associate with the MLflow run.

    Returns:
        A decorator function that wraps the original function and logs the experiment details to MLflow.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with mlflow.start_run(run_name=run_name, tags=tags):
                if params:
                    mlflow.log_params({p: kwargs.get(p) for p in params if p in kwargs})

                result = func(*args, **kwargs)

                if metrics:
                    mlflow.log_metrics({m: result[m] for m in metrics if m in result})

                if artifacts:
                    for artifact in artifacts:
                        mlflow.log_artifact(artifact)

                return result

        return wrapper

    return decorator


def log_model(model_name, version=None, stage=None):
    """
    A decorator that logs a trained model to MLflow and registers it in the MLflow Model Registry.

    This decorator can be used to wrap a function that trains a model. It will automatically log the trained model to MLflow, including the model artifact, and register the model in the MLflow Model Registry.

    Args:
        model_name (str): The name of the MLflow model.
        version (str, optional): The version of the MLflow model.
        stage (str, optional): The stage to transition the MLflow model version to (e.g. "Production", "Staging", "Archived").

    Returns:
        A decorator function that wraps the original function and logs the trained model to MLflow and registers it in the MLflow Model Registry.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)

            mlflow.sklearn.log_model(model, model_name)

            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            registered_model = mlflow.register_model(model_uri, model_name)

            if version:
                client = mlflow.tracking.MlflowClient()
                client.update_model_version(
                    name=model_name,
                    version=registered_model.version,
                    description=f"Model version: {version}",
                )

            if stage:
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=model_name, version=registered_model.version, stage=stage
                )

            return model

        return wrapper

    return decorator


def load_model(model_name, stage="latest", version=None):
    """
    A decorator that loads a trained model from the MLflow Model Registry.

    This decorator can be used to wrap a function that requires a trained model. It will automatically load the specified model from the MLflow Model Registry and pass it as the first argument to the wrapped function.

    Args:
        model_name (str): The name of the MLflow model to load.
        stage (str, optional): The stage of the MLflow model to load (e.g. "Production", "Staging", "Archived"). Defaults to "latest".
        version (str, optional): The specific version of the MLflow model to load.

    Returns:
        A decorator function that wraps the original function and loads the specified MLflow model.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                model_uri = f"models:/{model_name}/{stage}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            return func(loaded_model, *args, **kwargs)

        return wrapper

    return decorator
