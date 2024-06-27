FROM ghcr.io/mlflow/mlflow:latest

# Install SQLite
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MLFLOW_BACKEND_STORE_URI=/mlflow/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Create directories
RUN mkdir -p /mlflow/artifacts

# Set the working directory
WORKDIR /mlflow

# Start MLflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlflow/artifacts
