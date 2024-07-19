#!/bin/bash

# MLflow server connection
export MLFLOW_TRACKING_URI="http://localhost:8081"

# MinIO / S3 configuration
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
export AWS_ACCESS_KEY_ID="minioadmin"
export AWS_SECRET_ACCESS_KEY="minioadmin"

# Artifact storage
export MLFLOW_ARTIFACT_ROOT="s3://mlflow"

# Database configuration
export POSTGRES_USER="mlflow"
export POSTGRES_PASSWORD="mlflowpassword"
export POSTGRES_DB="mlflow"

# Temporary directory for MLflow
# PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
# TMP_DIR="$PROJECT_ROOT/tmp/mlflow"
# mkdir -p "$TMP_DIR"
# export MLFLOW_TMP_DIR="$TMP_DIR"

echo "MLflow environment variables set successfully."