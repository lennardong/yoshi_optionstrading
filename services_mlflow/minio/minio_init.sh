#!/bin/sh
set -e

# Start MinIO server in the background
minio server /data --console-address ":9001" &

# Wait for MinIO to be ready
until curl -sf http://localhost:9000/minio/health/live; do
    echo "Waiting for MinIO to be ready..."
    sleep 1
done

# Configure MinIO Client
mc alias set myminio http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD

# Create mlflow bucket
mc mb myminio/mlflow

# Set read and write permissions for the mlflow bucket
mc policy set download myminio/mlflow
mc policy set upload myminio/mlflow

echo "MinIO is ready, mlflow bucket is created, and permissions are set. Initialization complete."

# Bring the MinIO server to the foreground
wait


# #!/bin/bash
# set -e

# # Wait for MinIO to be ready
# until /usr/bin/mc config host add myminio http://localhost:9000 minioadmin minioadmin; do
#   echo "Waiting for MinIO to be ready..."
#   sleep 1
# done

# # Create the MLflow bucket if it doesn't exist
# /usr/bin/mc mb myminio/mlflow --ignore-existing

# # Set the bucket policy to allow read and write access
# /usr/bin/mc policy set download myminio/mlflow
# /usr/bin/mc policy set upload myminio/mlflow

# echo "MinIO initialized successfully"

# # Keep the script running to prevent the container from stopping
# exec "$@"