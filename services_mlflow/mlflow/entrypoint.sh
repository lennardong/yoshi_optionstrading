#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "postgres" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"

# Wait for MinIO to be ready
until curl -s http://minio:9000/minio/health/live; do
  >&2 echo "MinIO is unavailable - sleeping"
  sleep 1
done

>&2 echo "MinIO is up - executing command"

# Create the default bucket if it doesn't exist
mc config host add myminio http://minio:9000 $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY
mc mb myminio/mlflow --ignore-existing

# Set environment variables for MLflow
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export MLFLOW_S3_IGNORE_TLS=true

# Additional MLflow configurations
export MLFLOW_ARTIFACT_ROOT=s3://mlflow/
export MLFLOW_SERVE_ARTIFACTS=true
export MLFLOW_TRACKING_URI=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@postgres:5432/$POSTGRES_DB
export MLFLOW_SQLALCHEMY_DATABASE_URI=$MLFLOW_TRACKING_URI

# Start MLflow server with optimized settings
exec mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
  --backend-store-uri $MLFLOW_TRACKING_URI \
  --serve-artifacts \
  --workers 4 \
  --gunicorn-opts "--timeout 120 --keep-alive 120 --log-level warning" \
  --expose-prometheus