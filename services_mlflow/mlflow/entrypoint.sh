#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until PGPASSWORD=$POSTGRES_PASSWORD psql -h "postgres" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

>&2 echo "Postgres is up - executing command"

# Set environment variables for MLflow
export MLFLOW_ARTIFACT_ROOT=/mlflow/artifacts
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