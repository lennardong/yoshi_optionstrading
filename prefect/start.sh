#!/bin/bash

# Initialize the database
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="sqlite:////prefect/prefect.db"
prefect config set PREFECT_API_URL="http://0.0.0.0:4200/api"
prefect config set PREFECT_SERVER_API_HOST="0.0.0.0"

# Start the Prefect server
prefect server start --host 0.0.0.0 --port 4200
