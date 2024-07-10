#!/bin/bash

# Set up Prefect config
export PREFECT_API_URL="http://localhost:8080/api"

# Create a deployment for the flow
prefect deployment build toy_flow.py:simple_flow -n simple_flow_deployment -q default

# Apply the deployment
prefect deployment apply simple_flow-deployment.yaml

# Start a worker
prefect worker start -q default
