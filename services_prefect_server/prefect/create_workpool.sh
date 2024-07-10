#!/bin/bash
set -e

# Wait for the Prefect server to be up and running
echo ""
echo "#########################"
echo "WAITING FOR PREFECT SERVER TO BE READY"
echo ""

RETRY_COUNT=0
MAX_RETRIES=30
RETRY_DELAY=10

until prefect work-pool ls || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
  echo "Waiting for Prefect server... (attempt: $((RETRY_COUNT+1)))"
  RETRY_COUNT=$((RETRY_COUNT+1))
  sleep $RETRY_DELAY
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
  echo "Prefect server did not become ready in time. Exiting."
  exit 1
fi

# Create a default work pool
echo ""
echo "#########################"
echo "CREATING DEFAULT WORK POOL"
echo ""

prefect work-pool create "workpool_localprocess" --type "process"
prefect work-pool set-concurrency-limit "workpool_localprocess" 10
prefect work-pool ls
