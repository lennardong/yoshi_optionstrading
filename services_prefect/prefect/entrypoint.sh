#!/bin/bash
set -e

echo ""
echo "#########################"
echo "CONFIGURED ENV VARS"
echo ""
printenv
# Verify essential environment variables are set
: "${PREFECT_SERVER_DATABASE_URL:?Environment variable PREFECT_SERVER_DATABASE_URL is not set}"
: "${PREFECT_API_URL:?Environment variable PREFECT_API_URL is not set}"
: "${PREFECT_LOGGING_LEVEL:?Environment variable PREFECT_LOGGING_LEVEL is not set}"

echo ""
echo "#########################"
echo "SETTING UP PREFECT CONFIGS"
echo ""

prefect config set PREFECT_API_DATABASE_CONNECTION_URL="${PREFECT_SERVER_DATABASE_URL:-postgresql+asyncpg://prefect:prefectpassword@postgres:5432/prefect}"
prefect config set PREFECT_API_URL="${PREFECT_API_URL:-http://0.0.0.0:4200/api}"
prefect config set PREFECT_LOGGING_LEVEL="${PREFECT_LOGGING_LEVEL:-DEBUG}"

# Additional Configs
# prefect config set PREFECT_TELEMETRY_ENABLED="${PREFECT_TELEMETRY_ENABLED:-false}"
# prefect config set PREFECT_FLOWS_CHECKPOINTING="${PREFECT_FLOWS_CHECKPOINTING:-true}"
# prefect config set PREFECT_TASKS_DEFAULTS_RETRY_DELAY_SECONDS="${PREFECT_TASKS_DEFAULTS_RETRY_DELAY_SECONDS:-60}"
# prefect config set PREFECT_TASKS_DEFAULTS_RETRIES="${PREFECT_TASKS_DEFAULTS_RETRIES:-3}"
# prefect config set PREFECT_STORAGE_DEFAULT_STORAGE="${PREFECT_STORAGE_DEFAULT_STORAGE:-local}"
# prefect config set PREFECT_ENGINE_FLOW_RUNNER_DEFAULT_POLICY="${PREFECT_ENGINE_FLOW_RUNNER_DEFAULT_POLICY:-cancel}"
prefect config view --show-sources


echo "#########################"
echo ""
echo "RUNNING PREFECT SERVER"
echo ""
exec poetry run prefect server start --host $PREFECT_SERVER_API_HOST

