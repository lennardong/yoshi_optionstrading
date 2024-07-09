#!/bin/bash
set -e

echo "Starting Prefect server..."
echo "Environment variables:"
env

exec "$@"