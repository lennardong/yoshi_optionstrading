import os

import yaml

# Path to the docker-compose file
docker_compose_path = "/home/lennardong/Documents/mlops_project/services_prefect_server/docker-compose.dev_server.yml"

# Read the docker-compose file
with open(docker_compose_path, "r") as file:
    docker_compose = yaml.safe_load(file)

# Extract environment variables for the prefect-server service
prefect_server_env = docker_compose["services"]["prefect-server"]["environment"]

# Set the environment variables
for key, value in prefect_server_env.items():
    os.environ[key] = value
    print(f"Set {key}={value}")

# Print confirmation
print("\nEnvironment variables have been set for the backend.")
print("You can now run your backend scripts with these environment variables.")

# Optionally, you can print out all environment variables to verify
print("\nAll environment variables:")
for key, value in os.environ.items():
    print(f"{key}={value}")
