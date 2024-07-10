import subprocess

# Define the flow script and flow name
flow_script = "flow.py"
flow_name = "hello_world_flow"
deployment_name = "Hello World Flow"

# Build the deployment
subprocess.run(
    [
        "prefect",
        "deployment",
        "build",
        f"{flow_script}:{flow_name}",
        "-n",
        deployment_name,
        "--apply",
    ]
)

print(f"Flow '{flow_name}' registered successfully.")
