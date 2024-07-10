FROM python:3.8-slim

# Install dependencies
RUN pip install poetry

# Copy the agent script
COPY prefect_agent.py /app/prefect_agent.py

# Set the working directory
WORKDIR /app

# Install Prefect
RUN poetry add prefect

# Command to run the Prefect agent
CMD ["poetry", "run", "prefect", "agent", "start"]
