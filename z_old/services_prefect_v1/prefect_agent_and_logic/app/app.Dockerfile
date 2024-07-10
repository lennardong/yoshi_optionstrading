FROM python:3.8-slim

# Install dependencies
RUN pip install poetry

# Copy the logic script
COPY flow.py /app/flow.py

# Set the working directory
WORKDIR /app

# Install Prefect
RUN poetry add prefect

# Command to run the business logic
CMD ["poetry", "run", "python", "flow.py"]
