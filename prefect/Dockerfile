FROM prefecthq/prefect:2-latest

# Install SQLite (if not already present in the base image)
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
RUN pip install prefect-sqlalchemy

# Set environment variables
ENV PREFECT_API_DATABASE_CONNECTION_URL=sqlite:////prefect/prefect.db
ENV PREFECT_API_URL=http://0.0.0.0:4200/api
ENV PREFECT_SERVER_API_HOST=0.0.0.0

# Create directories
RUN mkdir -p /prefect/flows /prefect/storage

# Set the working directory
WORKDIR /prefect

# Copy a custom start script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Start Prefect server
CMD ["/start.sh"]
