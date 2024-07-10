FROM python:3.8-slim

WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    netcat-openbsd \
    iputils-ping \
    procps \
    htop \
    strace \
    net-tools \
    jq \
    git \
    openssl \
    ntp \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

COPY pyproject.toml poetry.lock ./

# Install Prefect and additional packages
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

COPY entrypoint.sh /app/entrypoint.sh
COPY create_workpool.sh /app/create_workpool.sh

RUN chmod +x /app/entrypoint.sh /app/create_workpool.sh

EXPOSE 4200

HEALTHCHECK CMD curl --fail http://localhost:4201/api/health || exit 1

CMD ["/bin/bash", "-c", "/app/create_workpool.sh & /app/entrypoint.sh"]
# This command starts create_workpool.sh in the background (&) and then runs entrypoint.sh. 
# This allows the work pool creation to happen concurrently with the server startup, but ensures that the container keeps running as long as the server 
# (started by entrypoint.sh) is running.
# Alternative is to mush all together into a single ENTRYPOINT command. 

