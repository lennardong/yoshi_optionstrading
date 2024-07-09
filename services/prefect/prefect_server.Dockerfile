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

COPY prefect.config.toml /root/.prefect/config.toml

EXPOSE 4200

# Configure 
RUN python prefect_configs.py

# Health Check & Diagnostics 
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD nc -z localhost 4200 || exit 1

ENTRYPOINT ["/entrypoint.sh"]

# Run

CMD ["poetry", "run", "prefect", "server", "start", "--host", "${PREFECT_SERVER_API_HOST:-0.0.0.0}"]






