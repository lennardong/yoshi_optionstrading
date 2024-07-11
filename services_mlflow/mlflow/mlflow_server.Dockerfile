FROM python:3.8-slim

WORKDIR /app

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

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

COPY . .

EXPOSE 5000

HEALTHCHECK CMD curl --fail http://localhost:5000/health || exit 1

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
