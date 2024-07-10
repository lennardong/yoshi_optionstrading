This is for the setup of infrastructure for mlops.

It covers:
- Prefect Server
- MLflow Server

For both, it has a dev env and prod env file.

All are managed via MAKEFILES

The directory structure is as follows: 

/prefect-deployment
  ├── docker-compose.yml
  ├── docker-compose.override.yml
  ├── Caddyfile
  ├── Caddyfile.local
  ├── .env
  ├── .env.local
  ├── Makefile
  ├── secrets/
  │   ├── postgres_db.txt
  │   ├── postgres_user.txt
  │   └── postgres_password.txt
  └── prefect/
      ├── Dockerfile
      ├── pyproject.toml
      ├── poetry.lock
      └── src/
          └── your_prefect_code/
              └── __init__.py


```mermaid
graph TD
    subgraph "Docker Host"
        subgraph "Network: prefect-network"
            direction TB
            A[Prefect Server Container]
            B[PostgreSQL Container]
            C[Caddy Container]
        end
    end

    subgraph "External Network"
        direction TB
        D[User Browser]
    end

    D -->|HTTP Request| C
    C -->|Reverse Proxy| A
    A -->|Database Connection| B

    subgraph "Prefect Server Container"
        direction TB
        A1[Python 3.8-slim]
        A2[Poetry]
        A3[Prefect Server]
        A4[Prefect Config]
        A1 --> A2
        A2 --> A3
        A3 --> A4
    end

    subgraph "PostgreSQL Container"
        direction TB
        B1[PostgreSQL 13.3]
        B2[Database: prefect]
        B1 --> B2
    end

    subgraph "Caddy Container"
        direction TB
        C1[Caddy 2.6.2-alpine]
        C2[Caddyfile.dev]
        C1 --> C2
    end

```

Connections
User Browser to Caddy Container: The user makes HTTP requests to the Caddy container.
Caddy Container to Prefect Server Container: The Caddy container acts as a reverse proxy, forwarding requests to the Prefect server container.
Prefect Server Container to PostgreSQL Container: The Prefect server container connects to the PostgreSQL container to interact with the database.
