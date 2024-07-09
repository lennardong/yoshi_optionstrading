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