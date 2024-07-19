# Define variables
DOCKER_COMPOSE_FILE := docker-compose.mlflow_dev.yml
MLFLOW_SERVICE := mlflow-server
DB_SERVICE := postgres

build:
	docker compose -f $(DOCKER_COMPOSE_FILE) build 

build-nocache:
	docker compose -f $(DOCKER_COMPOSE_FILE) build --no-cache

up:
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d

down:
	docker compose -f $(DOCKER_COMPOSE_FILE) down

logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f

clean: down
	docker system prune -f

mlflow-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(MLFLOW_SERVICE) >logs_mlflow.txt &

db-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(DB_SERVICE) >logs_db.txt &

mlflow-shell:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec $(MLFLOW_SERVICE) /bin/bash

db-shell:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec $(DB_SERVICE) /bin/bash

minio-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f minio >logs_minio.txt &

minio-shell:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec minio /bin/sh
mlflow-logstream:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(MLFLOW_SERVICE)
