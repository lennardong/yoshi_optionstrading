# Define variables
DOCKER_COMPOSE_FILE := docker-compose.prefect_dev.yml
PREFECT_SERVICE := prefect-server
POSTGRES_SERVICE := postgres
CADDY_SERVICE := caddy

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

prefect-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(PREFECT_SERVICE) >logs_prefect.txt &

postgre-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(POSTGRES_SERVICE) >logs_db.txt &

caddy-logs:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(CADDY_SERVICE) > logs_caddy.txt &

caddy-logs-stream:
	docker compose -f $(DOCKER_COMPOSE_FILE) logs -f $(CADDY_SERVICE)

prefect-shell:
	docker compose -f $(DOCKER_COMPOSE_FILE) exec $(PREFECT_SERVICE) /bin/bash
