.PHONY: build up down logs clean

build:
	docker compose -f docker-compose.dev.yml build

up:
	docker compose -f docker-compose.dev.yml up -d

down:
	docker compose -f docker-compose.dev.yml down

logs:
	docker compose -f docker-compose.dev.yml logs -f

clean: down
	docker system prune -f

prefect-logs:
	docker compose -f docker-compose.dev.yml logs -f prefect

shell-prefect:
	docker compose -f docker-compose.dev.yml exec prefect /bin/bash

# Define variables
DOCKER_COMPOSE_FILE := docker-compose.dev.yml
PREFECT_CONTAINER := prefect_dev_prefect

# # Test if Prefect server is running
# test-prefect-server:
# 	@echo "Testing Prefect server..."
# 	@docker compose -f $(DOCKER_COMPOSE_FILE) exec $(PREFECT_CONTAINER) \
# 		curl -s http://localhost:4200/api/health || \
# 		(echo "Prefect server is not running" && exit 1)
# 	@echo "Prefect server is running"

# # Check Prefect server process
# check-prefect-process:
# 	@echo "Checking Prefect server process..."
# 	@docker compose -f $(DOCKER_COMPOSE_FILE) exec $(PREFECT_CONTAINER) \
# 		ps aux | grep prefect | grep -v grep || \
# 		(echo "Prefect process not found" && exit 1)
# 	@echo "Prefect process is running"

# # Check Prefect server port
# check-prefect-port:
# 	@echo "Checking Prefect server port..."
# 	@docker compose -f $(DOCKER_COMPOSE_FILE) exec $(PREFECT_CONTAINER) \
# 		netstat -tuln | grep :4200 || \
# 		(echo "Prefect server not listening on port 4200" && exit 1)
# 	@echo "Prefect server is listening on port 4200"

# # # Run all Prefect server checks
# # check-prefect: test-prefect-server check-prefect-process check-prefect-port
# # 	@echo "All Prefect server checks passed"

# # Makefile

# # Check if Prefect container is running
# check-prefect-container:
# 	@docker ps --filter "name=prefect_dev-prefect" --format "{{.Names}}" | grep -q "prefect_dev-prefect" && echo "Prefect container is running." || echo "Prefect container is not running."

# # Check if Prefect process is running inside the container
# check-prefect-process:
# 	@docker exec -it prefect_dev-prefect pgrep -f "prefect server start" > /dev/null && echo "Prefect process is running." || echo "Prefect process is not running."

# # Combined check for both container and process
# check-prefect:
# 	@$(MAKE) check-prefect-container
# 	@$(MAKE) check-prefect-process
