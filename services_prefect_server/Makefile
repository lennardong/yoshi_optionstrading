build:
	docker compose -f docker-compose.dev_server.yml build 

build-nocache:
	docker compose -f docker-compose.dev_server.yml build --no-cache
up:
	docker compose -f docker-compose.dev_server.yml up -d

down:
	docker compose -f docker-compose.dev_server.yml down

logs:
	docker compose -f docker-compose.dev_server.yml logs -f

clean: down
	docker system prune -f

prefect-logs:
	docker compose -f docker-compose.dev_server.yml logs -f prefect-server >logs_prefect.txt &

postgre-logs:
	docker compose -f docker-compose.dev_server.yml logs -f postgres >logs_db.txt &

caddy-logs:
	docker compose -f docker-compose.dev_server.yml logs -f caddy > logs_caddy.txt &

prefect-shell:
	docker compose -f docker-compose.dev_server.yml exec prefect-server /bin/bash
