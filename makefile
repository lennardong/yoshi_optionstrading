.PHONY: check-api-health

PREFECT_API_URL ?= http://localhost:4200/api

check-api-health:
	@echo "Checking Prefect API health..."
	@curl -s -o /dev/null -w "%{http_code}" $(PREFECT_API_URL)/health | grep 200 > /dev/null && echo "API is healthy" || echo "API is not healthy"
