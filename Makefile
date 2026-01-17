# Ruby Wings Chatbot v5.2.1 - Makefile
# Simplify Docker operations

.PHONY: help build up down logs shell test clean prune health stats

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE = docker-compose
DOCKER_COMPOSE_DEV = docker-compose -f docker-compose.yml -f docker-compose.dev.yml
CONTAINER_NAME = ruby-wings-chatbot-v5.2.1

help: ## Show this help message
	@echo "Ruby Wings Chatbot v5.2.1 - Docker Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==================== PRODUCTION COMMANDS ====================

build: ## Build production Docker image
	@echo "🔨 Building production image..."
	$(DOCKER_COMPOSE) build --no-cache

up: ## Start production containers
	@echo "🚀 Starting production containers..."
	$(DOCKER_COMPOSE) up -d
	@echo "✅ Containers started! Check logs with: make logs"

down: ## Stop and remove containers
	@echo "🛑 Stopping containers..."
	$(DOCKER_COMPOSE) down

restart: down up ## Restart all containers

logs: ## View container logs
	$(DOCKER_COMPOSE) logs -f ruby-wings-chatbot

shell: ## Open shell in container
	$(DOCKER_COMPOSE) exec ruby-wings-chatbot /bin/bash

ps: ## List running containers
	$(DOCKER_COMPOSE) ps

# ==================== DEVELOPMENT COMMANDS ====================

dev-build: ## Build development image
	@echo "🔨 Building development image..."
	$(DOCKER_COMPOSE_DEV) build

dev-up: ## Start development containers with live reload
	@echo "🚀 Starting development containers..."
	$(DOCKER_COMPOSE_DEV) up

dev-down: ## Stop development containers
	$(DOCKER_COMPOSE_DEV) down

dev-logs: ## View development logs
	$(DOCKER_COMPOSE_DEV) logs -f

dev-shell: ## Open shell in development container
	$(DOCKER_COMPOSE_DEV) exec ruby-wings-chatbot /bin/bash

# ==================== TESTING & MONITORING ====================

health: ## Check container health
	@echo "🏥 Checking health..."
	@curl -f http://localhost:10000/health | python3 -m json.tool || echo "❌ Health check failed"

stats: ## Show container stats
	@echo "📊 Fetching stats..."
	@curl -s http://localhost:10000/stats | python3 -m json.tool || echo "❌ Stats failed"

meta-health: ## Check Meta CAPI health
	@echo "📱 Checking Meta CAPI..."
	@curl -s http://localhost:10000/meta-health | python3 -m json.tool || echo "❌ Meta health check failed"

test: ## Run integration tests
	@echo "🧪 Running tests..."
	@python3 test_integration.py http://localhost:10000

# ==================== UTILITY COMMANDS ====================

clean: ## Clean up stopped containers and unused images
	@echo "🧹 Cleaning up..."
	docker container prune -f
	docker image prune -f

prune: ## Remove all unused Docker resources (BE CAREFUL!)
	@echo "⚠️  WARNING: This will remove all unused Docker resources!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker system prune -a --volumes -f; \
	fi

backup-data: ## Backup data directory
	@echo "💾 Backing up data..."
	@tar -czf backup-data-$$(date +%Y%m%d-%H%M%S).tar.gz data/
	@echo "✅ Backup created!"

backup-logs: ## Backup logs directory
	@echo "📝 Backing up logs..."
	@tar -czf backup-logs-$$(date +%Y%m%d-%H%M%S).tar.gz logs/
	@echo "✅ Logs backed up!"

# ==================== DEPLOYMENT COMMANDS ====================

pull: ## Pull latest images
	$(DOCKER_COMPOSE) pull

deploy: build up ## Build and deploy
	@echo "✅ Deployment complete!"

redeploy: down deploy ## Full redeployment

# ==================== MONITORING COMMANDS ====================

monitor: ## Monitor container logs in real-time
	docker stats $(CONTAINER_NAME)

tail: ## Tail last 100 lines of logs
	$(DOCKER_COMPOSE) logs --tail=100 -f ruby-wings-chatbot

# ==================== MAINTENANCE COMMANDS ====================

update-knowledge: ## Update knowledge base (requires restart)
	@echo "📚 Updating knowledge base..."
	@echo "⚠️  Remember to restart container: make restart"

rebuild: down clean build up ## Clean rebuild and restart

# ==================== 2GB RAM UPGRADE COMMANDS ====================

upgrade-2gb: ## Upgrade to 2GB RAM configuration
	@echo "⬆️  Upgrading to 2GB RAM..."
	@echo "Updating .env file..."
	@sed -i.bak 's/RAM_PROFILE=512/RAM_PROFILE=2048/' .env
	@sed -i.bak 's/FAISS_ENABLED=false/FAISS_ENABLED=true/' .env
	@sed -i.bak 's/MEMORY_LIMIT=600M/MEMORY_LIMIT=2200M/' .env
	@sed -i.bak 's/GUNICORN_WORKERS=1/GUNICORN_WORKERS=2/' .env
	@echo "✅ .env updated! Run 'make rebuild' to apply changes."

# ==================== INFO COMMANDS ====================

version: ## Show version info
	@echo "Ruby Wings Chatbot v5.2.1"
	@echo "RAM Profile: $$(grep RAM_PROFILE .env | cut -d'=' -f2)"
	@echo "FAISS Enabled: $$(grep FAISS_ENABLED .env | cut -d'=' -f2)"

env-check: ## Check environment variables
	@echo "🔍 Checking environment..."
	@echo "OPENAI_API_KEY: $$(grep OPENAI_API_KEY .env | cut -d'=' -f2 | head -c 10)..."
	@echo "META_PIXEL_ID: $$(grep META_PIXEL_ID .env | cut -d'=' -f2)"
	@echo "CORS_ORIGINS: $$(grep CORS_ORIGINS .env | cut -d'=' -f2)"