# Prima — common dev & deploy commands. Run `make help` for the list.
# Fly app names (override on the CLI: `make deploy-web WEB_APP=my-app`).
WEB_APP ?= prima-web
ML_APP  ?= prima-ml
REGION  ?= yyz

.DEFAULT_GOAL := help
.PHONY: help install dev ml build lint typecheck check \
        fly-setup deploy deploy-web deploy-ml secrets logs logs-ml status open

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## ---- Local development ----
install: ## Install JS dependencies
	npm ci

dev: ## Run the Next.js dev server (Turbopack)
	npm run dev

ml: ## Run the Python ML ensemble locally (uv venv on first run)
	npm run ml

build: ## Production build (webpack builder → .next/standalone)
	npm run build

lint: ## ESLint
	npm run lint

typecheck: ## TypeScript, no emit
	npx tsc --noEmit

check: lint typecheck build ## Run the full CI gate locally

## ---- Fly.io: one-time setup ----
fly-setup: ## Create both apps + the web volume (run once)
	flyctl apps create $(ML_APP)
	flyctl apps create $(WEB_APP)
	flyctl volumes create prima_data --app $(WEB_APP) --size 1 --region $(REGION)
	@echo "Now set the API key:  make secrets KEY=sk-ant-..."

secrets: ## Set ANTHROPIC_API_KEY on the web app (make secrets KEY=sk-ant-...)
	@test -n "$(KEY)" || (echo "Usage: make secrets KEY=sk-ant-..." && exit 1)
	flyctl secrets set ANTHROPIC_API_KEY=$(KEY) --app $(WEB_APP)

## ---- Fly.io: deploy ----
deploy: deploy-ml deploy-web ## Deploy ML then web

deploy-web: ## Deploy the web app
	flyctl deploy --remote-only --app $(WEB_APP)

deploy-ml: ## Deploy the ML ensemble
	cd ml-service && flyctl deploy --remote-only --app $(ML_APP)

## ---- Fly.io: ops ----
status: ## Show status of both apps
	flyctl status --app $(WEB_APP)
	flyctl status --app $(ML_APP)

logs: ## Tail web app logs
	flyctl logs --app $(WEB_APP)

logs-ml: ## Tail ML app logs
	flyctl logs --app $(ML_APP)

open: ## Open the deployed web app in a browser
	flyctl open --app $(WEB_APP)
