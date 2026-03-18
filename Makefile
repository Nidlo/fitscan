# FitScan Makefile
# Run from stitchub/fitscan/ directory.
# Run `make help` to see all commands.

.PHONY: help setup dev test test-unit test-integration test-smoke \
        db-start db-migrate db-reset db-shell db-create-migration db-status \
        models lint format typecheck check clean docs \
        server-install server-deploy server-logs server-restart

PYTHON   := $(shell command -v python3.11 2>/dev/null || echo python3)
VENV     := venv
PIP      := $(VENV)/bin/pip
PYTEST   := $(VENV)/bin/pytest
UVICORN  := $(VENV)/bin/uvicorn
ALEMBIC  := $(VENV)/bin/alembic
BLACK    := $(VENV)/bin/black
RUFF     := $(VENV)/bin/ruff
MYPY     := $(VENV)/bin/mypy
PORT     := 8008

GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# ─── Help ───────────────────────────────────────────────────────────────────

help: ## Show all commands
	@echo ""
	@echo "  FitScan — run from stitchub/fitscan/"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ \
	  { printf "  $(GREEN)%-22s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""

# ─── Setup ──────────────────────────────────────────────────────────────────

setup: ## First-time setup: venv, packages, .env
	@echo "$(GREEN)Setting up FitScan...$(NC)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -e ".[dev]" --quiet
	@if [ ! -f .env ]; then \
	  cp .env.example .env; \
	  echo "$(YELLOW)Created fitscan/.env — fill in ANTHROPIC_API_KEY$(NC)"; \
	fi
	@echo "$(GREEN)Done. Next: make db-migrate && make models$(NC)"

models: ## Download ML model weights (SMPL, HMR2, rembg ~2GB)
	@echo "$(GREEN)Downloading ML models...$(NC)"
	$(VENV)/bin/python scripts/download_models.py
	$(VENV)/bin/python scripts/download_datasets.py
	$(VENV)/bin/python scripts/download_tailornet_models.py

# ─── Development ────────────────────────────────────────────────────────────

dev: ## Start dev server on http://localhost:8000 (hot-reload)
	@echo "$(GREEN)FitScan starting → http://localhost:$(PORT)$(NC)"
	@echo "$(YELLOW)API docs → http://localhost:$(PORT)/docs$(NC)"
	$(UVICORN) fitscan.api.main:app --reload --host 127.0.0.1 --port $(PORT)

start: ## Start production server (no reload, 4 workers)
	$(UVICORN) fitscan.api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

worker: ## Start Celery async worker
	$(VENV)/bin/celery -A fitscan.tasks worker --loglevel=info --concurrency=2

# ─── Database ───────────────────────────────────────────────────────────────

db-migrate: ## Apply all pending Alembic migrations
	@echo "$(GREEN)Running migrations...$(NC)"
	$(ALEMBIC) upgrade head

db-rollback: ## Roll back the last migration
	$(ALEMBIC) downgrade -1

db-status: ## Show current migration version
	$(ALEMBIC) current

db-create-migration: ## New migration (usage: make db-create-migration NAME=add_field)
	$(ALEMBIC) revision --autogenerate -m "$(NAME)"

db-reset: ## Drop and re-run all migrations — DEV ONLY
	@echo "$(RED)This destroys fitscan_dev data.$(NC)"
	@read -p "Type yes to confirm: " c && [ "$$c" = "yes" ]
	$(ALEMBIC) downgrade base
	$(ALEMBIC) upgrade head

db-shell: ## Open psql to the dev database
	psql $$DATABASE_URL

# ─── Testing ────────────────────────────────────────────────────────────────

test: ## Run all tests with coverage
	$(PYTEST) --cov=fitscan --cov-report=term-missing --cov-fail-under=55 -v

test-unit: ## Unit tests only (fast, no DB needed)
	$(PYTEST) tests/unit/ -v --tb=short

test-integration: ## Integration tests (needs DB running)
	$(PYTEST) tests/integration/ -v --tb=short

test-smoke: ## End-to-end smoke test (needs make dev running)
	@curl -sf http://localhost:$(PORT)/api/v1/health | python3 -m json.tool || \
	  (echo "$(RED)Start FitScan first: make dev$(NC)" && exit 1)
	$(VENV)/bin/python scripts/smoke_test.py

test-coverage: ## HTML coverage report
	$(PYTEST) --cov=fitscan --cov-report=html
	@echo "Open htmlcov/index.html"

# ─── Code quality ───────────────────────────────────────────────────────────

lint: ## Lint with ruff
	$(RUFF) check fitscan/ tests/

format: ## Auto-format with black + ruff
	$(BLACK) fitscan/ tests/
	$(RUFF) check --fix fitscan/ tests/

typecheck: ## Type check with mypy
	$(MYPY) fitscan/ --ignore-missing-imports

check: lint typecheck ## Lint + typecheck

# ─── Docs ───────────────────────────────────────────────────────────────────

docs: ## Export OpenAPI JSON to docs/openapi.json
	@mkdir -p docs
	$(VENV)/bin/python -c "import json, sys; sys.path.insert(0,'.'); from fitscan.api.main import app; print(json.dumps(app.openapi(), indent=2))" > docs/openapi.json
	@echo "$(GREEN)OpenAPI spec -> docs/openapi.json$(NC)"
	@echo "$(YELLOW)Live docs at http://localhost:$(PORT)/docs$(NC)"

# ─── Server deployment ──────────────────────────────────────────────────────

server-install: ## Run server install script (Ubuntu, run as sudo)
	bash deploy/install.sh

server-deploy: ## Pull + migrate + restart (run on the server)
	git pull origin main
	$(PIP) install -e . --quiet
	$(ALEMBIC) upgrade head
	sudo systemctl restart fitscan
	@echo "$(GREEN)FitScan deployed.$(NC)"

server-restart: ## Restart the systemd service
	sudo systemctl restart fitscan

server-logs: ## Follow live server logs
	sudo journalctl -u fitscan -f

server-status: ## Check if fitscan service is running
	sudo systemctl status fitscan

# ─── Utilities ──────────────────────────────────────────────────────────────

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/ *.egg-info/

keys-list: ## List all API keys (hashed — no secrets shown)
	@$(VENV)/bin/python -c "from sqlalchemy.orm import Session; from fitscan.infrastructure.result_store import engine, APIKeyRecord; s = Session(engine); [print(f'{k.owner_email:30s} tier={k.tier:8s} {k.usage_this_month}/{k.monthly_limit}/mo active={k.is_active}') for k in s.query(APIKeyRecord).all()]; s.close()"

health: ## Quick health check against running local API
	@curl -sf http://localhost:$(PORT)/api/v1/health | python3 -m json.tool
