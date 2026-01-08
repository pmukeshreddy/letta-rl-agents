.PHONY: install dev up down logs test clean lint format seed eval

# ============ Installation ============
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# ============ Docker ============
up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

build:
	docker-compose build

# ============ Local Development ============
api:
	uvicorn src.api.server:app --reload --port 8000

ui:
	python ui/dashboard.py

# ============ Database ============
seed:
	python scripts/seed_skills.py

embeddings:
	python scripts/generate_embeddings.py

# ============ Testing ============
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# ============ Code Quality ============
lint:
	ruff check src/ tests/

format:
	black src/ tests/

# ============ Evaluation ============
eval:
	python scripts/run_eval.py

# ============ Cleanup ============
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

# ============ Help ============
help:
	@echo "Available targets:"
	@echo "  install    - Install package"
	@echo "  dev        - Install with dev dependencies"
	@echo "  up/down    - Start/stop docker containers"
	@echo "  api        - Run API server locally"
	@echo "  ui         - Run Gradio dashboard"
	@echo "  seed       - Load skills into database"
	@echo "  embeddings - Generate skill embeddings"
	@echo "  test       - Run tests"
	@echo "  eval       - Run evaluation"
	@echo "  lint       - Run linter"
	@echo "  format     - Format code"
