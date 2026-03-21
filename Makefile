.PHONY: install clean lint test tf-init tf-plan tf-apply pipeline-run

install:
	pip install -e ".[dev]"
	pre-commit install || true

clean:
	rm -rf .pytest_cache .ruff_cache src/*.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/

test:
	pytest tests/unit/ -v --cov=src/chitrakatha --cov-fail-under=80

tf-init:
	cd infra/terraform && terraform init

tf-plan:
	cd infra/terraform && terraform plan

tf-apply:
	cd infra/terraform && terraform apply -auto-approve

pipeline-run:
	python pipeline/pipeline.py --execute
