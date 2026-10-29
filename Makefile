PACKAGES := $(wildcard packages/*)

.PHONY: test lint format typecheck sync

test:
	@for p in $(PACKAGES); do uv run pytest $$p/tests --rootdir $$p --cov --cov-config $$p/pyproject.toml || exit 1; done

lint:
	uv run ruff check packages scripts
	uv run ruff format --check packages scripts

format:
	uv run ruff format packages scripts
	uv run ruff check --fix packages scripts

typecheck:
	uv run mypy

sync:
	uv run python scripts/check_drex_sync.py

.PHONY: examples
examples:
	uv run python examples/workflow_branching.py
	uv run python examples/router_query_engine.py
	uv run python examples/evaluation.py
