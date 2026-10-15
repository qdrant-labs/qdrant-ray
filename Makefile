.PHONY: test test-local test-cluster start-qdrant stop-qdrant start-cluster stop-cluster lint format type

TEST ?= tests

start-qdrant:
	docker compose -f tests/compose-qdrant.yml up -V --force-recreate --wait

stop-qdrant:
	docker compose -f tests/compose-qdrant.yml down

start-cluster:
	docker compose -f tests/compose-cluster.yml up -V --force-recreate --wait

stop-cluster:
	docker compose -f tests/compose-cluster.yml down

test:
	QDRANT_URL=http://localhost:6333 QDRANT_GRPC_PORT=6334 uv run pytest $(TEST)

test-cluster:
	QDRANT_URL=http://localhost:7002 QDRANT_GRPC_PORT=7012 uv run pytest $(TEST)

# Unit tests only; skips the tests that need a Qdrant server.
test-local:
	QDRANT_URL= uv run pytest $(TEST)

lint:
	uv run ruff check ray_qdrant tests examples
	uv run ruff format --check ray_qdrant tests examples
	uv run mypy ray_qdrant

type:
	uv run mypy ray_qdrant

format:
	uv run ruff format ray_qdrant tests examples
	uv run ruff check --fix ray_qdrant tests examples
