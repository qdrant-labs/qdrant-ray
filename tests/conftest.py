"""Pytest configuration for ray_qdrant tests.

The integration tests need a running Qdrant server, e.g.:

    make start-qdrant
    make test

They're skipped when no server URL is given, unless QDRANT_REQUIRE_SERVER=true,
which CI sets so a missing server fails the run.
"""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--qdrant-url",
        action="store",
        default=None,
        help="URL of a running Qdrant server, e.g. http://localhost:6333. "
        "Required to run the integration tests. Falls back to the QDRANT_URL "
        "environment variable if not given.",
    )


@pytest.fixture(scope="session")
def qdrant_url(request):
    url = request.config.getoption("--qdrant-url") or os.getenv("QDRANT_URL")
    if not url:
        if os.getenv("QDRANT_REQUIRE_SERVER") == "true":
            pytest.fail("QDRANT_REQUIRE_SERVER is set, but no Qdrant URL was given.")
        pytest.skip("Set --qdrant-url or QDRANT_URL to run integration tests.")
    return url


@pytest.fixture(scope="session")
def qdrant_grpc_port():
    return int(os.getenv("QDRANT_GRPC_PORT", "6334"))
