"""End-to-end tests against a real Qdrant server and a local Ray cluster."""

import uuid

import numpy as np
import pytest
import ray

from ray_qdrant import write_qdrant


@pytest.fixture(scope="module")
def ray_cluster():
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(params=["rest", "grpc"])
def transport(request, qdrant_grpc_port):
    """Client options for each transport, since gRPC encodes data differently."""
    if request.param == "grpc":
        return {"prefer_grpc": True, "grpc_port": qdrant_grpc_port}
    return {}


@pytest.fixture
def client(qdrant_url):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=qdrant_url)
    yield client
    client.close()


@pytest.fixture
def collection_prefix(client):
    prefix = f"ray_qdrant_test_{uuid.uuid4().hex[:8]}"
    yield prefix
    for c in client.get_collections().collections:
        if c.name.startswith(prefix):
            client.delete_collection(c.name)


def make_dataset(n: int, dim: int = 4, num_blocks: int = 4) -> ray.data.Dataset:
    def add_columns(batch):
        ids = batch["id"]
        return {
            "id": ids,
            "vector": np.stack([np.full(dim, i + 1, dtype=np.float32) for i in ids]),
            "text": [f"doc-{i}" for i in ids],
            "tenant": [f"t{i % 2}" for i in ids],
        }

    return ray.data.range(n, override_num_blocks=num_blocks).map_batches(add_columns)


def test_write_single_collection(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    collection = f"{collection_prefix}_single"

    write_qdrant(
        make_dataset(100),
        url=qdrant_url,
        client_kwargs=transport,
        collection=collection,
        batch_size=16,
        distance="Euclid",
    )

    assert client.count(collection, exact=True).count == 100
    info = client.get_collection(collection)
    assert info.config.params.vectors.size == 4
    assert info.config.params.vectors.distance.value == "Euclid"
    [point] = client.retrieve(collection, ids=[7], with_vectors=True)
    assert point.payload == {"text": "doc-7", "tenant": "t1"}
    assert point.vector == [8.0] * 4


def test_rewrite_is_upsert(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    collection = f"{collection_prefix}_upsert"

    write_qdrant(
        make_dataset(10), url=qdrant_url, client_kwargs=transport, collection=collection
    )
    write_qdrant(
        make_dataset(20), url=qdrant_url, client_kwargs=transport, collection=collection
    )

    assert client.count(collection, exact=True).count == 20


def test_write_named_vector(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    collection = f"{collection_prefix}_named"

    write_qdrant(
        make_dataset(10),
        url=qdrant_url,
        client_kwargs=transport,
        collection=collection,
        vector_name="dense",
        distance="Dot",
    )

    [point] = client.retrieve(collection, ids=[3], with_vectors=True)
    assert point.vector == {"dense": [4.0] * 4}


def test_write_multi_collection(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    ds = make_dataset(50).map_batches(
        lambda b: {**b, "tenant": [f"{collection_prefix}_{t}" for t in b["tenant"]]}
    )

    write_qdrant(
        ds, url=qdrant_url, client_kwargs=transport, collection_column="tenant"
    )

    for tenant in ("t0", "t1"):
        collection = f"{collection_prefix}_{tenant}"
        assert client.count(collection, exact=True).count == 25
        [point] = client.scroll(collection, limit=1)[0]
        assert point.payload.keys() == {"text"}


def test_write_uuid_ids(ray_cluster, client, qdrant_url, collection_prefix, transport):
    collection = f"{collection_prefix}_uuid"
    ids = [str(uuid.uuid4()) for _ in range(5)]
    ds = ray.data.from_items([{"id": i, "vector": [1.0, 0.0]} for i in ids])

    write_qdrant(ds, url=qdrant_url, client_kwargs=transport, collection=collection)

    stored = {p.id for p in client.retrieve(collection, ids=ids)}
    assert stored == set(ids)


def test_concurrent_tasks_create_one_collection(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    collection = f"{collection_prefix}_concurrent"

    write_qdrant(
        make_dataset(2_000, num_blocks=20),
        url=qdrant_url,
        collection=collection,
        batch_size=50,
        client_kwargs=transport,
        concurrency=4,
        ray_remote_args={"max_retries": 0},
    )

    assert client.count(collection, exact=True).count == 2_000


def test_server_errors_reach_the_driver(
    ray_cluster, client, qdrant_url, collection_prefix, transport
):
    from qdrant_client import models

    collection = f"{collection_prefix}_mismatch"
    client.create_collection(
        collection, vectors_config=models.VectorParams(size=3, distance="Cosine")
    )

    with pytest.raises(Exception) as info:
        write_qdrant(
            make_dataset(4),
            url=qdrant_url,
            collection=collection,
            client_kwargs=transport,
            ray_remote_args={"max_retries": 0},
        )

    # Older Ray versions hid the server's message behind "Failed to unpickle
    # serialized exception".
    assert "Failed to unpickle" not in str(info.value)
    assert "expected dim" in str(info.value)
