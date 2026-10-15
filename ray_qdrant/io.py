"""Ray Data I/O functions for Qdrant."""

from typing import Any, Literal

from ray.data import Dataset

from .datasink import QdrantDatasink


def write_qdrant(
    ds: Dataset,
    *,
    url: str,
    collection: str | None = None,
    collection_column: str | None = None,
    api_key: str | None = None,
    id_column: str = "id",
    vector_column: str = "vector",
    vector_name: str | None = None,
    batch_size: int = 256,
    distance: Literal["Cosine", "Euclid", "Dot", "Manhattan"] = "Cosine",
    client_kwargs: dict[str, Any] | None = None,
    ray_remote_args: dict[str, Any] | None = None,
    concurrency: int | None = None,
) -> None:
    """Write a :class:`~ray.data.Dataset` to `Qdrant <https://qdrant.tech/>`_.

    Each row becomes a point: ``id_column`` is the ID, ``vector_column`` is
    the vector, and the other columns are the payload. Writes are upserts.
    If the collection doesn't exist, it's created with the vector size of the
    data.

    Examples:
        >>> import numpy as np
        >>> import ray
        >>> import ray_qdrant
        >>> ds = ray.data.range(100).map_batches(  # doctest: +SKIP
        ...     lambda batch: {
        ...         "id": batch["id"],
        ...         "vector": np.random.rand(len(batch["id"]), 4),
        ...     }
        ... )
        >>> ray_qdrant.write_qdrant(  # doctest: +SKIP
        ...     ds, url="http://localhost:6333", collection="docs"
        ... )

    Args:
        ds: The dataset to write.
        url: URL of the Qdrant server.
        collection: Collection to write to. Set this or
            ``collection_column``.
        collection_column: Column with each row's collection name. It isn't
            written to the payload.
        api_key: API key. Defaults to the ``QDRANT_API_KEY`` environment
            variable.
        id_column: Column of point IDs: unsigned integers or UUIDs.
            Rows with a null ID are skipped.
        vector_column: Column of vectors.
        vector_name: Named vector to write to. Defaults to the unnamed
            vector.
        batch_size: Maximum points per upsert request. Batches with large
            payloads are split into smaller requests.
        distance: Distance metric for a new collection: ``"Cosine"``,
            ``"Euclid"``, ``"Dot"``, or ``"Manhattan"``.
        client_kwargs: Extra arguments for ``qdrant_client.QdrantClient``.
        ray_remote_args: Kwargs passed to :func:`ray.remote` in the write
            tasks.
        concurrency: The maximum number of write tasks to run at once. It
            doesn't change the total number of tasks. By default, Ray Data
            picks it based on the available resources.
    """
    datasink = QdrantDatasink(
        collection=collection,
        collection_column=collection_column,
        url=url,
        api_key=api_key,
        id_column=id_column,
        vector_column=vector_column,
        vector_name=vector_name,
        batch_size=batch_size,
        distance=distance,
        client_kwargs=client_kwargs,
    )

    ds.write_datasink(
        datasink,
        ray_remote_args=ray_remote_args,
        concurrency=concurrency,
    )
