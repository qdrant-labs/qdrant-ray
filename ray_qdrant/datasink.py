"""Ray Data datasink that writes rows to Qdrant as points."""

import datetime
import functools
import itertools
import json
import math
import os
import pickle
import uuid
from collections.abc import Iterable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from qdrant_client import QdrantClient, models
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.datasink import Datasink

from ._retry import call_with_retry

if TYPE_CHECKING:
    from ray.data._internal.execution.interfaces import TaskContext

_DISTANCES = ("Cosine", "Euclid", "Dot", "Manhattan")

# The sink sets these itself, so they can't be passed in ``client_kwargs``.
_RESERVED_CLIENT_KWARGS = ("url", "api_key", "location", "host", "path")

# Qdrant rejects requests over 32 MiB by default. Splitting upserts at half
# that leaves room for request overhead.
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
# Upper bound on the JSON size of one vector element, like "-1.2345678901234567e-300, ".
_JSON_BYTES_PER_FLOAT = 26

_RETRYABLE_ERRORS = [
    "ResponseHandlingException",
    r"Unexpected Response: (408|429|5\d\d)",
    r"StatusCode\.(UNAVAILABLE|DEADLINE_EXCEEDED|RESOURCE_EXHAUSTED|ABORTED|INTERNAL)",
]


# Constructor arguments of ``qdrant_client``'s ``UnexpectedResponse``.
_UNEXPECTED_RESPONSE_ATTRS = ("status_code", "reason_phrase", "content", "headers")


def _is_picklable(error: Exception) -> bool:
    try:
        pickle.loads(pickle.dumps(error))
        return True
    except Exception:
        return False


def _to_picklable(error: Exception) -> Exception:
    """Return ``error``, or a stand-in, that Ray can send back to the driver.

    Some Ray versions report an exception that can't be unpickled as "Failed to
    unpickle serialized exception", hiding the actual error.
    """
    if _is_picklable(error):
        return error
    if not error.args and all(hasattr(error, a) for a in _UNEXPECTED_RESPONSE_ATTRS):
        # qdrant_client creates UnexpectedResponse with keyword arguments, so
        # pickle can't recreate it from its empty ``args``.
        error.args = tuple(getattr(error, a) for a in _UNEXPECTED_RESPONSE_ATTRS)
        if _is_picklable(error):
            return error
    return RuntimeError(f"{type(error).__module__}.{type(error).__qualname__}: {error}")


def _to_pylist(column: pa.ChunkedArray) -> list[Any]:
    """Convert a column to Python values.

    Ray converts its own tensor and Python object columns, because their
    ``to_pylist`` fails with some combinations of Ray and pyarrow versions.
    """
    if isinstance(column.type, pa.ExtensionType):
        return list(BlockAccessor.for_block(pa.table({"c": column})).to_numpy("c"))
    return column.to_pylist()


def _to_arrow(block: Block) -> pa.Table:
    try:
        return BlockAccessor.for_block(block).to_arrow()
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        import pandas as pd

        if not isinstance(block, pd.DataFrame):
            raise
        # Older Ray versions can't convert pandas columns of UUID objects.
        block = block.copy(deep=False)
        for name in block.columns:
            if block[name].dtype == object:
                block[name] = block[name].map(
                    lambda v: str(v) if isinstance(v, uuid.UUID) else v
                )
        return BlockAccessor.for_block(block).to_arrow()


def _is_valid(column: pa.ChunkedArray) -> pa.Array:
    """Return a mask of the non-missing values of ``column``.

    Python object columns store missing values, like pandas' None and NaN, as
    objects rather than nulls.
    """
    if isinstance(column.type, pa.ExtensionType):
        return pa.array(
            [
                not (v is None or (isinstance(v, float) and math.isnan(v)))
                for v in _to_pylist(column)
            ],
            pa.bool_(),
        )
    return column.is_valid()


class QdrantDatasink(Datasink):
    """Writes each row of a dataset to Qdrant as a point.

    ``id_column`` is the point ID, ``vector_column`` is the vector, and the other
    columns are the payload. Writes are upserts. If the collection doesn't exist,
    the sink creates it with the vector size of the data.

    Use :func:`ray_qdrant.write_qdrant`, or pass an instance to
    :meth:`ray.data.Dataset.write_datasink`.
    """

    def __init__(
        self,
        collection: str | None = None,
        *,
        collection_column: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
        id_column: str = "id",
        vector_column: str = "vector",
        vector_name: str | None = None,
        batch_size: int = 256,
        distance: Literal["Cosine", "Euclid", "Dot", "Manhattan"] = "Cosine",
        client_kwargs: dict[str, Any] | None = None,
    ):
        if collection and collection_column:
            raise ValueError(
                "Specify exactly one of 'collection' or 'collection_column', not both."
            )
        if not collection and not collection_column:
            raise ValueError(
                "Either 'collection' or 'collection_column' must be provided."
            )
        if not url:
            raise ValueError("'url' must be provided.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if distance not in _DISTANCES:
            raise ValueError(
                f"distance must be one of {list(_DISTANCES)}, got {distance!r}."
            )
        if id_column == vector_column:
            raise ValueError(
                "id_column and vector_column refer to the same column "
                f"'{id_column}'. They must be distinct."
            )
        if collection_column in (id_column, vector_column):
            raise ValueError(
                f"collection_column '{collection_column}' must not be the same "
                f"as id_column ('{id_column}') or vector_column "
                f"('{vector_column}')."
            )
        reserved = sorted(set(client_kwargs or {}) & set(_RESERVED_CLIENT_KWARGS))
        if reserved:
            raise ValueError(
                f"client_kwargs must not contain {reserved}; use the 'url' and "
                "'api_key' arguments instead."
            )

        self.collection = collection
        self.collection_column = collection_column
        self.url = url
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.id_column = id_column
        self.vector_column = vector_column
        self.vector_name = vector_name
        self.batch_size = batch_size
        self.distance = distance
        self.client_kwargs = client_kwargs or {}
        self._ready_collections: set[str] = set()

    def _get_client(self) -> QdrantClient:
        return QdrantClient(url=self.url, api_key=self.api_key, **self.client_kwargs)

    def write(
        self,
        blocks: Iterable[Block],
        ctx: "TaskContext",
    ) -> None:
        client = self._get_client()
        try:
            for block in blocks:
                table = _to_arrow(block)
                if table.num_rows == 0:
                    continue
                if self.collection_column:
                    self._write_multi_collection(client, table)
                else:
                    assert self.collection is not None
                    self._write_to_collection(client, table, self.collection)
        except Exception as e:
            picklable = _to_picklable(e)
            if picklable is e:
                raise
            raise picklable from e
        finally:
            client.close()

    def _write_multi_collection(self, client: QdrantClient, table: pa.Table) -> None:
        name = self.collection_column
        if name not in table.column_names:
            raise ValueError(
                f"Collection column '{name}' not found in table. "
                f"Available columns: {table.column_names}"
            )
        column = table.column(name)
        if column.null_count > 0:
            raise ValueError(
                f"Collection column '{name}' contains null values; "
                "fill or drop them before writing with collection_column."
            )
        try:
            column = column.cast(pa.string())
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            raise ValueError(
                f"Collection column '{name}' has type {column.type}; it must "
                "contain strings or values that can be cast to strings."
            ) from None

        # Sort rows by collection, keeping their order within each collection,
        # then write each run of rows with the same collection.
        order = pc.sort_indices(column)
        table = table.remove_column(table.schema.get_field_index(name)).take(order)
        start = 0
        for collection, rows in itertools.groupby(column.take(order).to_pylist()):
            num_rows = sum(1 for _ in rows)
            group = table.slice(start, num_rows)
            self._write_to_collection(client, group, collection)
            start += num_rows

    def _write_to_collection(
        self, client: QdrantClient, table: pa.Table, collection: str
    ) -> None:
        for column, kind in ((self.id_column, "ID"), (self.vector_column, "Vector")):
            if column not in table.column_names:
                raise ValueError(
                    f"{kind} column '{column}' not found in table. "
                    f"Available columns: {table.column_names}"
                )
        table = table.filter(_is_valid(table.column(self.id_column)))
        if table.column(self.vector_column).null_count > 0:
            raise ValueError(
                f"Vector column '{self.vector_column}' contains null values; "
                "fill or drop them before writing to Qdrant."
            )

        for offset in range(0, table.num_rows, self.batch_size):
            batch = table.slice(offset, self.batch_size)
            ids = self._to_point_ids(batch.column(self.id_column))
            vectors = self._to_vectors(batch)
            payloads = self._to_payloads(batch)
            self._ensure_collection(client, collection, len(vectors[0]))

            for start, end in self._split_by_request_size(vectors, payloads):
                chunk = vectors[start:end]
                batch_vectors: Any = (
                    {self.vector_name: chunk} if self.vector_name else chunk
                )
                points = models.Batch(
                    ids=ids[start:end],
                    vectors=batch_vectors,
                    payloads=payloads[start:end] if payloads else None,
                )
                call_with_retry(
                    functools.partial(
                        client.upsert,
                        collection_name=collection,
                        points=points,
                        wait=True,
                    ),
                    description=f"upsert batch to collection '{collection}'",
                    match=_RETRYABLE_ERRORS,
                    max_attempts=5,
                )

    @staticmethod
    def _split_by_request_size(
        vectors: list[list[float]], payloads: list[dict[str, Any]] | None
    ) -> Iterable[tuple[int, int]]:
        """Yield ``(start, end)`` row ranges whose upsert requests stay under
        ``_MAX_REQUEST_BYTES``. A row that is too large on its own gets its own
        range.
        """
        start, size = 0, 0
        for i, vector in enumerate(vectors):
            row_size = len(vector) * _JSON_BYTES_PER_FLOAT + 64
            if payloads:
                row_size += len(json.dumps(payloads[i]))
            if i > start and size + row_size > _MAX_REQUEST_BYTES:
                yield start, i
                start, size = i, 0
            size += row_size
        yield start, len(vectors)

    def _to_point_ids(self, column: pa.ChunkedArray) -> list[models.ExtendedPointId]:
        """Qdrant only allows unsigned integers or UUIDs as point IDs.
        Ref: https://qdrant.tech/documentation/manage-data/points/#point-ids
        """
        if pa.types.is_dictionary(column.type):
            column = column.cast(column.type.value_type)
        if pa.types.is_integer(column.type):
            ints = column.to_pylist()
            if min(ints) < 0:
                raise ValueError(
                    f"ID column '{self.id_column}' contains negative values; "
                    "Qdrant point IDs must be unsigned integers or UUIDs."
                )
            return ints

        ids: list[models.ExtendedPointId] = []
        for value in _to_pylist(column):
            if isinstance(value, np.generic):
                value = value.item()
            # A column of Python objects can mix integer and UUID IDs.
            if isinstance(value, int) and not isinstance(value, bool):
                if value < 0:
                    raise ValueError(
                        f"ID column '{self.id_column}' contains negative values; "
                        "Qdrant point IDs must be unsigned integers or UUIDs."
                    )
                ids.append(value)
                continue
            if isinstance(value, bytes) and len(value) == 16:
                value = uuid.UUID(bytes=value)
            try:
                ids.append(str(uuid.UUID(str(value))))
            except ValueError:
                raise ValueError(
                    f"ID column '{self.id_column}' contains {value!r}; Qdrant "
                    "point IDs must be unsigned integers or UUIDs."
                ) from None
        return ids

    def _to_vectors(self, table: pa.Table) -> list[list[float]]:
        invalid = ValueError(
            f"Vector column '{self.vector_column}' must contain non-empty, "
            "one-dimensional numeric vectors of the same length."
        )
        try:
            rows = list(BlockAccessor.for_block(table).to_numpy(self.vector_column))
        except pa.ArrowNotImplementedError:
            # Older pyarrow can't convert lists of some types, like float16, to
            # numpy, but can convert each row's values.
            try:
                rows = [
                    row.values.to_numpy(zero_copy_only=False)
                    for chunk in table.column(self.vector_column).chunks
                    for row in chunk
                ]
            except (AttributeError, pa.ArrowException):
                raise invalid from None
        try:
            vectors = np.stack(rows)
        except ValueError:
            raise invalid from None
        if (
            vectors.ndim != 2
            or vectors.shape[1] == 0
            or not np.issubdtype(vectors.dtype, np.number)
        ):
            raise invalid
        with np.errstate(over="ignore"):
            finite = np.isfinite(vectors.astype(np.float32)).all()
        if not finite:
            raise ValueError(
                f"Vector column '{self.vector_column}' contains NaN or infinite "
                "values, which Qdrant can't store."
            )
        return vectors.astype(np.float64).tolist()

    def _to_payloads(self, table: pa.Table) -> list[dict[str, Any]] | None:
        names = [
            name
            for name in table.column_names
            if name not in (self.id_column, self.vector_column)
        ]
        if not names:
            return None
        columns = [
            [self._to_payload_value(v, name) for v in _to_pylist(table.column(name))]
            for name in names
        ]
        return [
            dict(zip(names, row, strict=True)) for row in zip(*columns, strict=True)
        ]

    @staticmethod
    def _to_payload_value(value: Any, column: str) -> Any:
        """Convert a value to JSON."""
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isinf(value):
                raise ValueError(
                    f"Payload column '{column}' contains infinite values, which "
                    "Qdrant can't store."
                )
            return None if math.isnan(value) else value
        if isinstance(value, (datetime.date, datetime.time)):
            return value.isoformat()
        if isinstance(value, np.ndarray):
            return QdrantDatasink._to_payload_value(value.tolist(), column)
        if isinstance(value, dict):
            return {
                k: QdrantDatasink._to_payload_value(v, column) for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [QdrantDatasink._to_payload_value(v, column) for v in value]
        raise ValueError(
            f"Payload column '{column}' contains a value of type "
            f"{type(value).__name__}, which Qdrant can't store. Cast the column to "
            "a string or number type before writing."
        )

    def _ensure_collection(
        self, client: QdrantClient, collection: str, vector_size: int
    ) -> None:
        if collection in self._ready_collections:
            return

        params = models.VectorParams(
            size=vector_size, distance=models.Distance(self.distance)
        )

        def create_if_missing() -> None:
            if client.collection_exists(collection):
                return
            try:
                client.create_collection(
                    collection_name=collection,
                    vectors_config={self.vector_name: params}
                    if self.vector_name
                    else params,
                )
            except Exception:
                # Another write task may have created the collection at
                # the same time.
                if not client.collection_exists(collection):
                    raise

        call_with_retry(
            create_if_missing,
            description=f"create collection '{collection}'",
            match=_RETRYABLE_ERRORS,
            max_attempts=5,
        )
        self._ready_collections.add(collection)
