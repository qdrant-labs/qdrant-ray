import datetime
import decimal
import pickle
import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from ray.data.block import BlockAccessor
from ray.data.extensions import ArrowPythonObjectArray, ArrowTensorArray

from ray_qdrant import QdrantDatasink, write_qdrant


def _ray_supports_object_columns() -> bool:
    try:
        ArrowPythonObjectArray.from_objects([1]).to_numpy()
        return True
    except TypeError:
        # Ray 2.40's object columns break with pyarrow 21.
        return False


requires_object_columns = pytest.mark.skipif(
    not _ray_supports_object_columns(),
    reason="This Ray and pyarrow combination can't read Python object columns.",
)


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)


def make_sink(**kwargs) -> QdrantDatasink:
    params = {"collection": "c", "url": "http://localhost:6333", "api_key": "key"}
    params.update(kwargs)
    return QdrantDatasink(**params)


def make_client(collection_exists: bool = True) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = collection_exists
    return client


def write(sink: QdrantDatasink, *tables: pa.Table, client: MagicMock | None = None):
    client = client or make_client()
    with patch.object(sink, "_get_client", return_value=client):
        sink.write(list(tables), MagicMock())
    return client


def upserted(client: MagicMock) -> list:
    return [
        (c.kwargs["collection_name"], c.kwargs["points"])
        for c in client.upsert.call_args_list
    ]


class UnexpectedResponse(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"Unexpected Response: {status_code} (Error)")


class ResponseHandlingException(Exception):
    pass


class RpcError(Exception):
    def __init__(self, code: str):
        super().__init__(f"<RPC terminated with: status = StatusCode.{code}>")


class TestConstructorValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"collection": None}, "Either 'collection' or 'collection_column'"),
            ({"collection_column": "t"}, "exactly one"),
            ({"url": None}, "'url' must be provided"),
            ({"batch_size": 0}, "batch_size must be positive"),
            ({"distance": "cosine"}, "distance must be one of"),
            ({"id_column": "v", "vector_column": "v"}, "must be distinct"),
            ({"collection": None, "collection_column": "id"}, "must not be the"),
            ({"collection": None, "collection_column": "vector"}, "must not be the"),
            ({"client_kwargs": {"url": "http://x"}}, r"must not contain \['url'\]"),
            ({"client_kwargs": {"api_key": "k"}}, r"must not contain \['api_key'\]"),
            ({"client_kwargs": {"host": "h", "path": "p"}}, r"\['host', 'path'\]"),
            ({"client_kwargs": {"location": ":memory:"}}, "must not contain"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            make_sink(**kwargs)

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_API_KEY", "env-key")
        assert make_sink(api_key=None).api_key == "env-key"

    def test_api_key_is_optional(self, monkeypatch):
        monkeypatch.delenv("QDRANT_API_KEY", raising=False)
        assert make_sink(api_key=None).api_key is None

    def test_client_arguments(self):
        sink = make_sink(client_kwargs={"prefer_grpc": True})

        with patch("ray_qdrant.datasink.QdrantClient") as client_cls:
            sink._get_client()

        client_cls.assert_called_once_with(
            url="http://localhost:6333", api_key="key", prefer_grpc=True
        )


def test_write_qdrant_passes_arguments_to_sink():
    kwargs = dict(
        url="http://localhost:6333",
        collection="c",
        api_key="key",
        id_column="pk",
        vector_column="emb",
        vector_name="dense",
        batch_size=64,
        distance="Dot",
        client_kwargs={"prefer_grpc": True},
    )
    ds = MagicMock()

    write_qdrant(ds, ray_remote_args={"num_cpus": 1}, concurrency=2, **kwargs)

    [(args, call_kwargs)] = ds.write_datasink.call_args_list
    assert vars(args[0]) == vars(make_sink(**kwargs))
    assert call_kwargs == {"ray_remote_args": {"num_cpus": 1}, "concurrency": 2}


class TestWrite:
    def test_writes_ids_vectors_and_payloads(self):
        table = pa.table(
            {"id": [1, 2], "vector": [[0.1, 0.2], [0.3, 0.4]], "text": ["a", "b"]}
        )

        client = write(make_sink(), table)

        [(collection, points)] = upserted(client)
        assert collection == "c"
        assert points.ids == [1, 2]
        assert points.vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert points.payloads == [{"text": "a"}, {"text": "b"}]
        client.upsert.assert_called_once_with(
            collection_name="c", points=points, wait=True
        )

    def test_custom_columns_and_named_vector(self):
        table = pa.table({"pk": [1], "emb": [[0.1, 0.2]], "id": ["external"]})
        sink = make_sink(id_column="pk", vector_column="emb", vector_name="dense")

        [(_, points)] = upserted(write(sink, table))

        assert points.ids == [1]
        assert points.vectors == {"dense": [[0.1, 0.2]]}
        assert points.payloads == [{"id": "external"}]

    def test_no_payload_columns(self):
        table = pa.table({"id": [1], "vector": [[1.0]]})
        [(_, points)] = upserted(write(make_sink(), table))
        assert points.payloads is None

    def test_batches_by_batch_size(self):
        table = pa.table({"id": list(range(25)), "vector": [[1.0]] * 25})

        client = write(make_sink(batch_size=10), table)

        assert [len(points.ids) for _, points in upserted(client)] == [10, 10, 5]

    def test_writes_each_block(self):
        blocks = [
            pa.table({"id": [1, 2], "vector": [[1.0], [2.0]]}),
            pa.table({"id": [3], "vector": [[3.0]]}),
        ]

        client = write(make_sink(), *blocks)

        assert [points.ids for _, points in upserted(client)] == [[1, 2], [3]]

    def test_skips_empty_blocks_and_null_ids(self):
        blocks = [
            pa.table(
                {
                    "id": pa.array([], pa.int64()),
                    "vector": pa.array([], pa.list_(pa.float64())),
                }
            ),
            pa.table({"id": pa.array([None], pa.int64()), "vector": [[1.0]]}),
            pa.table({"id": pa.array([1, None], pa.int64()), "vector": [[1.0], [2.0]]}),
        ]

        client = write(make_sink(), *blocks)

        [(_, points)] = upserted(client)
        assert points.ids == [1]

    @pytest.mark.parametrize(
        "table,match",
        [
            (pa.table({"vector": [[1.0]]}), "ID column 'id' not found"),
            (pa.table({"id": [1]}), "Vector column 'vector' not found"),
            (pa.table({"id": [1, 2], "vector": [[1.0], None]}), "contains null values"),
        ],
    )
    def test_missing_or_null_columns_raise(self, table, match):
        with pytest.raises(ValueError, match=match):
            write(make_sink(), table)

    def test_closes_client(self):
        client = write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}))
        client.close.assert_called_once()

        client = make_client()
        client.upsert.side_effect = UnexpectedResponse(400)
        with pytest.raises(UnexpectedResponse):
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)
        client.close.assert_called_once()


class TestPointIds:
    def test_uuid_formats_are_normalized(self):
        u = uuid.uuid4()
        values = [str(u), u.hex, str(u).upper(), "{" + str(u) + "}"]

        assert make_sink()._to_point_ids(pa.chunked_array([values])) == [str(u)] * 4

    def test_dictionary_encoded_ids(self):
        u = str(uuid.uuid4())
        ints = pa.chunked_array([pa.array([7, 8, 7]).dictionary_encode()])
        uuids = pa.chunked_array([pa.array([u, u]).dictionary_encode()])

        assert make_sink()._to_point_ids(ints) == [7, 8, 7]
        assert make_sink()._to_point_ids(uuids) == [u, u]

    @requires_object_columns
    def test_python_object_column_with_mixed_ids(self):
        u = uuid.uuid4()
        column = ArrowPythonObjectArray.from_objects([1, u, np.uint32(3)])

        assert make_sink()._to_point_ids(pa.chunked_array([column])) == [1, str(u), 3]

    @requires_object_columns
    def test_python_object_column_skips_missing_ids(self):
        u = uuid.uuid4()
        table = pa.table(
            {
                "id": ArrowPythonObjectArray.from_objects([u, None, float("nan"), 5]),
                "vector": [[1.0], [2.0], [3.0], [4.0]],
                "row": [0, 1, 2, 3],
            }
        )

        [(_, points)] = upserted(write(make_sink(), table))

        assert points.ids == [str(u), 5]
        assert points.payloads == [{"row": 0}, {"row": 3}]

    @requires_object_columns
    def test_python_object_column_rejects_negative_ids(self):
        column = ArrowPythonObjectArray.from_objects([uuid.uuid4(), -1])

        with pytest.raises(ValueError, match="negative values"):
            make_sink()._to_point_ids(pa.chunked_array([column]))

    def test_uuid_bytes(self):
        u = uuid.uuid4()
        column = pa.chunked_array([pa.array([u.bytes], pa.binary(16))])

        assert make_sink()._to_point_ids(column) == [str(u)]

    @pytest.mark.parametrize(
        "values,match",
        [
            ([1, -1], "negative values"),
            (["not-a-uuid"], "must be unsigned integers or UUIDs"),
            (["123"], "must be unsigned integers or UUIDs"),
            ([1.5], "must be unsigned integers or UUIDs"),
            ([True], "must be unsigned integers or UUIDs"),
            ([b"short"], "must be unsigned integers or UUIDs"),
        ],
    )
    def test_invalid_ids_raise(self, values, match):
        with pytest.raises(ValueError, match=match):
            make_sink()._to_point_ids(pa.chunked_array([values]))


class TestVectors:
    @pytest.mark.parametrize(
        "vectors",
        [
            pa.array([[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]]).slice(1),
            pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float32(), 2)),
            # Older pyarrow can't build float16 values from Python floats.
            pa.LargeListArray.from_arrays(
                [0, 2, 4], pa.array(np.array([1, 2, 3, 4], np.float16))
            ),
            pa.array([[1, 2], [3, 4]]),
            ArrowTensorArray.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]])),
        ],
        ids=["sliced_list", "fixed_list", "large_list_f16", "ints", "tensor"],
    )
    def test_supported_layouts(self, vectors):
        result = make_sink()._to_vectors(pa.table({"vector": vectors}))

        assert result == [[1.0, 2.0], [3.0, 4.0]]
        assert all(type(v) is float for vector in result for v in vector)

    @pytest.mark.parametrize(
        "vectors,expected",
        [
            (pa.chunked_array([[[1.0, 2.0]], [[3.0, 4.0]]]), [[1.0, 2.0], [3.0, 4.0]]),
            (pa.chunked_array([[[1.0, 2.0], [3.0]]]), "same length"),
            (pa.chunked_array([pa.array([1.0, 2.0])]), "one-dimensional"),
        ],
        ids=["converted", "ragged", "not_lists"],
    )
    def test_fallback_when_ray_cannot_convert(self, monkeypatch, vectors, expected):
        # Simulate older pyarrow, which can't convert float16 lists to numpy.
        def to_numpy(self, columns=None):
            raise pa.ArrowNotImplementedError("halffloat")

        monkeypatch.setattr(
            type(BlockAccessor.for_block(pa.table({}))), "to_numpy", to_numpy
        )
        table = pa.table({"vector": vectors})

        if isinstance(expected, str):
            with pytest.raises(ValueError, match=expected):
                make_sink()._to_vectors(table)
        else:
            assert make_sink()._to_vectors(table) == expected

    @pytest.mark.parametrize(
        "vectors,match",
        [
            (pa.array([[1.0, 2.0], [1.0]]), "same length"),
            (pa.array([[]], pa.list_(pa.float32())), "non-empty"),
            (pa.array([["a", "b"]]), "numeric"),
            (pa.array([[True, False]]), "numeric"),
            (pa.array([[[1.0]]]), "one-dimensional"),
            (pa.array([1.0, 2.0]), "one-dimensional"),
            (ArrowTensorArray.from_numpy(np.ones((1, 2, 2))), "one-dimensional"),
            (pa.array([[1.0, None]]), "NaN or infinite"),
            (pa.array([[1.0, float("nan")]]), "NaN or infinite"),
            (pa.array([[float("inf"), 1.0]]), "NaN or infinite"),
            (pa.array([[1e300, 1.0]]), "NaN or infinite"),
        ],
    )
    def test_invalid_vectors_raise(self, vectors, match):
        with pytest.raises(ValueError, match=match):
            make_sink()._to_vectors(pa.table({"vector": vectors}))


class TestRequestSize:
    def test_small_batches_are_one_request(self):
        table = pa.table({"id": list(range(256)), "vector": [[1.0] * 1536] * 256})

        client = write(make_sink(), table)

        assert client.upsert.call_count == 1

    def test_large_payloads_are_split(self, monkeypatch):
        monkeypatch.setattr("ray_qdrant.datasink._MAX_REQUEST_BYTES", 1000)
        table = pa.table(
            {"id": list(range(10)), "vector": [[1.0]] * 10, "text": ["x" * 300] * 10}
        )

        client = write(make_sink(), table)

        batches = [points for _, points in upserted(client)]
        # Each row is ~400 bytes, so two fit in a request.
        assert [len(points.ids) for points in batches] == [2] * 5
        assert [i for points in batches for i in points.ids] == list(range(10))
        assert all(len(p.payloads) == len(p.ids) for p in batches)
        assert all(len(p.vectors) == len(p.ids) for p in batches)

    def test_split_keeps_named_vectors_and_no_payloads(self, monkeypatch):
        monkeypatch.setattr("ray_qdrant.datasink._MAX_REQUEST_BYTES", 200)
        table = pa.table({"id": [1, 2, 3], "vector": [[1.0, 2.0]] * 3})

        client = write(make_sink(vector_name="dense"), table)

        batches = [points for _, points in upserted(client)]
        assert len(batches) > 1
        assert [i for points in batches for i in points.ids] == [1, 2, 3]
        assert all(points.payloads is None for points in batches)
        assert all(set(points.vectors) == {"dense"} for points in batches)

    def test_row_larger_than_limit_is_sent_alone(self, monkeypatch):
        monkeypatch.setattr("ray_qdrant.datasink._MAX_REQUEST_BYTES", 100)
        table = pa.table({"id": [1, 2], "vector": [[1.0]] * 2, "t": ["x" * 500] * 2})

        client = write(make_sink(), table)

        assert [points.ids for _, points in upserted(client)] == [[1], [2]]


class TestPandasBlocks:
    def test_writes_pandas_block(self):
        df = pd.DataFrame({"id": [1, 2], "vector": [[1.0], [2.0]], "s": ["a", "b"]})

        [(_, points)] = upserted(write(make_sink(), df))

        assert points.ids == [1, 2]
        assert points.payloads == [{"s": "a"}, {"s": "b"}]

    def test_uuid_objects_fall_back_to_strings(self, monkeypatch):
        # Simulate older Ray versions, which can't convert UUID objects.
        real_for_block = BlockAccessor.for_block

        def for_block(block):
            if isinstance(block, pd.DataFrame) and any(
                isinstance(v, uuid.UUID) for col in block for v in block[col]
            ):
                accessor = MagicMock()
                accessor.to_arrow.side_effect = pa.ArrowInvalid("Could not convert")
                return accessor
            return real_for_block(block)

        monkeypatch.setattr(BlockAccessor, "for_block", for_block)
        u = [uuid.uuid4(), uuid.uuid4()]
        df = pd.DataFrame({"id": u, "vector": [[1.0], [2.0]], "ref": [u[1], None]})

        [(_, points)] = upserted(write(make_sink(), df))

        assert points.ids == [str(u[0]), str(u[1])]
        assert points.payloads == [{"ref": str(u[1])}, {"ref": None}]
        assert isinstance(df["id"][0], uuid.UUID)

    def test_other_conversion_errors_are_raised(self, monkeypatch):
        accessor = MagicMock()
        accessor.to_arrow.side_effect = pa.ArrowInvalid("boom")
        monkeypatch.setattr(BlockAccessor, "for_block", lambda block: accessor)

        with pytest.raises(pa.ArrowInvalid, match="boom"):
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}))


class TestPayloads:
    def test_converts_values_to_json(self):
        table = pa.table(
            {
                "id": [1],
                "vector": [[0.1]],
                "i": [1],
                "s": ["x"],
                "b": [True],
                "nan": [float("nan")],
                "nested": [{"a": [1.0, float("nan")]}],
                "date": [datetime.date(2026, 1, 2)],
                "ts": [datetime.datetime(2026, 1, 2, 3, 4, 5, 6)],
                "ts_ns": pa.array([1_700_000_000_123_456_789], pa.timestamp("ns")),
                "time": [datetime.time(1, 2, 3)],
                "tensor": ArrowTensorArray.from_numpy(np.array([[1.0, 2.0]])),
                "map": pa.array([[("k", 1)]], pa.map_(pa.string(), pa.int64())),
                "dict": pa.array(["cat"]).dictionary_encode(),
            }
        )

        assert make_sink()._to_payloads(table) == [
            {
                "i": 1,
                "s": "x",
                "b": True,
                "nan": None,
                "nested": {"a": [1.0, None]},
                "date": "2026-01-02",
                "ts": "2026-01-02T03:04:05.000006",
                "ts_ns": "2023-11-14T22:13:20.123456789",
                "time": "01:02:03",
                "tensor": [1.0, 2.0],
                "map": [["k", 1]],
                "dict": "cat",
            }
        ]

    @requires_object_columns
    def test_python_object_and_numpy_values(self):
        table = pa.table(
            {
                "id": [1, 2],
                "vector": [[0.1], [0.2]],
                "obj": ArrowPythonObjectArray.from_objects(
                    [np.float32(0.5), {"k": np.array([np.int64(1)])}]
                ),
                "tensor": ArrowTensorArray.from_numpy(np.array([[1, 2], [3, 4]])),
            }
        )

        assert make_sink()._to_payloads(table) == [
            {"obj": 0.5, "tensor": [1, 2]},
            {"obj": {"k": [1]}, "tensor": [3, 4]},
        ]

    @pytest.mark.parametrize(
        "values,match",
        [
            ([b"\x00"], "type bytes"),
            ([decimal.Decimal("1.5")], "type Decimal"),
            ([datetime.timedelta(seconds=1)], "type timedelta"),
            ([float("inf")], "infinite values"),
            ([{"a": b"x"}], "type bytes"),
        ],
    )
    def test_unsupported_values_raise(self, values, match):
        table = pa.table({"id": [1], "vector": [[0.1]], "p": values})

        with pytest.raises(ValueError, match=f"Payload column 'p' .*{match}"):
            make_sink()._to_payloads(table)


class TestCollectionCreation:
    def test_creates_missing_collection_before_first_upsert(self):
        client = write(
            make_sink(distance="Dot"),
            pa.table({"id": [1], "vector": [[1.0, 2.0]]}),
            client=make_client(collection_exists=False),
        )

        assert [c[0] for c in client.method_calls] == [
            "collection_exists",
            "create_collection",
            "upsert",
            "close",
        ]
        kwargs = client.create_collection.call_args.kwargs
        assert kwargs["collection_name"] == "c"
        assert kwargs["vectors_config"].size == 2
        assert kwargs["vectors_config"].distance == "Dot"

    def test_named_vector_config(self):
        client = write(
            make_sink(vector_name="dense"),
            pa.table({"id": [1], "vector": [[1.0, 2.0, 3.0]]}),
            client=make_client(collection_exists=False),
        )

        config = client.create_collection.call_args.kwargs["vectors_config"]
        assert config["dense"].size == 3

    def test_existing_collection_is_used_as_is(self):
        client = write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}))
        client.create_collection.assert_not_called()

    def test_checks_existence_once_per_collection(self):
        table = pa.table({"id": list(range(5)), "vector": [[1.0]] * 5})
        client = write(make_sink(batch_size=1), table, table)
        assert client.collection_exists.call_count == 1

    def test_invalid_data_creates_no_collection(self):
        client = make_client(collection_exists=False)
        table = pa.table({"id": [1], "vector": [[float("nan")]]})

        with pytest.raises(ValueError, match="NaN"):
            write(make_sink(), table, client=client)
        client.create_collection.assert_not_called()

    def test_tolerates_concurrent_creation(self):
        client = make_client()
        client.collection_exists.side_effect = [False, True]
        client.create_collection.side_effect = UnexpectedResponse(409)

        write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        client.upsert.assert_called_once()

    def test_create_failure_raises(self):
        client = make_client(collection_exists=False)
        client.create_collection.side_effect = UnexpectedResponse(403)

        with pytest.raises(UnexpectedResponse):
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)
        client.create_collection.assert_called_once()


class TestRetries:
    @pytest.mark.parametrize(
        "error",
        [
            ResponseHandlingException("Connection refused"),
            UnexpectedResponse(429),
            UnexpectedResponse(503),
            RpcError("UNAVAILABLE"),
            RpcError("DEADLINE_EXCEEDED"),
        ],
        ids=["connection", "http_429", "http_503", "grpc_unavailable", "grpc_deadline"],
    )
    def test_transient_errors_are_retried(self, error):
        client = make_client()
        client.upsert.side_effect = [error, error, None]

        write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        assert client.upsert.call_count == 3

    def test_retries_are_limited(self):
        client = make_client()
        client.upsert.side_effect = UnexpectedResponse(503)

        with pytest.raises(UnexpectedResponse):
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)
        assert client.upsert.call_count == 5

    @pytest.mark.parametrize(
        "error",
        [
            UnexpectedResponse(400),
            UnexpectedResponse(401),
            RpcError("INVALID_ARGUMENT"),
            RpcError("UNAUTHENTICATED"),
            ValueError("bad input"),
        ],
        ids=["http_400", "http_401", "grpc_invalid", "grpc_unauthenticated", "value"],
    )
    def test_permanent_errors_are_not_retried(self, error):
        client = make_client()
        client.upsert.side_effect = error

        with pytest.raises(type(error)):
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)
        client.upsert.assert_called_once()

    def test_collection_check_is_retried(self):
        client = make_client()
        client.collection_exists.side_effect = [RpcError("UNAVAILABLE"), True]

        write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        assert client.collection_exists.call_count == 2
        client.upsert.assert_called_once()


class TestMultiCollectionWrites:
    @pytest.mark.parametrize(
        "tenants",
        [
            pa.array(["b", "a", "b"]),
            pa.array(["b", "a", "b"]).dictionary_encode(),
            pa.array(["b", "a", "b"], pa.large_string()),
        ],
        ids=["string", "dictionary", "large_string"],
    )
    def test_routes_rows_and_drops_column(self, tenants):
        table = pa.table(
            {"tenant": tenants, "id": [1, 2, 3], "vector": [[1.0]] * 3, "x": [1, 2, 3]}
        )

        client = write(make_sink(collection=None, collection_column="tenant"), table)

        writes = dict(upserted(client))
        assert writes["a"].ids == [2]
        assert writes["b"].ids == [1, 3]
        assert writes["b"].payloads == [{"x": 1}, {"x": 3}]

    def test_many_collections_keep_row_order(self):
        tenants = [f"t{i % 13}" for i in range(200)]
        table = pa.table(
            {"tenant": tenants, "id": list(range(200)), "vector": [[1.0]] * 200}
        )

        client = write(make_sink(collection=None, collection_column="tenant"), table)

        writes = upserted(client)
        assert [c for c, _ in writes] == sorted(set(tenants))
        for collection, points in writes:
            assert points.ids == [i for i, t in enumerate(tenants) if t == collection]

    def test_uncastable_collection_column_raises(self):
        table = pa.table({"tenant": [[1]], "id": [1], "vector": [[1.0]]})

        with pytest.raises(ValueError, match="must contain strings"):
            write(make_sink(collection=None, collection_column="tenant"), table)

    def test_integer_collection_names_become_strings(self):
        table = pa.table({"tenant": [7], "id": [1], "vector": [[1.0]]})

        client = write(make_sink(collection=None, collection_column="tenant"), table)

        assert [collection for collection, _ in upserted(client)] == ["7"]

    @pytest.mark.parametrize(
        "table,match",
        [
            (
                pa.table({"id": [1], "vector": [[1.0]]}),
                "Collection column 'tenant' not found",
            ),
            (
                pa.table(
                    {
                        "tenant": pa.array([None], pa.string()),
                        "id": [1],
                        "vector": [[1.0]],
                    }
                ),
                "contains null values",
            ),
        ],
    )
    def test_invalid_collection_column_raises(self, table, match):
        with pytest.raises(ValueError, match=match):
            write(make_sink(collection=None, collection_column="tenant"), table)


class KeywordOnlyError(Exception):
    """Like qdrant_client's UnexpectedResponse: created with keyword arguments,
    so pickle can't recreate it from ``args``."""

    def __init__(self, status_code, reason_phrase, content, headers):
        self.status_code = status_code
        self.reason_phrase = reason_phrase
        self.content = content
        self.headers = headers

    def __str__(self):
        return f"Unexpected Response: {self.status_code} ({self.reason_phrase})"


class UnpicklableError(Exception):
    def __init__(self, *, detail):
        super().__init__()
        self.detail = detail

    def __str__(self):
        return f"failed: {self.detail}"


class TestErrorsReachTheDriver:
    def test_unexpected_response_stays_picklable(self):
        client = make_client()
        client.upsert.side_effect = KeywordOnlyError(
            status_code=400, reason_phrase="Bad Request", content=b"x", headers={}
        )

        with pytest.raises(KeywordOnlyError) as info:
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        restored = pickle.loads(pickle.dumps(info.value))
        assert str(restored) == "Unexpected Response: 400 (Bad Request)"
        assert restored.status_code == 400

    def test_unpicklable_error_becomes_runtime_error(self):
        client = make_client()
        client.upsert.side_effect = UnpicklableError(detail="boom")

        with pytest.raises(
            RuntimeError, match="UnpicklableError: failed: boom"
        ) as info:
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        assert isinstance(info.value.__cause__, UnpicklableError)
        pickle.loads(pickle.dumps(info.value))

    def test_picklable_error_is_raised_unchanged(self):
        error = ValueError("bad input")
        client = make_client()
        client.upsert.side_effect = error

        with pytest.raises(ValueError) as info:
            write(make_sink(), pa.table({"id": [1], "vector": [[1.0]]}), client=client)

        assert info.value is error


class TestSerialization:
    def test_pickle_round_trip(self):
        sink = make_sink(
            collection=None,
            collection_column="tenant",
            vector_name="dense",
            batch_size=64,
            distance="Dot",
            client_kwargs={"prefer_grpc": True},
        )

        unpickled = pickle.loads(pickle.dumps(sink))

        assert vars(unpickled) == vars(sink)
