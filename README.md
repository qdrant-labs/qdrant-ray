# ray-qdrant

Write [Ray](https://www.ray.io/) datasets to [Qdrant](https://qdrant.tech/) in parallel.

## Installation

```bash
pip install ray-qdrant
```

Requires Python 3.10+, Ray 2.40+, and Qdrant 1.16+.

## Usage

```python
import numpy as np
import ray
import ray_qdrant

ds = ray.data.range(1_000).map_batches(
    lambda batch: {
        "id": batch["id"],
        "vector": np.random.rand(len(batch["id"]), 384),
        "text": [f"document {i}" for i in batch["id"]],
    }
)

ray_qdrant.write_qdrant(ds, url="http://localhost:6333", collection="docs")
```

Each row becomes a point. By default, the `id` column holds the point ID (an unsigned integer or UUID) and the `vector` column holds the vector. Use `id_column` and `vector_column` to read them from other columns. Rows with a null ID are skipped, and every other column goes into the payload. Writes are upserts, and a missing collection is created with the vector size of the data.

### Named vectors

```python
ray_qdrant.write_qdrant(
    ds, url="http://localhost:6333", collection="docs", vector_name="dense"
)
```

### Parameters

| Parameter           | Default           | Description                                                                             |
| ------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `url`               | required          | URL of the Qdrant server.                                                               |
| `collection`        | `None`            | Collection to write to. Set this or `collection_column`.                                |
| `collection_column` | `None`            | Column with each row's collection name. Not written to the payload.                     |
| `api_key`           | `$QDRANT_API_KEY` | API key.                                                                                |
| `id_column`         | `"id"`            | Column of point IDs.                                                                    |
| `vector_column`     | `"vector"`        | Column of vectors.                                                                      |
| `vector_name`       | `None`            | Named vector to write to. Defaults to the unnamed vector.                               |
| `batch_size`        | `256`             | Maximum points per upsert request.                                                      |
| `distance`          | `"Cosine"`        | Distance metric for new collections: `"Cosine"`, `"Euclid"`, `"Dot"`, or `"Manhattan"`. |
| `client_kwargs`     | `None`            | Extra arguments for `qdrant_client.QdrantClient`, like `{"prefer_grpc": True}`.         |
| `ray_remote_args`   | `None`            | Arguments passed to `ray.remote` for the write tasks.                                   |
| `concurrency`       | `None`            | Maximum number of concurrent write tasks.                                               |

## License

Apache 2.0
