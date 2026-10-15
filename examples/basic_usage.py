"""Write a Ray dataset of embeddings to Qdrant, then search it.

Start Qdrant first:

    docker run -p 6333:6333 qdrant/qdrant
"""

import numpy as np
import ray
from qdrant_client import QdrantClient

import ray_qdrant

QDRANT_URL = "http://localhost:6333"
COLLECTION = "ray_qdrant_example"
DIM = 8


def embed(batch):
    # Stand-in for a real embedding model.
    rng = np.random.default_rng(seed=int(batch["id"][0]))
    return {
        "id": batch["id"],
        "vector": rng.random((len(batch["id"]), DIM), dtype=np.float32),
        "text": [f"document {i}" for i in batch["id"]],
    }


def main():
    ray.init(ignore_reinit_error=True)

    ds = ray.data.range(1_000).map_batches(embed)
    ray_qdrant.write_qdrant(ds, url=QDRANT_URL, collection=COLLECTION)

    client = QdrantClient(url=QDRANT_URL)
    print("Points:", client.count(COLLECTION, exact=True).count)
    hits = client.query_points(COLLECTION, query=[0.5] * DIM, limit=3).points
    for hit in hits:
        print(hit.id, round(hit.score, 3), hit.payload)
    client.delete_collection(COLLECTION)


if __name__ == "__main__":
    main()
