"""Ray-Qdrant: write Ray Data datasets to Qdrant."""

from importlib.metadata import version

from .datasink import QdrantDatasink
from .io import write_qdrant

__version__ = version("ray-qdrant")

__all__ = ["QdrantDatasink", "write_qdrant"]
