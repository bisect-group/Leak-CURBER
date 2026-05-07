from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class CacheMetadata:
    embedder: str
    model_name: str
    version: str
    storage_dtype: str
    embedding_dim: Optional[int]
    max_shard_bytes: int
    key_type: str
    key_field: str

    def to_dict(self) -> dict:
        return asdict(self)
