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
    # Output-changing settings (dtype, pooling) that the path does not show.
    # A store written with other settings fails the metadata check. Empty means "not used".
    settings: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        if not self.settings:
            del data["settings"]
        return data
