from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


def sha256_short(value: str, length: int = 24) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def load_pickle_items(input_path: Path, key_field: str) -> list[str]:
    with open(input_path, "rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, list):
        raise ValueError(
            f"Unsupported input type {type(payload)} from {input_path}; expected list."
        )

    if not payload:
        return []

    first_item = payload[0]
    if isinstance(first_item, str):
        return payload

    if isinstance(first_item, dict):
        try:
            return [item[key_field] for item in payload]
        except KeyError as exc:
            raise ValueError(
                f"Input dictionaries in {input_path} must contain '{key_field}'."
            ) from exc

    raise ValueError(
        f"Unsupported payload element type {type(first_item)} in {input_path}."
    )
