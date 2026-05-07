from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from emulator_bench.common import embedding_cache_path, ensure_parent, normalize_sequence


def resolve_amp_dtype(device: torch.device) -> Tuple[Optional[torch.dtype], str]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed"
    return torch.float16, "fp16-mixed"


def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_encoder(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def embed_text_batch(
    texts: Sequence[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    autocast_dtype: Optional[torch.dtype],
) -> np.ndarray:
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
    with torch.no_grad(), autocast_context(device, autocast_dtype):
        output = model(**encoded)
        pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return pooled.detach().cpu().float().numpy()


def save_embedding(path: Path, embedding: np.ndarray, dtype: str) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    target_dtype = np.float16 if dtype == "float16" else np.float32
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, embedding=embedding.astype(target_dtype, copy=False), dim=np.asarray([embedding.shape[-1]], dtype=np.int32))
    tmp_path.replace(path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class EmbeddingStore:
    def __init__(
        self,
        embeddings_dir: Path,
        kind: str,
        model_name: str,
        max_length: int,
        values: Optional[Sequence[str]] = None,
        preload: bool = False,
        max_items: int = 4096,
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.kind = kind
        self.model_name = model_name
        self.max_length = int(max_length)
        self.max_items = max(1, int(max_items))
        self._cache: Dict[str, Dict[str, np.ndarray]] = {}
        if preload and values is not None:
            for value in sorted(set(self._normalize(value) for value in values)):
                self._cache[value] = self._load(value)

    def _normalize(self, value: str) -> str:
        return normalize_sequence(value) if self.kind == "proteins" else str(value).strip()

    def path_for(self, value: str) -> Path:
        return embedding_cache_path(self.embeddings_dir, self.kind, self._normalize(value), self.model_name, self.max_length)

    def _load(self, value: str) -> Dict[str, np.ndarray]:
        path = self.path_for(value)
        if not path.exists():
            raise FileNotFoundError(f"Missing cached {self.kind[:-1]} embedding: {path}")
        return load_npz(path)

    def get(self, value: str) -> Dict[str, np.ndarray]:
        key = self._normalize(value)
        if key in self._cache:
            return self._cache[key]
        item = self._load(key)
        self._cache[key] = item
        if len(self._cache) > self.max_items:
            first_key = next(iter(self._cache))
            self._cache.pop(first_key, None)
        return item
