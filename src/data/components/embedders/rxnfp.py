from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from src.utils.rich_progress import progress

from src.data.components.embedders.base import BaseShardEmbedder


class RxnFPShardEmbedder(BaseShardEmbedder):
    def __init__(self, cfg: DictConfig) -> None:
        from importlib.resources import files

        from rxnfp.tokenization import RegexTokenizer
        from transformers import BertModel

        self.batch_size = cfg.embeddings.rxnfp_batch_size

        super().__init__(
            cfg,
            input_path=cfg.embeddings.rxnfp_embeddings_input_pkl_path,
            log_file_name=cfg.embeddings.rxnfp_embeddings_log_file_name,
            embedder_name="rxnfp",
            model_name="default",
            version="v1",
            storage_dtype="float16",
            key_field="rxn_smiles",
            key_type="rxn_smiles",
            max_shard_bytes=512 * 1024 * 1024,
            compute_chunk_size=100_000,
        )

        self.logger.info("Loading RXNFP model")
        # The bundled model is loaded directly: rxnfp's own tokenizer class
        # does not tokenize SMILES under current transformers.
        path = files("rxnfp") / "models/transformers/bert_ft"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = BertModel.from_pretrained(str(path)).to(self.device).eval()
        self.max_len = self.model.config.max_position_embeddings
        self.vocab = {
            token: i for i, token in enumerate((path / "vocab.txt").read_text().splitlines())
        }
        self.tokenizer = RegexTokenizer()

    @torch.no_grad()
    def _fingerprints(self, batch: list[str]) -> np.ndarray:
        """[CLS] vectors of the reactions, with tokens beyond the model limit cut off."""
        vocab = self.vocab
        rows = [
            [vocab["[CLS]"]]
            + [vocab.get(t, vocab["[UNK]"]) for t in self.tokenizer.tokenize(rxn)][
                : self.max_len - 2
            ]
            + [vocab["[SEP]"]]
            for rxn in batch
        ]
        pad = vocab["[PAD]"]
        width = max(map(len, rows))
        input_ids = torch.tensor(
            [row + [pad] * (width - len(row)) for row in rows], device=self.device
        )
        out = self.model(input_ids=input_ids, attention_mask=(input_ids != pad).long())
        return out["last_hidden_state"][:, 0].float().cpu().numpy()

    def compute_many(
        self,
        keys: list[str],
    ) -> tuple[dict[str, np.ndarray], list[dict]]:
        arrays_by_key = {}
        failures = []
        max_len = self.max_len
        n_truncated = 0

        for start in progress(
            range(0, len(keys), self.batch_size),
            desc="Embedding RXNFP batches",
            leave=False,
        ):
            batch = keys[start : start + self.batch_size]
            # The model cuts input at max_len tokens with no flag; count them (plus [CLS] and [SEP]) and log the total.
            n_truncated += sum(
                len(self.tokenizer.tokenize(rxn_smiles)) + 2 > max_len
                for rxn_smiles in batch
            )
            try:
                batch_fps = self._fingerprints(batch)
                pairs = list(zip(batch, batch_fps))
            except Exception:
                # Retry row by row so one bad reaction does not fail the batch.
                pairs = []
                for rxn_smiles in batch:
                    try:
                        pairs.append((rxn_smiles, self._fingerprints([rxn_smiles])[0]))
                    except Exception as exc:
                        failures.append(
                            {
                                "raw_key": rxn_smiles,
                                "canonical_key": rxn_smiles,
                                "error": str(exc),
                            }
                        )
            for rxn_smiles, fp in pairs:
                arrays_by_key[rxn_smiles] = np.asarray(fp, dtype=np.float16)

        if n_truncated:
            self.logger.warning(
                f"{n_truncated} of {len(keys)} reactions exceed {max_len} tokens and were truncated."
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return arrays_by_key, failures
