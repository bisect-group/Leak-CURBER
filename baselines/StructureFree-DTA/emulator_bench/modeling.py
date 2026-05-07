import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models import Mish, ResidualInceptionBlock  # noqa: E402


class CachedAffinityPredictor(nn.Module):
    """StructureFree-DTA fusion/regressor head fed by cached encoder embeddings."""

    def __init__(
        self,
        protein_dim: int,
        molecule_dim: int,
        hidden_sizes: Sequence[int] = (1024, 768, 512, 256, 1),
        dropout: float = 0.05,
    ):
        super().__init__()
        combined_dim = int(protein_dim) + int(molecule_dim)
        self.inc1 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        self.inc2 = ResidualInceptionBlock(combined_dim, combined_dim, dropout=dropout)
        layers = []
        input_dim = combined_dim
        for output_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, int(output_dim)))
            if int(output_dim) != 1:
                layers.append(Mish())
            input_dim = int(output_dim)
        self.regressor = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        combined = torch.cat([batch["protein_embedding"], batch["molecule_embedding"]], dim=1).unsqueeze(2)
        combined = self.inc1(combined)
        combined = self.inc2(combined).squeeze(2)
        return self.regressor(self.dropout(combined)).view(-1)
