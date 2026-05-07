import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "code"
BAM_ROOT = CODE_ROOT / "BindingAffinityModule"
for candidate in (REPO_ROOT, CODE_ROOT, BAM_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from models.model_Yang import GNN
from models.resnet import ResnetEncoderModel


DEFAULT_RESNET_BLOCKS = [16, 32, 64, 32, 16]
DEFAULT_RESNET_DEPTHS = [2, 2, 2, 2, 2]


class AttentivePooling(nn.Module):
    def __init__(self, chem_hidden_size: int, prot_hidden_size: int):
        super().__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, chem_embedding: torch.Tensor, prot_embedding: torch.Tensor):
        param = self.relu(self.param)
        param = self.dropout(param)

        wm_chem = torch.matmul(prot_embedding.unsqueeze(1), param.transpose(0, 1).unsqueeze(0))
        wm_chem = self.relu(wm_chem)
        wm_chem = self.dropout(wm_chem)

        wm_prot = torch.matmul(chem_embedding.unsqueeze(1), param.unsqueeze(0))
        wm_prot = self.relu(wm_prot)
        wm_prot = self.dropout(wm_prot)

        score_chem = F.softmax(wm_chem, dim=2)
        score_prot = F.softmax(wm_prot, dim=2)

        rep_chem = torch.sum(chem_embedding.unsqueeze(1) * score_chem, dim=1)
        rep_prot = torch.sum(prot_embedding.unsqueeze(1) * score_prot, dim=1)
        return rep_chem, rep_prot


class EmosaicBindingRegressor(nn.Module):
    """TrustAffinity-style binding affinity regressor.

    Mirrors `code/BindingAffinityModule/model.py::BindingModel2` but auto-infers the
    flattened ResNet output size from a dummy forward so the block works with any
    cached protein embedding width (e.g., ESMFold `s_s`).
    """

    def __init__(
        self,
        num_layer: int = 2,
        emb_dim: int = 300,
        dropout: float = 0.2,
        gnn_type: str = "gin",
        max_length: int = 700,
        prot_input_dim: int = 384,
    ):
        super().__init__()
        self.max_length = int(max_length)
        self.prot_input_dim = int(prot_input_dim)

        self.prot_resnet = ResnetEncoderModel(
            in_channels=1,
            blocks_sizes=DEFAULT_RESNET_BLOCKS,
            depths=DEFAULT_RESNET_DEPTHS,
            activation="relu",
        )
        self.chem_gnn = GNN(
            num_layer=num_layer,
            emb_dim=emb_dim,
            JK="last",
            drop_ratio=dropout,
            gnn_type=gnn_type,
            pretrained=True,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.max_length, self.prot_input_dim)
            out = self.prot_resnet(dummy)
            prot_hidden_size = int(out.reshape(out.shape[0], -1).shape[1])
        self.prot_hidden_size = prot_hidden_size

        self.attention = AttentivePooling(
            chem_hidden_size=emb_dim,
            prot_hidden_size=prot_hidden_size,
        )
        combined_dim = emb_dim + prot_hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        prot_emb = batch.prot_emb
        prot_mask = batch.prot_mask
        if prot_emb.dim() == 4:
            prot_emb = prot_emb.squeeze(1)    # (B, L, D)
        if prot_mask.dim() == 3:
            prot_mask = prot_mask.squeeze(1)  # (B, L) bool/float

        prot_res_inp = prot_emb.unsqueeze(dim=1)   # (B, 1, L, D)
        prot_out = self.prot_resnet(prot_res_inp)

        # mask is (B, L) — expand to (B, 1, L, 1) so F.interpolate works in 2D
        # and broadcasts correctly against prot_out (B, C, L', D')
        mask_4d = prot_mask.float().unsqueeze(1).unsqueeze(-1)     # (B, 1, L, 1)
        mask_reshaped = F.interpolate(mask_4d, prot_out.size()[2:])  # (B, 1, L', D')
        binary_mask = (mask_reshaped >= 0.5).to(prot_out.dtype)
        prot_out = prot_out * binary_mask
        prot_flat = prot_out.reshape(prot_out.shape[0], -1)

        node_rep = self.chem_gnn(batch.x, batch.edge_index, batch.edge_attr)
        dense_rep, _ = torch_geometric.utils.to_dense_batch(node_rep, batch.batch)
        chem_pooled = dense_rep.sum(dim=1)

        prot_final, chem_final = self.attention(chem_pooled, prot_flat)
        combined = torch.cat((prot_final, chem_final), dim=1)
        return self.mlp(combined).squeeze(-1)


def build_model(
    num_layer: int = 2,
    emb_dim: int = 300,
    dropout: float = 0.2,
    gnn_type: str = "gin",
    max_length: int = 700,
    prot_input_dim: int = 384,
) -> EmosaicBindingRegressor:
    return EmosaicBindingRegressor(
        num_layer=num_layer,
        emb_dim=emb_dim,
        dropout=dropout,
        gnn_type=gnn_type,
        max_length=max_length,
        prot_input_dim=prot_input_dim,
    )
