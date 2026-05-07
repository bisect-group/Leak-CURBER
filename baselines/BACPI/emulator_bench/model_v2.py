"""
BACPIv2 — memory-efficient variant of the original BACPI architecture.

Changes vs code/model.py:
  1. GATLayerV2: replaces repeat_interleave O(B·N²·2D) expansion with
     broadcast addition O(B·N²). Mathematically identical to the original
     because e[i,j] = leaky_relu([Wh_i||Wh_j]@a) = leaky_relu(Wh_i@a_left + Wh_j@a_right).
     For N=100 atoms, D=50: peak intermediate goes from ~16 M floats to ~160 K floats per head.
     This is the primary source of VRAM fluctuation in the original.

  2. Joint mask product for bidirectional attention precomputed once outside
     the bidat loop instead of being recomputed 4×.

  3. Minor cleanup: nn.ModuleList for GAT layers (original used add_module),
     removed unused layer_out attribute.

Forward signature is identical to the original BACPI class. Weight keys differ
(a_left/a_right vs a; gat_layers.N.* vs gat_layer_N.*) so checkpoints are not
directly transferable — use convert_v1_state_dict() if needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayerV2(nn.Module):
    """Graph Attention Layer using broadcast-based attention scoring.

    The attention coefficient a ∈ R^{2D} from the original GAT is split into
    a_left ∈ R^D (source nodes) and a_right ∈ R^D (target nodes).  The scores
    are then:
        e[i,j] = leaky_relu(Wh[i] @ a_left + Wh[j] @ a_right)
    which equals the original [Wh[i]||Wh[j]] @ a, computed without any N²·D
    intermediate tensor.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

        self.a_left = nn.Parameter(torch.empty(out_features, 1))
        self.a_right = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.a_left, gain=1.414)
        nn.init.xavier_uniform_(self.a_right, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = torch.matmul(h, self.W)  # [B, N, D]

        # [B, N, 1] + [B, 1, N] → [B, N, N] without any N²·D materialization
        e = F.leaky_relu(
            torch.matmul(Wh, self.a_left) + torch.matmul(Wh, self.a_right).transpose(1, 2),
            self.alpha,
        )

        zero_vec = -9e15 * torch.ones_like(e)
        # adj arrives as bool from the collate fn; bool() is a no-op on bool tensors
        # but handles the case where an int adjacency is passed (e.g. during testing)
        attention = torch.where(adj.bool(), e, zero_vec)
        attention = F.softmax(attention, dim=2)

        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime) if self.concat else h_prime


class BACPIv2(nn.Module):
    """BACPI with memory-efficient GAT and an otherwise training-identical forward pass."""

    def __init__(self, task: str, n_atom: int, n_amino: int, params):
        super().__init__()

        comp_dim = params.comp_dim
        prot_dim = params.prot_dim
        gat_dim = params.gat_dim
        num_head = params.num_head
        dropout = params.dropout
        alpha = params.alpha
        window = params.window
        layer_cnn = params.layer_cnn
        latent_dim = params.latent_dim

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.bidat_num = 4

        # Embeddings
        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino + 1, prot_dim)

        # Compound: multi-head GAT + output GAT + projection
        self.gat_layers = nn.ModuleList([
            GATLayerV2(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(num_head)
        ])
        self.gat_out = GATLayerV2(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        # Protein: 2D CNN over [L, D] (Conv2d preserved from original — not a bug;
        # the square kernel intentionally captures both sequence and embedding correlations)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 1, kernel_size=2 * window + 1, stride=1, padding=window)
            for _ in range(layer_cnn)
        ])
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        # Morgan fingerprint MLP (1024 → latent_dim → latent_dim)
        self.fp0 = nn.Parameter(torch.empty(1024, latent_dim))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(latent_dim, latent_dim))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        # Bidirectional attention (4 layers)
        self.U = nn.ParameterList([
            nn.Parameter(torch.empty(latent_dim, latent_dim)) for _ in range(self.bidat_num)
        ])
        for p in self.U:
            nn.init.xavier_uniform_(p, gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)

        if task == "affinity":
            self.output = nn.Linear(latent_dim * latent_dim * 2, 1)
        elif task == "interaction":
            self.output = nn.Linear(latent_dim * latent_dim * 2, 2)
        else:
            raise ValueError("task must be 'affinity' or 'interaction', got: %s" % task)

    # ------------------------------------------------------------------
    # Sub-networks
    # ------------------------------------------------------------------

    def comp_gat(self, atoms: torch.Tensor, atoms_mask: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino: torch.Tensor, amino_mask: torch.Tensor) -> torch.Tensor:
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = amino_vector.unsqueeze(1)  # [B, 1, L, D]
        for conv in self.conv_layers:
            amino_vector = F.leaky_relu(conv(amino_vector), self.alpha)
        amino_vector = amino_vector.squeeze(1)    # [B, L, D]
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector

    def mask_softmax(self, a: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max) * mask
        return a_exp / (a_exp.sum(dim, keepdim=True) + 1e-6)

    def bidirectional_attention_prediction(
        self,
        atoms_vector: torch.Tensor,
        atoms_mask: torch.Tensor,
        fps: torch.Tensor,
        amino_vector: torch.Tensor,
        amino_mask: torch.Tensor,
    ) -> torch.Tensor:
        b = atoms_vector.shape[0]

        # Precompute joint mask once (original recomputed this 4× inside the loop)
        mask_joint = torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

        cat_cf = cat_pf = None
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            A = A * mask_joint

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), atoms_mask.view(b, -1))
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), amino_mask.view(b, -1))

            cf = torch.sum(atoms_vector * atoms_att.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_att.view(b, -1, 1), dim=1)

            if cat_cf is None:
                cat_cf, cat_pf = cf, pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        cf_final = torch.cat([self.comb_c(cat_cf).view(b, -1), fps.view(b, -1)], dim=1)
        pf_final = self.comb_p(cat_pf)
        cf_pf = F.leaky_relu(
            torch.matmul(cf_final.view(b, -1, 1), pf_final.view(b, 1, -1)).view(b, -1),
            0.1,
        )
        return self.output(cf_pf)

    def forward(
        self,
        atoms: torch.Tensor,
        atoms_mask: torch.Tensor,
        adjacency: torch.Tensor,
        amino: torch.Tensor,
        amino_mask: torch.Tensor,
        fps: torch.Tensor,
    ) -> torch.Tensor:
        atoms_vector = self.comp_gat(atoms, atoms_mask, adjacency)
        amino_vector = self.prot_cnn(amino, amino_mask)
        super_feature = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        super_feature = F.leaky_relu(torch.matmul(super_feature, self.fp1), 0.1)
        return self.bidirectional_attention_prediction(atoms_vector, atoms_mask, super_feature, amino_vector, amino_mask)


# ------------------------------------------------------------------
# Checkpoint conversion helper
# ------------------------------------------------------------------

def convert_v1_state_dict(v1_state: dict) -> dict:
    """Convert a state dict from the original BACPI (code/model.py) to BACPIv2.

    The only structural difference is in the GAT layers:
      v1: gat_layer_N.W, gat_layer_N.a
      v2: gat_layers.N.W, gat_layers.N.a_left, gat_layers.N.a_right

    The output GAT layer follows the same pattern (gat_out.*).
    All other keys are identical.
    """
    v2_state = {}
    for k, v in v1_state.items():
        # gat_layer_N.* → gat_layers.N.*
        if k.startswith("gat_layer_") and not k.startswith("gat_layer_out"):
            # e.g. "gat_layer_0.W" → "gat_layers.0.W"
            rest = k[len("gat_layer_"):]       # "0.W" or "0.a"
            idx, attr = rest.split(".", 1)
            if attr == "a":
                D = v.shape[0] // 2
                v2_state["gat_layers.%s.a_left" % idx] = v[:D].clone()
                v2_state["gat_layers.%s.a_right" % idx] = v[D:].clone()
            else:
                v2_state["gat_layers.%s.%s" % (idx, attr)] = v
        # gat_out.a → gat_out.a_left / gat_out.a_right
        elif k == "gat_out.a":
            D = v.shape[0] // 2
            v2_state["gat_out.a_left"] = v[:D].clone()
            v2_state["gat_out.a_right"] = v[D:].clone()
        else:
            v2_state[k] = v
    return v2_state
