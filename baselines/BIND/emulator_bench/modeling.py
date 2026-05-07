from typing import Dict, List

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.utils import unbatch

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cross_attention_graph import CrossAttentionGraphModule
from emulator_bench.common import HEAD_ORDER


class PoolWrapper(nn.Module):
    def __init__(self, pool):
        super().__init__()
        self.pool = pool

    def forward(self, x, batch):
        unbatched_sequences = unbatch(x, batch)
        output = []
        for unbatched in unbatched_sequences:
            output.append(self.pool(unbatched))
        return torch.cat(output, dim=0)


class BindRegressionModel(nn.Module):
    def __init__(
        self,
        head_order: List[str] = None,
        cross_attention_dropout: float = 0.1,
        leaky_relu_negative_slope: float = 0.05,
    ):
        super().__init__()
        self.head_order = list(head_order or HEAD_ORDER)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)

        self.x_embed = nn.Linear(15, 16)
        self.x_embed_ln = gnn.LayerNorm(16)

        self.e_embed = nn.Linear(5, 2)
        self.e_embed_ln = nn.LayerNorm(2)

        self.conv1 = gnn.GATv2Conv(16, 64, edge_dim=2, negative_slope=0.05)
        self.ln1 = gnn.LayerNorm(64)
        self.crossattention1 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv2 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.05)
        self.ln2 = gnn.LayerNorm(64)
        self.crossattention2 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv3 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.05)
        self.ln3 = gnn.LayerNorm(64)
        self.crossattention3 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv4 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.05)
        self.ln4 = gnn.LayerNorm(64)
        self.crossattention4 = CrossAttentionGraphModule(num_heads=16, node_feature_size=64, latent_size=1280, dropout=cross_attention_dropout)

        self.conv5 = gnn.GATv2Conv(64, 64, edge_dim=2, negative_slope=0.05)
        self.ln5 = gnn.LayerNorm(64)

        self.pool = PoolWrapper(gnn.LCMAggregation(64, 1024))

        self.dense1 = nn.Linear(1024, 1024)
        self.lnf1 = nn.LayerNorm(1024)
        self.dense2 = nn.Linear(1024, 1024)
        self.lnf2 = nn.LayerNorm(1024)

        self.heads = nn.ModuleDict({head_name: nn.Linear(1024, 1) for head_name in self.head_order})

    def forward(self, graph, protein_layers, attention_mask, return_attentions: bool = False) -> Dict[str, torch.Tensor]:
        padding_mask = ~attention_mask.bool()
        protein_layers = [layer.float() for layer in protein_layers]

        graph = graph.sort()
        x, edge_index, edge_features = graph.x.float(), graph.edge_index, graph.edge_features.float()

        x = self.x_embed(x)
        x = self.leaky_relu(x)
        x = self.x_embed_ln(x, graph.batch)

        edge_features = self.e_embed(edge_features)
        edge_features = self.leaky_relu(edge_features)
        edge_features = self.e_embed_ln(edge_features)

        x = self.conv1(x, edge_index, edge_features)
        x = self.leaky_relu(x)
        x = self.ln1(x, graph.batch)
        x, av_attn1 = self.crossattention1(x, graph.batch, protein_layers[0], padding_mask, return_attention_weights=return_attentions)

        x = self.conv2(x, edge_index, edge_features)
        x = self.leaky_relu(x)
        x = self.ln2(x, graph.batch)
        x, av_attn2 = self.crossattention2(x, graph.batch, protein_layers[1], padding_mask, return_attention_weights=return_attentions)

        x = self.conv3(x, edge_index, edge_features)
        x = self.leaky_relu(x)
        x = self.ln3(x, graph.batch)
        x, av_attn3 = self.crossattention3(x, graph.batch, protein_layers[2], padding_mask, return_attention_weights=return_attentions)

        x = self.conv4(x, edge_index, edge_features)
        x = self.leaky_relu(x)
        x = self.ln4(x, graph.batch)
        x, av_attn4 = self.crossattention4(x, graph.batch, protein_layers[3], padding_mask, return_attention_weights=return_attentions)

        x = self.conv5(x, edge_index, edge_features)
        x = self.leaky_relu(x)
        x = self.ln5(x, graph.batch)

        x = self.pool(x, graph.batch)
        x = self.dense1(x)
        x = self.leaky_relu(x)
        x = self.lnf1(x)
        x = self.dense2(x)
        x = self.leaky_relu(x)
        x = self.lnf2(x)

        outputs = {head_name: head_layer(x) for head_name, head_layer in self.heads.items()}
        if return_attentions:
            return outputs, [av_attn1, av_attn2, av_attn3, av_attn4]
        return outputs


def build_model(
    head_order: List[str] = None,
    cross_attention_dropout: float = 0.1,
    leaky_relu_negative_slope: float = 0.05,
):
    return BindRegressionModel(
        head_order=head_order or HEAD_ORDER,
        cross_attention_dropout=cross_attention_dropout,
        leaky_relu_negative_slope=leaky_relu_negative_slope,
    )
