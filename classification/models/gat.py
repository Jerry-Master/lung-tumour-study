"""
Graph Attention Network.

References
----------
Paper: https://arxiv.org/abs/1710.10903
"""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, num_layers):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # two-layer GAT
        self.gat_layers.append(
            GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        for l in range(num_layers-1):
            self.gat_layers.append(
                GATConv(
                    hid_size * heads[l],
                    hid_size,
                    heads[l+1],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            GATConv(
                hid_size * heads[-2],
                out_size,
                heads[-1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers)-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h