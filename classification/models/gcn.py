"""
Semi-Supervised Classification with Graph Convolutional Networks

References
----------
Paper: https://arxiv.org/abs/1609.02907
"""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from .norm import Norm

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate=0.5):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConv(in_feats, h_feats, activation=F.elu))
        self.conv_layers.append(nn.Dropout(drop_rate)) # Feature map dropout
        self.conv_layers.append(Norm(norm_type='bn', hidden_dim=h_feats))
        for l in range(1,num_layers):
            self.conv_layers.append(GraphConv(h_feats, h_feats, activation=F.elu))
            self.conv_layers.append(nn.Dropout(drop_rate))
            self.conv_layers.append(Norm(norm_type='bn', hidden_dim=h_feats))
        self.conv_layers.append(GraphConv(h_feats, num_classes))

    def forward(self, g, in_feat):
        h = in_feat
        for i, layer in enumerate(self.conv_layers):
            if i % 3 == 1:
                h = layer(h) # Dropout
            else:
                h = layer(g, h) # Other layers
        return h