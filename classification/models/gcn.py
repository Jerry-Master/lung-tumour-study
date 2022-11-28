"""
Semi-Supervised Classification with Graph Convolutional Networks

References
----------
Paper: https://arxiv.org/abs/1609.02907
"""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate=0.5):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConv(in_feats, h_feats, activation=F.elu))
        self.conv_layers.append(nn.Dropout(drop_rate))
        for l in range(1,num_layers):
            self.conv_layers.append(GraphConv(h_feats, h_feats, activation=F.elu))
            self.conv_layers.append(nn.Dropout(drop_rate))
        self.conv_layers.append(GraphConv(h_feats, num_classes))

    def forward(self, g, in_feat):
        h = in_feat
        for i, layer in enumerate(self.conv_layers):
            if i % 2 == 0:
                h = layer(g, h)
            else:
                h = layer(h)
        return h