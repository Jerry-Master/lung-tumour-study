"""
Semi-Supervised Classification with Graph Convolutional Networks, 
with some minor modifications.

References
----------
Paper: https://arxiv.org/abs/1609.02907

Copyright (C) 2023  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact information: joseperez2000@hotmail.es
"""

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from .norm import Norm

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_layers, drop_rate, norm_type):
        super(GCN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConv(in_feats, h_feats, activation=F.elu))
        self.conv_layers.append(nn.Dropout(drop_rate)) # Feature map dropout
        self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))
        for l in range(1,num_layers):
            self.conv_layers.append(GraphConv(h_feats, h_feats, activation=F.elu))
            self.conv_layers.append(nn.Dropout(drop_rate))
            self.conv_layers.append(Norm(norm_type=norm_type, hidden_dim=h_feats))
        self.conv_layers.append(GraphConv(h_feats, num_classes))

    def forward(self, g, in_feat):
        h = in_feat
        for i, layer in enumerate(self.conv_layers):
            if i % 3 == 1:
                h = layer(h) # Dropout
            else:
                h = layer(g, h) # Other layers
        return h