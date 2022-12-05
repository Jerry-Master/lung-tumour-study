"""

Module to create graph from nodes.

"""
from itertools import tee
from typing import Tuple, List, Optional, Callable
import sys
import os
import numpy as np

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import get_names
from utils.nearest import generate_tree
from classification.read_nodes import read_node_matrix
import torch
from torch.utils.data import Dataset
import dgl

class GraphDataset(Dataset):
    """
    Torch Dataset to load graphs from .nodes.csv files.

    Generated graph is in DGL format, with node attributes in .ndata
    and edge attributes in .edata.

    Graph edges are generated on the fly.
    """
    def __init__(self, node_dir: str, max_dist: float, max_degree: int,
        files: Optional[List[str]] = None,
        transform: Optional[Callable[[np.array], np.array]] = None):
        """
        node_dir: Path to .nodes.csv files.
        max_dist: Maximum distance to consider two nodes as neighbours.
        max_degree: Maximum degree for each node.
        files: List of names to include in the dataset. If None all names are included.
        """
        super().__init__()
        self.node_dir = node_dir
        if files is not None:
            self.node_names = set(get_names(node_dir, '.nodes.csv'))
            self.node_names = sorted(list(self.node_names.intersection(set(files))))
        else:
            self.node_names = sorted(get_names(node_dir, '.nodes.csv'))
        self.max_dist = max_dist
        self.max_degree = max_degree
        self.transform = transform

    def __getitem__(self, idx):
        file_name = self.node_names[idx] + '.nodes.csv'
        X, y, xx, yy = read_node_matrix(os.path.join(self.node_dir, file_name), return_coordinates=True)
        if self.transform is not None:
            X = self.transform(X)
        source, dest, dists = GraphDataset.create_edges(xx, yy, self.max_degree, self.max_dist)
        g = dgl.graph((source, dest), num_nodes=len(X))
        g.ndata['X'] = torch.tensor(X, dtype=torch.float32)
        g.ndata['y'] = torch.tensor(y, dtype=torch.long)
        g.edata['dist'] = torch.tensor(dists, dtype=torch.float32).reshape((-1,1))
        return g

    def __len__(self):
        return len(self.node_names)

    @staticmethod
    def create_edges(xx: List[float], yy: List[float],
        max_degree: int, threshold: float) -> Tuple[List[int], List[int]]:
        """
        Creates edges between nearby nodes.

        xx: X coordinates of nodes.
        yy: Y coordinates of nodes.
        max_degree: Maximum degree for each node.
        threshold: Maximum distance to look at.

        Returns
        source: List of source nodes id.
        dest: List of destination nodes id.
        distances: Distances between nodes in edges.
        """
        tree = generate_tree(zip(xx, yy))
        source, dest, distances = [], [], []
        for i, (x, y) in enumerate(zip(xx, yy)):
            dists, idx = tree.query((x,y), k=max_degree, distance_upper_bound=threshold)
            tmp = list(filter(lambda x: x[0] > 1e-10 and x[1] < len(xx), zip(dists, idx)))
            tmp1, tmp2 = tee(tmp)
            dists, idx = list(x[0] for x in tmp1), list(x[1] for x in tmp2)
            source.extend([i for _ in range(len(idx))])
            dest.extend(idx)
            distances.extend(dists)
        return source, dest, distances