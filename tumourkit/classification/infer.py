"""
Script to generate predictions from given GNN model.

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
from typing import Dict, Tuple, Any
from ..utils.preprocessing import parse_path, create_dir
import torch
from torch import nn
import torch.nn.functional as F
from .train_gnn import load_model
from .read_graph import GraphDataset
import dgl
from dgl.dataloading import GraphDataLoader
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import os
                  

def load_saved_model(weights_path: str, conf_path: str, num_classes: int) -> nn.Module:
    """
    Loads state_dict into a torch module.
    Configuration file must match with state_dict.
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    model = load_model(conf, num_classes)
    model.load_state_dict(state_dict)
    return model

def load_normalizer(norm_path: str) -> Tuple[Any]:
    """
    Returns normalizers used in training save at norm_path.
    """
    with open(norm_path, 'rb') as f:
        normalizers = pickle.load(f)
    return normalizers

def run_inference(model: nn.Module, loader: GraphDataLoader, device: str, num_classes: int) -> Dict[str, np.ndarray]:
    """
    Computes probabilities for all the nodes.
    """
    probs = {}
    for g, name in loader:
        # self-loops
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        # data
        features = g.ndata['X'].to(device)
        # Forward
        logits = model(g, features)
        if num_classes == 2:
            prob = F.softmax(logits, dim=1).detach().numpy()[:,1].reshape(-1,1)
        else:
            prob = F.softmax(logits, dim=1).detach().numpy()
        probs[name[0]] = prob
    return probs

def save_probs(probs: Dict[str, np.ndarray], node_dir: str, output_dir: str, num_classes: int) -> None:
    """
    Saves probabilities in .nodes.csv files.
    It appends a column to original .nodes.csv file.
    """
    for name, prob in probs.items():
        orig = pd.read_csv(os.path.join(node_dir, name))
        if num_classes == 2:
            orig['prob1'] = prob
        else:
            for k in range(1, num_classes + 1):
                orig['prob' + str(k)] = prob[:, (k - 1)]
        orig.to_csv(output_dir + name, index=False)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-dir', type=str, required=True,
                        help='Folder containing .nodes.csv')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Folder to save .nodes.csv containing probabilities.')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights.')  
    parser.add_argument('--conf', type=str, required=True,
                        help='Configuration file for the model.')
    parser.add_argument('--normalizers', type=str, required=True,
                        help='Path to normalizer objects for the model.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args):
    node_dir = parse_path(args.node_dir)
    output_dir = parse_path(args.output_dir)
    create_dir(output_dir)
    normalizers = load_normalizer(args.normalizers)
    eval_dataset = GraphDataset(
        node_dir=node_dir, return_names=True, is_inference=True,
        max_dist=200, max_degree=10, normalizers=normalizers)
    eval_dataloader = GraphDataLoader(eval_dataset, batch_size=1, shuffle=False)
    model = load_saved_model(args.weights, args.conf, args.num_classes)
    model.eval()
    probs = run_inference(model, eval_dataloader, 'cpu', args.num_classes)
    save_probs(probs, node_dir, output_dir, args.num_classes)

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main(args)