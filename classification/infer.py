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
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path, create_dir
import torch
from torch import nn
import torch.nn.functional as F
from train_gnn import load_model
from read_graph import GraphDataset
import dgl
from dgl.dataloading import GraphDataLoader
import json
import pickle
import numpy as np
import pandas as pd
import argparse

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

def load_saved_model(weights_path: str, conf_path: str) -> nn.Module:
    """
    Loads state_dict into a torch module.
    Configuration file must match with state_dict.
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    model = load_model(conf)
    model.load_state_dict(state_dict)
    return model

def load_normalizer(norm_path: str) -> Tuple[Any]:
    """
    Returns normalizers used in training save at norm_path.
    """
    with open(norm_path, 'rb') as f:
        normalizers = pickle.load(f)
    return normalizers

def evaluate_model(model: nn.Module, loader: GraphDataLoader, device: str) -> Dict[str,np.ndarray]:
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
        prob = F.softmax(logits, dim=1).detach().numpy()[:,1].reshape(-1,1)
        probs[name[0]] = prob
    return probs

def save_probs(probs: Dict[str, np.ndarray]) -> None:
    """
    Saves probabilities in .nodes.csv files.
    It appends a column to original .nodes.csv file.
    """
    for name, prob in probs.items():
        orig = pd.read_csv(NODE_DIR + name)
        orig['prob1'] = prob
        orig.to_csv(OUTPUT_DIR + name, index=False)


def main(args):
    normalizers = load_normalizer(args.normalizers)
    eval_dataset = GraphDataset(
        node_dir=NODE_DIR, return_names=True,
        max_dist=200, max_degree=10, normalizers=normalizers)
    eval_dataloader = GraphDataLoader(eval_dataset, batch_size=1, shuffle=False)
    model = load_saved_model(args.weights, args.conf)
    model.eval()
    probs = evaluate_model(model, eval_dataloader, 'cpu')
    save_probs(probs)

if __name__=='__main__':
    args = parser.parse_args()
    NODE_DIR = parse_path(args.node_dir)
    OUTPUT_DIR = parse_path(args.output_dir)
    create_dir(args.output_dir)
    main(args)