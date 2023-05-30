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
from .train_graphs import load_model
from .read_graph import GraphDataset
import dgl
from dgl.dataloading import GraphDataLoader
import json
import pickle
import numpy as np
import pandas as pd
import argparse
import os


def load_saved_model(weights_path: str, conf_path: str, num_classes: int, num_feats: int) -> nn.Module:
    """
    Loads a saved model from the given weights_path and conf_path.

    :param weights_path: The path to the saved weights file.
    :type weights_path: str
    :param conf_path: The path to the configuration file.
    :type conf_path: str
    :param num_classes: The number of classes for the model.
    :type num_classes: int
    :param num_feats: The number of features for the model.
    :type num_feats: int
    :return: The loaded model.
    :rtype: nn.Module
    """
    state_dict = torch.load(weights_path, map_location='cpu')
    with open(conf_path, 'r') as f:
        conf = json.load(f)
    model = load_model(conf, num_classes, num_feats)
    model.load_state_dict(state_dict)
    return model


def load_normalizer(norm_path: str) -> Tuple[Any]:
    """
    Loads the normalizers used in training from the given norm_path.

    :param norm_path: The path to the saved normalizers file.
    :type norm_path: str
    :return: The loaded normalizers.
    :rtype: Tuple[Any]
    """
    with open(norm_path, 'rb') as f:
        normalizers = pickle.load(f)
    return normalizers


def run_inference(model: nn.Module, loader: GraphDataLoader, device: str, num_classes: int) -> Dict[str, np.ndarray]:
    """
    Runs inference using the specified model on the provided data loader.

    :param model: The model used for inference.
    :type model: nn.Module
    :param loader: The graph data loader.
    :type loader: GraphDataLoader
    :param device: The device used for inference (e.g., 'cpu' or 'cuda').
    :type device: str
    :param num_classes: The number of classes.
    :type num_classes: int
    :return: The probabilities for all the nodes.
    :rtype: Dict[str, np.ndarray]
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
            prob = F.softmax(logits, dim=1).detach().numpy()[:, 1].reshape(-1, 1)
        else:
            prob = F.softmax(logits, dim=1).detach().numpy()
        probs[name[0]] = prob
    return probs


def save_probs(probs: Dict[str, np.ndarray], node_dir: str, output_dir: str, num_classes: int) -> None:
    """
    Saves the probabilities in .nodes.csv files by appending a column to the original .nodes.csv file.

    :param probs: The probabilities for each graph.
    :type probs: Dict[str, np.ndarray]
    :param node_dir: The directory containing the original .nodes.csv files.
    :type node_dir: str
    :param output_dir: The directory where the updated .nodes.csv files will be saved.
    :type output_dir: str
    :param num_classes: The number of classes.
    :type num_classes: int
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
    parser.add_argument('--disable-prior', action='store_true', help='If True, remove hovernet probabilities from node features.')
    parser.add_argument('--disable-morph-feats', action='store_true', help='If True, remove morphological features from node features.')
    return parser


def main_with_args(args):
    node_dir = parse_path(args.node_dir)
    output_dir = parse_path(args.output_dir)
    create_dir(output_dir)
    normalizers = load_normalizer(args.normalizers)
    eval_dataset = GraphDataset(
        node_dir=node_dir, return_names=True, is_inference=True,
        max_dist=200, max_degree=10, normalizers=normalizers,
        remove_morph=args.disable_morph_feats, remove_prior=args.disable_prior)
    eval_dataloader = GraphDataLoader(eval_dataset, batch_size=1, shuffle=False)
    num_feats = 18 + (1 if args.num_classes == 2 else args.num_classes)
    if args.disable_morph_feats:
        num_feats -= 18
    if args.disable_prior:
        num_feats -= (1 if args.num_classes == 2 else args.num_classes)
    if num_feats == 0:
        num_feats = 1
    model = load_saved_model(args.weights, args.conf, args.num_classes, num_feats)
    model.eval()
    probs = run_inference(model, eval_dataloader, 'cpu', args.num_classes)
    save_probs(probs, node_dir, output_dir, args.num_classes)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main(args)
