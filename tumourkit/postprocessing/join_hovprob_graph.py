"""
Extract HoVer-Net probabilities from json and
concatenates the result as new column (prob1) to .nodes.csv.

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
from tqdm import tqdm
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import argparse
from argparse import Namespace
import os
from ..utils.preprocessing import create_dir, parse_path, get_names, read_json, save_graph
from ..utils.nearest import generate_tree, find_nearest
import logging
from logging import Logger


def parse_centroids_probs(nuc: Dict[str, Any], logger: Optional[Logger] = None, num_classes: Optional[int] = 2) -> List[Tuple[int, int, int]]:
    """
    Parses the centroids and probabilities from the Hovernet JSON nuclei dictionary.

    This function takes the Hovernet JSON nuclei dictionary as input, which is obtained from the modified run_infer.py script.
    It extracts the centroids and probabilities from the dictionary and returns them as a list of tuples.
    Each tuple represents a centroid and consists of (X, Y, prob1) for binary classification or (X, Y, prob1, prob2, ..., probN) for multiclass classification.

    :param nuc: The Hovernet JSON nuclei dictionary.
    :type nuc: Dict[str, Any]
    :param logger: Optional logger object to log warnings.
    :type logger: Optional[Logger]
    :param num_classes: Optional number of classes. Defaults to 2 for binary classification.
    :type num_classes: Optional[int]
    :return: The list of centroids with probabilities.
    :rtype: List[Tuple[int, int, int]]
    """
    centroids_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_centroid = inst_info['centroid']
        inst_prob1 = inst_info['prob1']
        if num_classes > 2:
            inst_probs = []
            for k in range(1, num_classes + 1):
                inst_probs.append(inst_info['prob' + str(k)])
        inst_type = inst_info['type']
        if inst_type == 0:
            if logger is None:
                logging.warning('Found cell with class 0, removing it.')
            else:
                logger.warning('Found cell with class 0, removing it.')
        else:
            if num_classes == 2:
                centroids_.append((inst_centroid[1], inst_centroid[0], inst_prob1))
            else:
                centroids_.append((inst_centroid[1], inst_centroid[0], *inst_probs))
    return centroids_


def add_probability(
        graph: pd.DataFrame,
        hov_json: Dict[str, Any],
        logger: Optional[Logger] = None,
        num_classes: Optional[int] = 2,
        enable_background: Optional[bool] = False,
        ) -> pd.DataFrame:
    """
    Adds probability information from the Hovernet JSON nuclei dictionary to the graph DataFrame.

    This function extracts the type probabilities from the Hovernet JSON nuclei dictionary (`hov_json`) and adds them as columns to the `graph` DataFrame.
    The join between the `graph` DataFrame and the `hov_json` dictionary is based on the 'id' field.

    :param graph: The graph DataFrame containing the nodes.
    :type graph: pd.DataFrame
    :param hov_json: The Hovernet JSON nuclei dictionary.
    :type hov_json: Dict[str, Any]
    :param logger: Optional logger object to log warnings.
    :type logger: Optional[Logger]
    :param num_classes: Optional number of classes. Defaults to 2 for binary classification.
    :type num_classes: Optional[int]
    :param enable_background: If True, extra cells are included. They are represented with a value of 1 in the column background.
    :type enable_background: Optional[bool]
    :return: The updated graph DataFrame with probability information.
    :rtype: pd.DataFrame
    """
    graph = graph.copy()
    if 'prob1' not in graph.columns:
        n_cols = len(graph.columns)
        graph.insert(n_cols, 'prob1', [-1] * len(graph))
    else:
        graph['prob1'] = -1
    if enable_background:
        if 'background' not in graph.columns:
            n_cols = len(graph.columns)
            graph.insert(n_cols, 'background', [0] * len(graph))
        else:
            graph['background'] = 0
    if num_classes > 2:
        for k in range(2, num_classes + 1):
            if not 'prob' + str(k) in graph.columns:
                n_cols = len(graph.columns)
                graph.insert(n_cols, 'prob' + str(k), [-1] * len(graph))
            else:
                graph['prob' + str(k)] = -1
    pred_centroids = parse_centroids_probs(hov_json, logger, num_classes)
    pred_centroids = np.array(pred_centroids)
    assert len(pred_centroids) > 0, 'Hov json must contain at least one cell.'
    pred_tree = generate_tree(pred_centroids[:, :2])
    gt_centroids = graph[['X', 'Y']].to_numpy(dtype=int)
    gt_tree = generate_tree(gt_centroids)
    for point_id, point in enumerate(pred_centroids):
        closest_id = find_nearest(point[:2], gt_tree)
        closest = graph.loc[closest_id, ['X', 'Y', 'prob1']]
        if point_id == find_nearest(closest[:2], pred_tree):
            graph.loc[closest_id, 'prob1'] = point[2]  # 1-1 matchings
            if num_classes > 2:
                for k in range(2, num_classes + 1):
                    graph.loc[closest_id, 'prob' + str(k)] = point[k + 1]  # 1-1 matchings
        elif enable_background:
            graph.loc[closest_id, 'background'] = 1  # Pred not in GT
            graph.loc[closest_id, 'prob1'] = point[2]  # closest matching
            if num_classes > 2:
                for k in range(2, num_classes + 1):
                    graph.loc[closest_id, 'prob' + str(k)] = point[k + 1]  # closest matching
    graph.drop(graph[graph['prob1'] == -1].index, inplace=True)
    return graph


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json-dir', type=str, required=True,
        help='Path to folder containing HoVer-Net json outputs.'
    )
    parser.add_argument(
        '--graph-dir', type=str, required=True,
        help='Path to directory to .nodes.csv containing graph information.'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Path where to save new .nodes.csv. If same as --graph-dir, overwrites its content.'
    )
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--enable-background', action='store_true', help='If enabled, GNNs are allowed to predict the class 0 (background) and correct extra cells.')
    return parser


def main_with_args(args: Namespace, logger: Optional[Logger] = None) -> None:
    json_dir = parse_path(args.json_dir)
    graph_dir = parse_path(args.graph_dir)
    output_dir = parse_path(args.output_dir)
    create_dir(output_dir)
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        try:
            graph = pd.read_csv(os.path.join(graph_dir, name + '.nodes.csv'))
            hov_json = read_json(os.path.join(json_dir, name + '.json'))
            graph = add_probability(graph, hov_json, logger, args.num_classes, args.enable_background)
            save_graph(graph, os.path.join(output_dir, name + '.nodes.csv'))
        except FileNotFoundError:
            continue
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
