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


def parse_centroids_probs(nuc: Dict[str, Any], logger: Optional[Logger] = None, num_classes: Optional[int] = 2) -> List[Tuple[int,int,int]]:
    """
    Input: Hovernet json nuclei dictionary as given by modified run_infer.py.
    Output: List of (X,Y,prob1) tuples representing centroids.
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


def add_probability(graph: pd.DataFrame, hov_json: Dict[str, Any], logger: Optional[Logger] = None, num_classes: Optional[int] = 2) -> pd.DataFrame:
    """
    Extracts type_prob from json and adds it as column prob1.
    Makes the join based on id.
    """
    centroids = parse_centroids_probs(hov_json, logger, num_classes)
    centroids = np.array(centroids)
    assert len(centroids) > 0, 'Hov json must contain at least one cell.'
    graph = graph.copy()
    if not 'prob1' in graph.columns:
        n_cols = len(graph.columns)
        graph.insert(n_cols, 'prob1', [-1] * len(graph))
    else:
        graph['prob1'] = -1
    if num_classes > 2:
        for k in range(2, num_classes + 1):
            if not 'prob' + str(k) in graph.columns:
                n_cols = len(graph.columns)
                graph.insert(n_cols, 'prob' + str(k), [-1] * len(graph))
            else:
                graph['prob' + str(k)] = -1
    gt_tree = generate_tree(centroids[:,:2])
    pred_centroids = graph[['X', 'Y']].to_numpy(dtype=int)
    pred_tree = generate_tree(pred_centroids)
    for point_id, point in enumerate(centroids):
        closest_id = find_nearest(point[:2], pred_tree)
        closest = graph.loc[closest_id, ['X', 'Y', 'prob1']]
        if point_id == find_nearest(closest[:2], gt_tree):
            graph.loc[closest_id, 'prob1'] = point[2] # 1-1 matchings
            if num_classes > 2:
                for k in range(2, num_classes + 1):
                    graph.loc[closest_id, 'prob' + str(k)] = point[k + 1] # 1-1 matchings
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
            graph = add_probability(graph, hov_json, logger, args.num_classes)
            save_graph(graph, os.path.join(output_dir, name + '.nodes.csv'))
        except FileNotFoundError:
            continue
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
