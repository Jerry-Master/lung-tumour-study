"""
Given .nodes.csv prediction files and .centroids.csv GT files,
replaces target value of .nodes.csv with GT ones.

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
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
from argparse import Namespace
import os
from ..utils.preprocessing import parse_path, create_dir, get_names, read_centroids, save_graph
from ..utils.nearest import generate_tree, find_nearest


def merge_labels(graph: pd.DataFrame, centroids: np.ndarray) -> pd.DataFrame:
    """
    Finds 1-1 matchings of nodes and substitutes labels in graph
    by those in centroids. Internally uses KD-trees.
    """
    assert len(centroids) > 0, 'GT must contain at least one cell.'
    graph = graph.copy()
    graph['prob1'] = graph['class'] - 1
    gt_tree = generate_tree(centroids[:,:2])
    pred_centroids = graph[['X', 'Y']].to_numpy(dtype=int)
    pred_tree = generate_tree(pred_centroids)
    for point_id, point in enumerate(centroids):
        if point[2] == -1:
            continue
        closest_id = find_nearest(point[:2], pred_tree)
        closest = graph.loc[closest_id, ['X', 'Y', 'class']]
        if closest[2] == -1:
            continue
        if point_id == find_nearest(closest[:2], gt_tree):
            graph.loc[closest_id, 'class'] = point[2] # 1-1 matchings
    for point_id, point in enumerate(pred_centroids):
        closest_id = find_nearest(point[:2], gt_tree)
        closest = centroids[closest_id]
        if point_id != find_nearest(closest[:2], pred_tree):
            graph.drop(point_id, inplace=True) # Remove prediction not matched in GT
    return graph


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph-dir', type=str, required=True,
        help='Path to folder containing prediction .nodes.csv files.'
    )
    parser.add_argument(
        '--centroids-dir', type=str, required=True,
        help='Path to folder containing .centroids.csv files.'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Path to folder where to save new .nodes.csv. If same as --graph-dir, overwrites files.'
    )
    return parser


def main_with_args(args: Namespace) -> None:
    graph_dir = parse_path(args.graph_dir)
    centroids_dir = parse_path(args.centroids_dir)
    output_dir = parse_path(args.output_dir)
    create_dir(output_dir)
    names = sorted(get_names(centroids_dir, '.centroids.csv'))
    for name in tqdm(names):
        try:
            centroids = read_centroids(name, centroids_dir)
            graph = pd.read_csv(os.path.join(graph_dir, name + '.nodes.csv'))
            graph = merge_labels(graph, centroids)
            save_graph(graph, os.path.join(output_dir, name + '.nodes.csv'))
        except FileNotFoundError:
            continue
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)