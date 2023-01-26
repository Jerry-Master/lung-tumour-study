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
import os
import sys
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)
from utils.preprocessing import parse_path, create_dir, get_names, read_centroids, save_graph
from utils.nearest import generate_tree, find_nearest

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


def merge_labels(graph: pd.DataFrame, centroids: np.ndarray) -> pd.DataFrame:
    """
    Finds 1-1 matchings of nodes and substitutes labels in graph
    by those in centroids. Internally uses KD-trees.
    """
    assert len(centroids) > 0, 'GT must contain at least one cell.'
    graph = graph.copy()
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


def main() -> None:
    names = sorted(get_names(CENTROIDS_DIR, '.centroids.csv'))
    for name in tqdm(names):
        centroids = read_centroids(name, CENTROIDS_DIR)
        graph = pd.read_csv(os.path.join(GRAPH_DIR, name + '.nodes.csv'))
        graph = merge_labels(graph, centroids)
        save_graph(graph, os.path.join(OUTPUT_DIR, name + '.nodes.csv'))
    return

if __name__ == '__main__':
    args = parser.parse_args()
    GRAPH_DIR = parse_path(args.graph_dir)
    CENTROIDS_DIR = parse_path(args.centroids_dir)
    OUTPUT_DIR = parse_path(args.output_dir)
    create_dir(OUTPUT_DIR)
    main()