"""
Converts the .nodes.csv files into .centroids.csv files.

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
import argparse
from argparse import Namespace
from ..utils.preprocessing import get_names, create_dir, parse_path, save_centroids, read_graph
import pandas as pd
import numpy as np
from tqdm import tqdm


def graph2centroids(graph_file: pd.DataFrame) -> np.ndarray:
    """
    Extracts X, Y and class attributes from graphs nodes.

    :param graph_file: The Pandas DataFrame containing the graph nodes.
    :type graph_file: pd.DataFrame
    :return: A NumPy array containing the X, Y and class attributes from the graph nodes.
    :rtype: np.ndarray

    If the graph file has a 'class' column, the function returns the 'X', 'Y' and 'class' columns as a NumPy array with integer data type.
    Otherwise, the function assumes that the graph file has a 'prob1' column, which contains the probability of a node belonging to a particular class.
    In this case, the function sets the class attribute to 1 if the probability is greater than 0.5, and to 2 otherwise.
    """
    if 'class' in graph_file.columns:
        return graph_file[['X', 'Y', 'class']].to_numpy(dtype=int)
    res = graph_file[['X', 'Y', 'prob1']].to_numpy()
    res[:, 2] = (res[:, 2] > 0.5) * 1 + 1
    return np.array(res, dtype=int)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, required=True, help='Path to folder containing .nodes.csv.')
    parser.add_argument('--centroids-dir', type=str, required=True, help='Path to folder where to save .centroids.csv.')
    return parser


def main_with_args(args: Namespace) -> None:
    graph_dir = parse_path(args.graph_dir)
    centroids_dir = parse_path(args.centroids_dir)
    create_dir(centroids_dir)
    names = get_names(graph_dir, '.nodes.csv')
    for name in tqdm(names):
        graph_file = read_graph(name, graph_dir)
        centroids_file = graph2centroids(graph_file)
        save_centroids(centroids_file, centroids_dir, name)
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)