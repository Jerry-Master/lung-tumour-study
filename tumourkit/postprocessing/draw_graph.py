"""
Draws the graph into an image.
Input format: PNG / CSV
Output format: PNG

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
from tqdm import tqdm
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
import dgl
from dgl import DGLHeteroGraph
import networkx as nx
from ..utils.preprocessing import get_names
from ..preprocessing import pngcsv2graph_main
from ..classification.read_graph import GraphDataset
from ..utils.read_nodes import read_node_matrix


def get_colors(type_info: Dict[str, Tuple[str, List[int]]]) -> List[str]:
    """
    Retrieves a list of hexadecimal colors from a dictionary of RGB tuples.

    :param type_info: A dictionary containing type information as keys and RGB tuples as values.
    :type type_info: Dict[str, Tuple[str, List[int]]]
    
    :return: A list of hexadecimal colors converted from the RGB tuples.
    :rtype: List[str]
    """
    def to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
        """
        Converts an RGB tuple to a hexadecimal color representation.

        :param rgb_tuple: A tuple containing three integer values representing RGB channels.
        :type rgb_tuple: Tuple[int, int, int]

        :return: A string representing the hexadecimal color code.
        :rtype: str
        """
        hex_values = [format(value, '02x') for value in rgb_tuple]
        return '#' + ''.join(hex_values)
    return [to_hex(rgb_tuple) for name, rgb_tuple in type_info.values()]


def draw_graph(
        orig: np.ndarray,
        graph: DGLHeteroGraph,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        type_info: Dict[str, Tuple[str, List[int]]],
        save_path: str
        ) -> np.ndarray:
    """
    Draws a graph into an image using Matplotlib and NetworkX.

    :param orig: The original data used to construct the graph.
    :type orig: np.ndarray

    :param graph: The graph object to be visualized.
    :type graph: DGLHeteroGraph

    :param x: Node features used for plotting.
    :type x: np.ndarray

    :param y: Node coordinates used for plotting.
    :type y: np.ndarray

    :param labels: Node labels used for visualization.
    :type labels: np.ndarray

    :param type_info: A dictionary containing type information as keys and RGB tuples as values.
    :type type_info: Dict[str, Tuple[str, List[int]]]

    :param save_path: The file path to save the resulting graph visualization.
    :type save_path: str

    :return: The image of the graph.
    :rtype: np.ndarray
    """
    gx = dgl.to_networkx(graph)
    pos = {k: (y[k], x[k]) for k in range(len(x))}
    colors = get_colors(type_info)
    cols = [colors[label + 1] for label in labels]
    fig = plt.figure(frameon=True)
    fig.set_size_inches(4,4)
    fig.tight_layout()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    nx.draw_networkx(gx, pos=pos, arrows=False, node_color=cols, with_labels=False, node_size=10, ax=ax)
    ax.imshow(orig, aspect='auto')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, dpi=256, bbox_inches=extent)
    return orig


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    parser.add_argument('--type-info', type=str, help='Path to type_info.json.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-degree', type=int, default=10, help='Maximum degree allowed for each node.')
    parser.add_argument('--max-distance', type=int, default=200, help='Maximum allowed distance between nodes, in pixels.')
    return parser


def main_subthread(
        args: Namespace,
        name: str,
        graph_dataset: GraphDataset,
        type_info: Dict[str, Tuple[str, List[int]]],
        k: int,
        pbar: tqdm,
        ):
    orig = cv2.imread(os.path.join(args.orig_dir, name + '.png'), cv2.IMREAD_COLOR)[:, :, ::-1]
    X, y, xx, yy = read_node_matrix(
        os.path.join(args.output_dir, 'graphs', name + '.nodes.csv'), return_coordinates=True, return_class=True,
        remove_prior=False, remove_morph=False
        )
    graph = graph_dataset[k]
    draw_graph(orig, graph, xx, yy, y, type_info, os.path.join(args.output_dir, name + '.graph-overlay.png'))
    nx_graph = dgl.to_networkx(graph)
    nodes = [
        (i, {'target': str(y[i])}) for i in nx_graph.nodes()
    ]
    nx_graph.add_nodes_from(nodes)
    nx.write_gml(nx_graph, os.path.join(args.output_dir, 'graphs', name + '.gml'))
    pbar.update(1)


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    newargs = Namespace(
        png_dir=args.png_dir,
        csv_dir=args.csv_dir,
        orig_dir=args.orig_dir,
        output_path=os.path.join(args.output_dir, 'graphs'),
        num_workers=args.num_workers
    )
    pngcsv2graph_main(newargs)
    graph_dataset = GraphDataset(
        node_dir = os.path.join(args.output_dir, 'graphs'),
        max_degree = args.max_degree,
        max_dist = args.max_distance
    )
    names = sorted(get_names(args.orig_dir, '.png'))
    with open(args.type_info, 'r') as f:
        type_info = json.load(f)
    pbar = tqdm(total=len(names))
    plt.switch_backend('Agg')
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for k, name in enumerate(names):
                executor.submit(main_subthread, args, name, graph_dataset, type_info, k, pbar)
    else:
        for k, name in enumerate(names):
            main_subthread(args, name, graph_dataset, type_info, k, pbar)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
