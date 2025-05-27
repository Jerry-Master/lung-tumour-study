"""
Draws Vietoris-Rips complexes over the images.
Input format: PNG / CSV
Output format: PNG

Copyright (C) 2025  Jose Pérez Cano

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
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from gudhi import RipsComplex
import networkx as nx
import dgl
from dgl import DGLHeteroGraph
from ..utils.preprocessing import get_names
from ..preprocessing import pngcsv2graph_main
from ..utils.read_nodes import read_node_matrix
from ..classification.read_graph import GraphDataset
from .draw_graph import get_colors


def compute_graph_distance_matrix(graph: DGLHeteroGraph) -> np.ndarray:
    """
    Compute the unweighted graph shortest-path distance matrix.

    :param graph: Graph of the cells.
    :type graph: DGLHeteroGraph
    
    :return: A 2D numpy matrix with shortest path distances.
    :rtype: np.ndarray
    """
    gx = dgl.to_networkx(graph)
    n = gx.number_of_nodes()
    # Initialize distance matrix with infinity
    D = np.full((n, n), np.inf)
    # Fill in the shortest path lengths
    length_dict = dict(nx.all_pairs_shortest_path_length(gx))
    for i in range(n):
        D[i, i] = 0  # distance to self is zero
        for j, d in length_dict[i].items():
            D[i, j] = d
    return D


def draw_rips(
        orig: np.ndarray,
        D: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        labels: np.ndarray,
        type_info: Dict[str, Tuple[str, List[int]]],
        save_path: str,
        epsilon: float,
        use_metric: bool
        ) -> np.ndarray:
    """
    Draws the Vietoris Rips diagram for a given theshold.

    :param orig: The original data used to construct the graph.
    :type orig: np.ndarray

    :param D: Distance matrix.
    :type D: np.ndarray

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

    :param epsilon: The threshold for the complex creation.
    :type epsilon: float

    :param use_metric: If true use points, when false use graph.
    :type use_metric: bool

    :return: The image of the graph.
    :rtype: np.ndarray
    """
    # Prepare coordinates and colors
    coords = np.stack((x, y), axis=1)
    pos = {i: (y[i], x[i]) for i in range(len(x))}  # (row, col) for plotting
    colors = get_colors(type_info)
    cols = [colors[label + 1] for label in labels]
    # Set up plot
    fig = plt.figure(frameon=True)
    fig.set_size_inches(4, 4)
    fig.tight_layout()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # Create Vietoris-Rips complex from GUDHI
    if use_metric:
        # Using euclidean distance
        rips = RipsComplex(points=coords, max_edge_length=epsilon)
    else:
        # Using graph distance
        rips = RipsComplex(distance_matrix=D, max_edge_length=epsilon)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    # Build 1-skeleton as a networkx.Graph
    G_rips = nx.Graph()
    for simplex in simplex_tree.get_skeleton(1):
        vertices, _ = simplex
        if len(vertices) == 1:
            G_rips.add_node(vertices[0])
        elif len(vertices) == 2:
            G_rips.add_edge(vertices[0], vertices[1])
    # Plot the 1-skeleton graph
    rips_node_colors = [cols[i] for i in G_rips.nodes]
    nx.draw_networkx(G_rips, pos=pos, arrows=False, node_color=rips_node_colors, with_labels=False, node_size=10, ax=ax)
    # Draw triangles from Vietoris-Rips complex
    for simplex in simplex_tree.get_skeleton(2):
        vertices, _ = simplex
        if len(vertices) == 3:
            triangle = [pos[i] for i in vertices]
            poly = Polygon(triangle, closed=True, facecolor='red', alpha=0.3, edgecolor='none')
            ax.add_patch(poly)
    # Plot background image
    ax.imshow(orig, aspect='auto')
    # Save figure
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path, dpi=256, bbox_inches=extent)
    plt.close(fig)
    return orig


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
    os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
    if args.use_metric:
        _range = np.linspace(1, args.max_distance, 5)
        D = None
    else:
        _range = np.linspace(1, 3, 3)
        D = compute_graph_distance_matrix(graph)
    for eps in _range:
        draw_rips(
            orig, D,
            xx, yy, y, type_info,
            os.path.join(args.output_dir, name, f'{eps}.{"metric" if args.use_metric else "graph"}.vr-complex.png'),
            eps, args.use_metric)
    pbar.update(1)


def main_with_args(args: Namespace) -> None:
    graph_dir = os.path.join(args.output_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
    names = sorted(get_names(args.orig_dir, '.png'))
    # Avoid repeated computation
    if args.force_overwrite or len(os.listdir(graph_dir)) != len(names):
        newargs = Namespace(
            png_dir=args.png_dir,
            csv_dir=args.csv_dir,
            orig_dir=args.orig_dir,
            output_path=graph_dir,
            num_workers=args.num_workers
        )
        pngcsv2graph_main(newargs)
    graph_dataset = GraphDataset(
        node_dir = os.path.join(args.output_dir, 'graphs'),
        max_degree = args.max_degree,
        max_dist = args.max_distance
    )
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


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    parser.add_argument('--type-info', type=str, help='Path to type_info.json.')
    parser.add_argument('--use-metric', action='store_true', help='True: use euclidean distance between nodes. False: use graph distance between nodes.')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--force-overwrite', action='store_true', help='By default if the output folder already contains graph subfolder it avoid computing graphs. Activate to overwrite the .nodes.csv there.')
    parser.add_argument('--max-degree', type=int, default=10, help='Maximum degree allowed for each node.')
    parser.add_argument('--max-distance', type=int, default=200, help='Maximum allowed distance between nodes, in pixels.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
