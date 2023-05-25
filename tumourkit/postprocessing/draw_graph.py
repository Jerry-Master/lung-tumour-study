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
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import json
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
import dgl
from dgl import DGLHeteroGraph
from ..utils.preprocessing import get_names
from ..preprocessing import pngcsv2graph
from ..classification.read_graph import GraphDataset
from ..utils.read_nodes import read_node_matrix


def get_colors(type_info: Dict[str, Tuple[str, List[int]]]) -> List[str]:
    """
    Retrieves a list of hexadecimal colors from a list of rgb tuples.
    """
    def to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
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
    Draws graph into image using plt and networkx.
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
    draw_graph(orig, graph, xx, yy, y, type_info, os.path.join(args.output_dir, name + '.overlay.png'))
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
    pngcsv2graph(newargs)
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
