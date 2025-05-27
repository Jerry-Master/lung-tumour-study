"""
Draws the persistence diagram of the cells.
Input format: PNG / CSV
Output format: PNG (matplotlib plot), CSV (births and deaths)

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
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from gudhi.sklearn.rips_persistence import RipsPersistence
from ..utils.preprocessing import get_names
from ..preprocessing import pngcsv2graph_main
from ..utils.read_nodes import read_node_matrix
from ..classification.read_graph import GraphDataset
from .draw_rips import compute_graph_distance_matrix


def draw_barcode(
        rips_persistence: RipsPersistence,
        X: Any,
        max_edge_length: float,
        max_dim: int,
        save_path: str
        ):
    """
    Draws the persistence diagram and saves it into csv.

    :param rips_persistence: Object for computing the persistence diagram.
    :type rips_persistence: RipsPersistence

    :param X: Either point array or distance matrix.
    :type X: List of List of float or List of np.ndarray

    :param max_edge_length: Maximum edge legth for including in complexes.
    :type max_edge_length: float

    :param max_dim: Compute diagrams up to this dimension, included.
    :type max_dim: int

    :param save_path: The file path to save the resulting graph visualization.
    :type save_path: str
    """
    diagram = rips_persistence.transform(X)[0]
    with open(save_path + '.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dimension', 'birth', 'death'])
        for dim in range(max_dim + 1):
            diag = diagram[dim]
            for birth, death in diag:
                writer.writerow([dim, birth, death if np.isfinite(death) else -1])
    fig, ax = plt.subplots(figsize=(6, 4))
    for dim in range(max_dim + 1):
        subdiagram = diagram[dim]
        for i, (b, d) in enumerate(subdiagram):
            ax.hlines(y=i + dim * 20, xmin=b, xmax=d if np.isfinite(d) else max_edge_length, color='C' + str(dim), lw=0.5, label=f"H{int(dim)}" if i == 0 else "")
    ax.set_title("Persistence Barcode")
    ax.set_xlabel("Filtration Value")
    ax.set_yticks([])
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path + '.png', dpi=300)
    plt.close(fig)


def main_subthread(
        args: Namespace,
        name: str,
        graph_dataset: GraphDataset,
        k: int,
        rips_persistence: RipsPersistence,
        max_edge_length: float,
        max_dim: int,
        pbar: tqdm,
        ):
    _X, y, xx, yy = read_node_matrix(
        os.path.join(args.output_dir, 'graphs', name + '.nodes.csv'), return_coordinates=True, return_class=True,
        remove_prior=False, remove_morph=False
        )
    graph = graph_dataset[k]
    os.makedirs(os.path.join(args.output_dir, name), exist_ok=True)
    if args.use_metric:
        X = [np.array([[x, y] for x, y in zip(xx, yy)])]
    else:
        X = [compute_graph_distance_matrix(graph)]
    draw_barcode(
        rips_persistence,
        X,
        max_edge_length,
        max_dim,
        os.path.join(args.output_dir, name, f'{"metric" if args.use_metric else "graph"}.barcode')
    )
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
    pbar = tqdm(total=len(names))
    plt.switch_backend('Agg')
    max_edge_length = args.max_distance if args.use_metric else 10
    max_dimension = 2
    rips_persistence = RipsPersistence(
        input_type='point cloud' if args.use_metric else 'full distance matrix',
        threshold=max_edge_length,
        homology_dimensions=list(range(max_dimension + 1))
    )
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for k, name in enumerate(names):
                executor.submit(main_subthread, args, name, graph_dataset, k, rips_persistence, max_edge_length, max_dimension, pbar)
    else:
        for k, name in enumerate(names):
            main_subthread(args, name, graph_dataset, k, rips_persistence, max_edge_length, max_dimension, pbar)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
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
