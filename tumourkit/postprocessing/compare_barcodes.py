"""
Computes q-Wassertein distance and bottleneck distance of persistence diagrams.
Input format: CSV (births and deaths)
Output format: TXT

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
import csv
import numpy as np
from gudhi.wasserstein import wasserstein_distance
from gudhi import bottleneck_distance
import os


def read_diagram_from_csv(path: str, homology_dim: int) -> np.ndarray:
    """
    Reads a persistence diagram from a CSV file and filters by homology dimension.

    :param path: Path to the CSV file.
    :type path: str

    :param homology_dim: Homology dimension to extract (e.g., 0 for H0).
    :type homology_dim: int

    :return: Numpy array of (birth, death) pairs for the specified homology dimension.
    :rtype: np.ndarray
    """
    diagram = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim = int(row['dimension'])
            if dim == homology_dim:
                birth = float(row['birth'])
                death = float(row['death'])
                if death == -1:
                    continue  # Ignore infinity values
                diagram.append((birth, death))
    return np.array(diagram)


def compute_wasserstein_distance(dgm1: np.ndarray, dgm2: np.ndarray, q: int) -> float:
    """
    Computes the q-Wasserstein distance between two persistence diagrams.

    :param dgm1: Persistence diagram 1 as numpy array of (birth, death) pairs.
    :type dgm1: np.ndarray

    :param dgm2: Persistence diagram 2 as numpy array of (birth, death) pairs.
    :type dgm2: np.ndarray

    :param q: Order of the Wasserstein distance. None if bottleneck.
    :type q: int

    :return: Computed Wasserstein distance.
    :rtype: float
    """
    if q is None:
        return bottleneck_distance(dgm1, dgm2)
    return wasserstein_distance(dgm1, dgm2, order=q, internal_p=float(q))


def main_with_args(args: Namespace) -> None:
    """
    Main processing function to compute Wasserstein distance from two barcode CSV files.

    :param args: Parsed command-line arguments.
    :type args: Namespace
    """
    if not os.path.isfile(args.barcode1) or not os.path.isfile(args.barcode2):
        print("One or both input barcode files do not exist.")
        return

    dgm1 = read_diagram_from_csv(args.barcode1, args.homology_dim)
    dgm2 = read_diagram_from_csv(args.barcode2, args.homology_dim)

    dist = compute_wasserstein_distance(dgm1, dgm2, args.q)

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)

    dist_name = f'Wasserstein-{args.q}' if args.q is not None else "Bottleneck"
    if args.output_path is not None:
        with open(args.output_path, 'w') as f:
            f.write(f"{dist_name} distance (H{args.homology_dim}): {dist:.6f}\n")
        print(f"Result saved to {args.output_path}")
    print(f"{dist_name} distance (H{args.homology_dim}): {dist:.6f}")


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--barcode1', type=str, required=True, help='Path to first barcode CSV file.')
    parser.add_argument('--barcode2', type=str, required=True, help='Path to second barcode CSV file.')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save the Wasserstein distance result.')
    parser.add_argument('-q', type=int, default=None, help='If not provided, then bottleneck is computed.')
    parser.add_argument('--homology-dim', type=int, default=0)
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
