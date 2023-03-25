"""
Computes centroids csv from png <-> csv labels.

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
from tqdm import tqdm
from ..utils.preprocessing import read_labels, parse_path, create_dir, get_names, save_centroids, get_centroid_by_id
import numpy as np
import pandas as pd
from typing import List, Tuple


def extract_centroids(img: np.ndarray, csv: pd.DataFrame) -> List[Tuple[int,int,int]]:
    """
    Extracts the centroids of cells from a labeled image. The third coordinate is the class.

    :param img: A 2D NumPy array representing the labeled image. Each unique non-zero value represents a different cell.
    :type img: np.ndarray
    :param csv: A pandas DataFrame representing the labels. Just two colums, one for id and another for the class.
    :type csv: pd.DataFrame
    :return: A list of tuples containing the x and y coordinates of the centroid, and the cell class.
    :rtype: List[Tuple[int,int,int]]
    """
    centroids = []
    for _, row in csv.iterrows():
        x, y = get_centroid_by_id(img, row.id)
        if x == -1:
            continue
        centroids.append((x, y, row.label))
    return centroids


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv files.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    return parser


def main_with_args(args):
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    output_path = parse_path(args.output_path)
    create_dir(output_path)

    names = get_names(png_dir, '.GT_cells.png')
    for name in tqdm(names):
        img, csv = read_labels(name, png_dir, csv_dir)
        centroids = extract_centroids(img, csv)
        save_centroids(centroids, output_path, name)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
