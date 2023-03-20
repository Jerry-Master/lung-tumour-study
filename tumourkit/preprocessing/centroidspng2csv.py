"""
Merges .centroids.csv and .GT_cells.png into .class.csv files.

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
from ..utils.preprocessing import get_names, parse_path, create_dir, read_png, save_csv, read_centroids
import pandas as pd
import numpy as np
from typing import List, Tuple
from ..utils.nearest import generate_tree, find_nearest
from tqdm import tqdm


def get_centroid_by_id(img: np.ndarray, idx: int) -> Tuple[int, int]:
    """
    img contains a different id value per component at each pixel
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()


def extract_centroids(img: np.ndarray) -> List[Tuple[int,int]]:
    """
    Output format: list of (x,y,idx) tuples, where idx is the index associated
        with the cell.
    """
    centroids = []
    for idx in np.unique(img):
        if idx != 0:
            x, y = get_centroid_by_id(img, idx)
            if x == -1:
                continue
            centroids.append((x,y, idx))
    return centroids


def centroidspng2csv(centroids_file: np.ndarray, png_file: np.ndarray) -> pd.DataFrame:
    """
    For each cell in png_file it finds the nearest centroid and associates its class to it.
    """
    centroids_tree = generate_tree(centroids_file)
    png_centroids = extract_centroids(png_file)
    csv_list = []
    for point in png_centroids:
        closest_id = find_nearest(point, centroids_tree)
        closest = centroids_file[closest_id]
        csv_list.append([point[2], closest[2]])
    return pd.DataFrame(csv_list)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--centroids-dir', type=str, required=True, help='Path to folder containing .centroids.csv.')
    parser.add_argument('--png-dir', type=str, required=True, help='Path to folder containing .GT_cells.png.')
    parser.add_argument('--csv-dir', type=str, required=True, help='Path to folder where to save .class.csv.')
    return parser


def main_with_args(args: Namespace) -> None:
    centroids_dir = parse_path(args.centroids_dir)
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    create_dir(csv_dir)
    names_centroids = sorted(get_names(centroids_dir, '.centroids.csv'))
    names_png = sorted(get_names(png_dir, '.GT_cells.png'))
    names = list(set(names_png).intersection(set(names_centroids)))
    for name in tqdm(names):
        centroids_file = read_centroids(name, centroids_dir)
        png_file = read_png(name, png_dir)
        csv_file = centroidspng2csv(centroids_file, png_file)
        save_csv(csv_file, csv_dir, name)
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)