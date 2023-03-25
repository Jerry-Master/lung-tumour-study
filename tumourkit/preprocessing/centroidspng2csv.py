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
from typing import List, Tuple
import argparse
from argparse import Namespace
from ..utils.preprocessing import get_names, parse_path, create_dir, read_png, save_csv, read_centroids, get_centroid_by_id
import pandas as pd
import numpy as np
from ..utils.nearest import generate_tree, find_nearest
from tqdm import tqdm

def extract_centroids(img: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Extracts the centroids of cells from a labeled image. The third coordinates is the index.

    :param img: A 2D NumPy array representing the labeled image. Each unique non-zero value represents a different cell.
    :type img: np.ndarray
    :return: A list of tuples containing the x and y coordinates of the centroid, and the cell index.
    :rtype: List[Tuple[int,int,int]]
    """
    centroids = []
    for idx in np.unique(img):
        if idx != 0:
            x, y = get_centroid_by_id(img, idx)
            if x == -1:
                continue
            centroids.append((x, y, idx))
    return centroids


def centroidspng2csv(centroids_file: np.ndarray, png_file: np.ndarray) -> pd.DataFrame:
    """
    Converts a PNG file with cell labels and a CSV file with cell centroids into a CSV file
    associating each cell label with the closest centroid. 

    :param centroids_file: A NumPy array with three columns, representing the X and Y coordinates
                           and class of each centroid.
    :type centroids_file: np.ndarray
    :param png_file: A NumPy array with the same dimensions as the corresponding image file,
                     where each pixel is labeled with an integer representing the cell it belongs to.
    :type png_file: np.ndarray
    :return: A Pandas DataFrame with two columns: the first represents the cell ID from the PNG file,
             and the second represents the class of the closest centroid from the CSV file.
    :rtype: pd.DataFrame

    This function first generates a k-d tree from the centroids in the `centroids_file` array.
    For each centroid in the `png_file` array, the function finds the closest centroid in the
    `centroids_file` array using the k-d tree. It then associates the cell label from the PNG file
    with the class of the closest centroid in a list.
    Finally, the function returns a Pandas DataFrame containing the list of cell IDs and centroid
    classes.
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