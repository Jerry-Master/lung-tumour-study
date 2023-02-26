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
from typing import List, Tuple
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from ..utils.preprocessing import read_labels, parse_path, create_dir, get_names


def get_centroid_by_id(img: np.ndarray, idx: int) -> Tuple[int, int]:
    """
    img contains a different id value per component at each pixel
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()


def extract_centroids(img: np.ndarray, csv: pd.DataFrame) -> List[Tuple[int,int,int]]:
    """
    Output format: list of (x,y,class) tuples
    """
    centroids = []
    for _, row in csv.iterrows():
        x, y = get_centroid_by_id(img, row.id)
        if x == -1:
            continue
        centroids.append((x,y,row.label))
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


def main():
    parser = _create_parser()
    args = parser.parse_args()
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    output_path = parse_path(args.output_path)
    create_dir(output_path)

    names = get_names(png_dir, '.GT_cells.png')
    for name in tqdm(names):
        img, csv = read_labels(name, png_dir, csv_dir)
        centroids = extract_centroids(img, csv)
        df = pd.DataFrame(centroids, columns=['X','Y','class'])
        df.to_csv(output_path + name + '.centroids.csv', index=False)
