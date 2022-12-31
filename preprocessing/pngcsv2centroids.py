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
import pandas as pd
import numpy as np
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--png-dir', type=str, required=True,
                    help='Path to png files.')
parser.add_argument('--csv-dir', type=str, required=True,
                    help='Path to csv files.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to save files.')


def get_centroid_by_id(img: np.ndarray, idx: int) -> tuple[int, int]:
    """
    img contains a different id value per component at each pixel
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()

def extract_centroids(img: np.ndarray, csv: pd.DataFrame) -> list[tuple[int,int,int]]:
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

if __name__ == '__main__':
    args = parser.parse_args()
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)
    OUTPUT_PATH = parse_path(args.output_path)
    create_dir(OUTPUT_PATH)

    names = get_names(PNG_DIR, '.GT_cells.png')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        img, csv = read_labels(name, PNG_DIR, CSV_DIR)
        centroids = extract_centroids(img, csv)
        df = pd.DataFrame(centroids, columns=['X','Y','class'])
        df.to_csv(OUTPUT_PATH + name + '.centroids.csv', index=False)
    print()
