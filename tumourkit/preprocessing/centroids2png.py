"""
Creates images with pixel value 255 at the centroids coordinates.

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
import numpy as np
import cv2
from ..utils.preprocessing import parse_path, create_dir, get_names, read_centroids
import argparse
from tqdm import tqdm


def centroids2png(centroids: List[Tuple[int,int,int]]) -> np.ndarray:
    """
    Generates a blank image with pixel value 255 at the coordinates of each centroid.

    :param centroids: A list of centroids represented as (X, Y, class) tuples.
    :type centroids: List[Tuple[int,int,int]]
    :return: A NumPy array representing the generated image.
    :rtype: np.ndarray
    """
    png = np.zeros((1024,1024))
    for x,y,cls in centroids:
        png[x,y] = 255
    return png


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--centroids-path', type=str, required=True,
                        help='Path to centroid files.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save png files with points.')
    return parser


def main_with_args(args):
    centroids_path = parse_path(args.centroids_path)
    output_path = parse_path(args.output_path)
    create_dir(output_path)
    names = get_names(centroids_path, '.centroids.csv')
    for name in tqdm(names):
        centroids = read_centroids(name, centroids_path)
        png = centroids2png(centroids)
        cv2.imwrite(output_path + name + '.points.png', png)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)