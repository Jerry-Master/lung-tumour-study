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
import numpy as np
import cv2
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--centroids-path', type=str, required=True,
                    help='Path to centroid files.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to save png files with points.')

def centroids2png(centroids: list[tuple[int,int,int]]) -> np.ndarray:
    """
    Generates blank image with pixel value 255 at centroids coordinates.
    """
    png = np.zeros((1024,1024))
    for x,y,cls in centroids:
        png[x,y] = 255
    return png

if __name__ == '__main__':
    args = parser.parse_args()
    CENTROIDS_PATH = parse_path(args.centroids_path)
    OUTPUT_PATH = parse_path(args.output_path)
    create_dir(OUTPUT_PATH)
    names = get_names(CENTROIDS_PATH, '.centroids.csv')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        centroids = read_centroids(name, CENTROIDS_PATH)
        png = centroids2png(centroids)
        cv2.imwrite(OUTPUT_PATH + name + '.points.png', png)