"""

Creates images with pixel value 255 at the centroids coordinates.

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
parser.add_argument('--centroids_path', type=str, required=True,
                    help='Path to centroid files.')
parser.add_argument('--output_path', type=str, required=True,
                    help='Path to save png files with points.')

def centroids2png(centroids: list[tuple[int,int,int]]) -> np.array:
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