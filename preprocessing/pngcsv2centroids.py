"""

Computes centroids csv from png <-> csv labels.

"""
import argparse
import pandas as pd
import os
import cv2
import numpy as np
from ..utils/preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--png_dir', type=str, required=True,
                    help='Path to png files.')
parser.add_argument('--csv_dir', type=str, required=True,
                    help='Path to csv files.')
parser.add_argument('--output_path', type=str, required=True,
                    help='Path to save files.')

def read_labels(name, png_path, csv_path):
    img = cv2.imread(png_path + name + '.GT_cells.png', -1)
    csv = pd.read_csv(csv_path + name + '.class.csv')
    csv.columns = ['id', 'label']
    return img, csv

def get_centroid_by_id(img, idx):
    """
    img contains a different id value per component at each pixel
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()

def extract_centroids(img, csv):
    """
    Output format: list of (x,y,class) tuples
    """
    centroids = []
    for i, row in csv.iterrows():
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
