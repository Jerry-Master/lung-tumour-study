"""

Converts our png <-> csv format into QuPath geojson format.

"""
import argparse
import pandas as pd
import cv2
import numpy as np
import geojson
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--png_dir', type=str, required=True,
                    help='Path to png files.')
parser.add_argument('--csv_dir', type=str, required=True,
                    help='Path to csv files.')
parser.add_argument('--gson_dir', type=str, required=True,
                    help='Path to save files.')


def save_geojson(gson, name, path):
    """
    Save geojson to file path + name.
    """
    with open(path + name + '.geojson', 'w') as f:
        geojson.dump(gson, f)

def create_mask(png, csv, label):
    """
    Returns the image with only the pixels of the class given in label.
    The pixel values are truncated to uint8.
    """
    mask = png.copy()
    for i, (idx, cell_label) in csv.iterrows():
        if cell_label != label:
            mask[mask==idx] = 0
    return np.array(mask, dtype=np.uint8)

def format_contour(contour):
    """
    Auxiliary function to pass from the cv2.findContours format to
    an array of shape (N,2). Additionally, the first point is added
    to the end to close the contour.
    """
    new_contour = np.reshape(contour, (-1,2)).tolist()
    new_contour.append(new_contour[0])
    return new_contour

def pngcsv2features(png, csv, label):
    """
    Computes geojson features of contours of a given class.
    """
    mask = create_mask(png, csv, label)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c), label) for c in contours])
    return create_geojson(contours)

def pngcsv2geojson(png, csv):
    """
    Computes geojson as list of features representing contours.
    """
    features_tumour = pngcsv2features(png, csv, 2)
    features_nontumour = pngcsv2features(png, csv, 1)
    features_tumour.extend(features_nontumour)
    return features_tumour


if __name__ == '__main__':
    args = parser.parse_args()
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)
    OUTPUT_PATH = parse_path(args.gson_dir)
    create_dir(OUTPUT_PATH)

    names = get_names(PNG_DIR, '.GT_cells.png')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        png, csv = read_labels(name, PNG_DIR, CSV_DIR)
        gson = pngcsv2geojson(png, csv)
        save_geojson(gson, name, OUTPUT_PATH)
    print()
