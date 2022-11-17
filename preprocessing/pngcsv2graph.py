"""

Computes a graph representation from the images and pngcsv labels.

"""
from typing import Dict, Tuple
import argparse
import pandas as pd
import numpy as np
import sys
import os
import cv2

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--png_dir', type=str, required=True,
                    help='Path to png files.')
parser.add_argument('--csv_dir', type=str, required=True,
                    help='Path to csv files.')
parser.add_argument('--orig_dir', type=str, required=True,
                    help='Path to original images.')
parser.add_argument('--output_path', type=str, required=True,
                    help='Path to save files.')

def read_image(name: str, path: str) -> np.array:
    """
    Given name image (without extension) and folder path,
    returns array with pixel values (RGB).
    """
    aux = cv2.imread(os.path.join(path, name+'.png'))
    return cv2.cvtColor(aux, cv2.COLOR_BGR2RGB)

def get_mask(png: np.array, idx: int) -> np.array:
    """
    Given segmentation mask with indices as pixel values, 
    returns the mask corresponding to the given index.
    """
    png_aux = png.copy()
    png_aux[png_aux!=idx] = 0
    png_aux[png_aux!=0] = 1
    return png_aux

def apply_mask(img: np.array, mask: np.array) -> Tuple[np.array, int, int]:
    """
    Given RGB image and binary mask, 
    returns the masked image.
    For efficiency only the bounded box of the mask is returned,
    together with the center of the box.
    Coordinates are indices of array.
    """
    img_aux = img.copy()
    img_aux = img_aux * mask.reshape(*mask.shape, 1)
    x,y,w,h = cv2.boundingRect((mask * 255).astype(np.uint8))
    return img_aux[y:y+h, x:x+w].copy(), int(y+h/2), int(x+w/2)

def compute_perimeter(c: Contour) -> float:
    """
    Given contour returns its perimeter.
    Prerequisites: First and last point must be the same.
    """
    diff = np.diff(c, axis=0)
    dists = np.hypot(diff[:,0], diff[:,1])
    return dists.sum()

def extract_features(msk_img: np.array) -> Dict[str, np.array]:
    """
    Given RGB bounding box of a cell,
    returns a dictionary with different extracted features.
    """
    if len(msk_img) == 0 or msk_img.max() == 0:
        return {}
    gray_msk = cv2.cvtColor(msk_img, cv2.COLOR_RGB2GRAY)
    bin_msk = (gray_msk > 0) * 1
    feats = {}
    feats['area'] = bin_msk.sum()

    contours, _ = cv2.findContours(
        (bin_msk * 255).astype(np.uint8), 
        mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    contour = format_contour(contours)
    feats['perimeter'] = compute_perimeter(contour)
    return feats

def add_node(graph: Dict[str, float], feats: Dict[str, np.array]) -> None:
    """
    [Inplace operation]
    Given a dictionary with vectorial features,
    converts them into one column per dimension
    and add them into the global dictionary.
    """
    for k, v in feats.items():
        if hasattr(v, '__len__'):
            for i, coord in enumerate(v):
                if k + str(i) not in graph:
                    graph[k + str(i)] = []
                graph[k + str(i)].append(coord)
        else:
            if k not in graph:
                graph[k] = []
            graph[k].append(v)

def create_graph(img: np.array, png: np.array, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Given original image and pngcsv labels, 
    returns nodes with extracted attributes in a DataFrame.
    Current attributes are:
        - X, Y of centroid
        - Area
        - Regularity (not yet)
        - Histogram (not yet)
    """
    graph = {}
    for idx, cls in csv.itertuples(index=False, name=None):
        mask = get_mask(png, idx)
        msk_img, X, Y  = apply_mask(img, mask)
        feats = extract_features(msk_img)
        if len(feats) > 0:
            feats['class'] = cls
            feats['id'] = idx
            feats['X'] = X
            feats['Y'] = Y
        add_node(graph, feats)
    return pd.DataFrame(graph)

def save_graph(graph: pd.DataFrame, path: str) -> None:
    """
    Saves dataframe in given path.
    """
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)
    graph.to_csv(path)

if __name__ == '__main__':
    args = parser.parse_args()
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)
    ORIG_DIR = parse_path(args.orig_dir)
    OUTPUT_PATH = parse_path(args.output_path)
    create_dir(OUTPUT_PATH)

    names = get_names(PNG_DIR, '.GT_cells.png')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        png, csv = read_labels(name, PNG_DIR, CSV_DIR)
        img = read_image(name, ORIG_DIR)
        graph = create_graph(img, png, csv)
        save_graph(graph, os.path.join(OUTPUT_PATH, name+'.csv'))
    print()
