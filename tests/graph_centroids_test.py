import pytest
import sys
import os
import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from preprocessing.pngcsv2graph import create_graph
from utils.preprocessing import get_names, parse_path, read_labels, read_centroids
from utils.nearest import generate_tree, find_nearest_dist_idx

PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 10


@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.GT_cells.png')[-5:])
def test_graph_centroids(name):
    png, csv = read_labels(name, PNGCSV_DIR, PNGCSV_DIR)
    false_img = np.ones((*png.shape, 3)) * png.reshape((*png.shape, 1))
    false_img = false_img.astype(np.uint8)
    graph = create_graph(false_img, png, csv)
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)

    centroids = read_centroids(name, PNGCSV_DIR)
    centr_tree = generate_tree(centroids)
    for _, row in graph.iterrows():
        dist, idx = find_nearest_dist_idx(row[['X','Y', 'class']], centr_tree)
        closest_cls = centroids[idx][2]
        assert(dist < THRESHOLD and closest_cls == row['class'])
