import pytest
import sys
import os
import numpy as np
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from preprocessing.pngcsv2graph import create_graph
from utils.preprocessing import get_names, parse_path, read_labels

PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 0.1


@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.GT_cells.png')[:5])
def test_indices(name):
    png, csv = read_labels(name, PNGCSV_DIR, PNGCSV_DIR)
    false_img = np.ones((*png.shape, 3)) * png.reshape((*png.shape, 1))
    false_img = false_img.astype(np.uint8)
    graph = create_graph(false_img, png, csv)
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)
    csv.columns = ['id', 'class']
    csv.set_index('id', inplace=True)
    csv.sort_index(inplace=True)
    join = pd.merge(left=csv, right=graph, 
         left_index=True, right_index=True, how='outer').dropna()
    assert((join.class_x == join.class_y).all())
