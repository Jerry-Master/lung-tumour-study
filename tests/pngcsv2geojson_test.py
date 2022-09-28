import pytest
import sys
import os
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from preprocessing.geojson2pngcsv import geojson2pngcsv
from preprocessing.pngcsv2geojson import pngcsv2geojson, read_labels
from utils.preprocessing import get_names, parse_path

PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 0.1

def parse_labels(png, csv):
    negatives = set(csv[csv.label==1].id)
    positives = set(csv[csv.label==2].id)
    assert(not (csv.label==0).any())
    def parser(x):
        if x in negatives:
            return 1
        elif x in positives:
            return 2
        else:
            return 0
    vec_parser = np.vectorize(parser)
    return vec_parser(png)

def compare(img1, img2):
    return np.sum(np.abs(img1-img2)) / (np.multiply.reduce(img1.shape))

def check_equal(png1, csv1, png2, csv2):
    img_label1 = parse_labels(png1, csv1)
    img_label2 = parse_labels(png2, csv2)
    assert(len(np.unique(img_label1)) == 3)
    assert(len(np.unique(img_label2)) == 3)
    diff = compare(img_label1, img_label2)
    print(diff)
    return diff < THRESHOLD

@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.GT_cells.png'))
def test_conversion(name):
    png_orig, csv_orig = read_labels(name, PNGCSV_DIR, PNGCSV_DIR)
    gson = pngcsv2geojson(png_orig, csv_orig)
    png_new, csv_new = geojson2pngcsv(gson)
    if not check_equal(png_orig,csv_orig, png_new,csv_new):
        print(name)
        assert(False)
    assert(True)
