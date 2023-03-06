"""
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
import pytest
import os
import numpy as np

from tumourkit.preprocessing.geojson2pngcsv import geojson2pngcsv
from tumourkit.preprocessing.pngcsv2geojson import pngcsv2geojson
from tumourkit.utils.preprocessing import get_names, parse_path, read_labels

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
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
