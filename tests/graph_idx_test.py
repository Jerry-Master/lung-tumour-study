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
import pandas as pd

from tumourkit.preprocessing.pngcsv2graph import pngcsv2graph
from tumourkit.utils.preprocessing import get_names, parse_path, read_labels

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 0.1


@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.GT_cells.png')[:5])
def test_indices(name):
    png, csv = read_labels(name, PNGCSV_DIR, PNGCSV_DIR)
    false_img = np.ones((*png.shape, 3)) * png.reshape((*png.shape, 1))
    false_img = false_img.astype(np.uint8)
    graph = pngcsv2graph(false_img, png, csv)
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)
    csv.columns = ['id', 'class']
    csv.set_index('id', inplace=True)
    csv.sort_index(inplace=True)
    join = pd.merge(left=csv, right=graph, 
         left_index=True, right_index=True, how='outer').dropna()
    assert((join.class_x == join.class_y).all())
