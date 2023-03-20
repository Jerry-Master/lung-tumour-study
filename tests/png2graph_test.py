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
import pandas as pd
import numpy as np

from tumourkit.preprocessing.png2graph import png2graph
from tumourkit.utils.preprocessing import get_names, parse_path, read_png, read_graph
from tumourkit.utils.nearest import generate_tree, find_nearest_dist_idx

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
GRAPHS_DIR = parse_path(TEST_DIR) + 'graphs/'
THRESHOLD = 5


def check_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a_tree = generate_tree(a[['X', 'Y']].to_numpy())
    mean_dist = 0
    for point in b[['X', 'Y']].to_numpy():
        closest_dist, _ = find_nearest_dist_idx(point, a_tree)
        mean_dist += closest_dist
    mean_dist /= len(b)
    return mean_dist < THRESHOLD


@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.GT_cells.png'))
def test_png2graph(name):
    png_file = read_png(name, PNGCSV_DIR)
    false_orig = np.ones((png_file.shape[0], png_file.shape[1], 3), dtype=np.uint8)
    graph = png2graph(false_orig, png_file)
    ref_graph = read_graph(name, GRAPHS_DIR)
    if not check_equal(graph, ref_graph):
        assert False, name
    assert(True)
