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

from tumourkit.preprocessing.graph2centroids import graph2centroids
from tumourkit.utils.preprocessing import get_names, parse_path, read_graph, read_centroids
from tumourkit.utils.nearest import generate_tree, find_nearest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = parse_path(TEST_DIR) + 'graphs/'
CENTROIDS_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 0.1


def check_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a_tree = generate_tree(a)
    for point in b:
        closest_idx = find_nearest(point, a_tree)
        closest = a[closest_idx]
        if closest[2] != point[2]:
            return False
    return True


@pytest.mark.parametrize("name", get_names(GRAPH_DIR, '.nodes.csv'))
def test_graph2centroids(name):
    graph_file = read_graph(name, GRAPH_DIR)
    centroids_file = graph2centroids(graph_file)
    ref_centroids_file = read_centroids(name, CENTROIDS_DIR)
    if not check_equal(centroids_file, ref_centroids_file):
        assert False, name
    assert(True)
