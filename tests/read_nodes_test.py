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

from tumourkit.classification.read_nodes import create_node_splits
from tumourkit.utils.preprocessing import parse_path 

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = parse_path(TEST_DIR) + 'graphs/'


@pytest.mark.parametrize("val_size,test_size", [
    (0.2,0.2), (0.2,0.1), (0.8,0.1), (0.1,0.8), (0.45,0.45)
])
def test_graph_centroids(val_size, test_size):
    X_tr, X_val, X_ts, y_tr, y_val, y_ts = \
        create_node_splits(GRAPHS_DIR, val_size, test_size, mode='total')
    X_tr2, X_val2, X_ts2, y_tr2, y_val2, y_ts2 = \
        create_node_splits(GRAPHS_DIR, val_size, test_size, mode='by_img')
    assert X_tr.shape[0] == y_tr.shape[0] and X_val.shape[0] == y_val.shape[0] \
        and X_ts.shape[0] == y_ts.shape[0] and X_tr2.shape[0] == y_tr2.shape[0] \
        and X_val2.shape[0] == y_val2.shape[0] and X_ts2.shape[0] == y_ts2.shape[0] \
        and X_tr.shape[0] + X_val.shape[0] + X_ts.shape[0] == X_tr2.shape[0] + X_val2.shape[0] + X_ts2.shape[0] \
        and len(y_tr.shape) == 1 and len(y_val.shape) == 1 and len(y_ts.shape) == 1 \
        and len(y_tr2.shape) == 1 and len(y_val2.shape) == 1 and len(y_ts2.shape) == 1 \
        and y_tr.max() == 1 and y_val.max() == 1 and y_ts.max() == 1
