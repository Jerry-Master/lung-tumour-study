import pytest
import sys
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from classification.read_nodes import create_node_splits
from utils.preprocessing import parse_path 

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
