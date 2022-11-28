import pytest
import sys
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.preprocessing import get_names, parse_path
from evaluate import get_pairs

CENTROIDS_DIR = parse_path(TEST_DIR) + 'centroids/'


"""
The format of the names should be [name].A.csv and [name].B.csv,
being A and B two files to compare the metric. And [name].result.csv
should be the expected result. 
Labels should be only 1 and 2.
CSV files must have headers, except for [name].result.csv which should
only be the confusion matrix.
"""
@pytest.mark.parametrize("name", get_names(CENTROIDS_DIR, '.result.csv'))
def test_metric_pairs(name):
    A_centroids = pd.read_csv(CENTROIDS_DIR + name + '.A.csv').to_numpy()
    B_centroids = pd.read_csv(CENTROIDS_DIR + name + '.B.csv').to_numpy()
    result = pd.read_csv(CENTROIDS_DIR + name + '.result.csv', header=None).to_numpy()
    true, pred = get_pairs(A_centroids, B_centroids)
    conf_matrix = confusion_matrix(true, pred, labels=[0,1])
    assert((result == conf_matrix).all())
