import pytest
import sys
import os
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.preprocessing import get_names, parse_path
from evaluate import get_confusion_matrix

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
def test_metric(name):
    A_centroids = pd.read_csv(CENTROIDS_DIR + name + '.A.csv').to_numpy()
    B_centroids = pd.read_csv(CENTROIDS_DIR + name + '.B.csv').to_numpy()
    result = pd.read_csv(CENTROIDS_DIR + name + '.result.csv', header=None).to_numpy()
    confusion_matrix = get_confusion_matrix(A_centroids, B_centroids)
    assert((result == confusion_matrix).all())
