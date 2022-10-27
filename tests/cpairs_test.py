import pytest
import sys
import os
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path
from utils.postprocessing import get_N_closest_pairs_dists

RSWOOSH_DIR = parse_path(TEST_DIR) + 'rswoosh/'
THRESHOLD = 0.0001

@pytest.mark.parametrize("A,B,result", [
    (
        np.hstack((np.linspace(0,30,100).reshape(-1,1), np.ones((100,1)))),
        np.hstack((np.linspace(0,30,100).reshape(-1,1), np.ones((100,1))*2)), 
        np.ones((10,))
    ),
    (
        np.hstack((np.linspace(0,300,101).reshape(-1,1), np.ones((101,1))))+np.array([[1,0]]),
        np.hstack((np.linspace(0,300,101).reshape(-1,1), np.ones((101,1))*2)), 
        np.ones((20,)) * np.sqrt(2)
    ),
    (
        np.hstack((np.linspace(0,30,11).reshape(-1,1)**2, np.ones((11,1)))),
        np.hstack((np.linspace(0,30,11).reshape(-1,1), np.ones((11,1))*2)), 
        np.array([1, 1, np.sqrt(3**2+1), np.sqrt(3**2+1), np.sqrt(3**2+1), np.sqrt(6**2+1), np.sqrt(6**2+1)])
    )
])
def test_conversion(A,B, result):
    pred = get_N_closest_pairs_dists(A, B, len(result))
    pred = sorted(pred)
    if len(pred) != len(result):
        assert(False)
    for i in range(len(pred)):
        if abs(pred[i] - result[i]) > THRESHOLD:
            assert(False)
    assert(True)
