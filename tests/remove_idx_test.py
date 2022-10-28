import pytest
import sys
import os
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path
from utils.postprocessing import remove_idx

RSWOOSH_DIR = parse_path(TEST_DIR) + 'rswoosh/'
THRESHOLD = 0.0001

@pytest.mark.parametrize("a,a_idx,out", [
    ([0,0,[[0,0],[1,1],[2,2],[3,3]]], [2], [0,0,[[3,3],[0,0],[1,1]]]),
    (
        [0,0,[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]], 
        [9,10,0,1,2,6,8], 
        [0,0,[[3,3],[4,4],[5,5],[6,6],[7,7]]],
    ),
    (
        [0,0,[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]]], 
        [3,2,0,4,11,5,6,9],
        [0,0,[[7,7],[8,8],[9,9],[10,10],[11,11],[0,0],[1,1]]], 
    )
])
def test_conversion(a,a_idx,out):
    pred = remove_idx(a, a_idx)
    assert(pred==out)
