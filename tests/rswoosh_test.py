import pytest
import sys
import os
import json

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from postprocessing.rswoosh import rswoosh
from utils.preprocessing import get_names, parse_path
from utils.postprocessing import create_comparator, merge_cells

RSWOOSH_DIR = parse_path(TEST_DIR) + 'rswoosh/'
THRESHOLD = 1
NUM_FRONTIER = 30

@pytest.mark.parametrize("name", get_names(RSWOOSH_DIR, '.input.json'))
def test_conversion(name):
    with open(RSWOOSH_DIR + name + '.input.json', 'r') as f:
        inp = json.load(f)
    with open(RSWOOSH_DIR + name + '.output.json', 'r') as f:
        expected_out = json.load(f)
    predicted_out = rswoosh(inp, create_comparator(THRESHOLD, NUM_FRONTIER), merge_cells)
    if not len(predicted_out) == len(expected_out):
        print(name)
        assert(False)
    assert(True)
