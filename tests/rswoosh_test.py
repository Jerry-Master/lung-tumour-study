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
import json

from tumourkit.postprocessing.rswoosh import rswoosh
from tumourkit.utils.preprocessing import get_names, parse_path
from tumourkit.utils.postprocessing import create_comparator, merge_cells

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
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
