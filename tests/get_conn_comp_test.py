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

from tumourkit.utils.preprocessing import parse_path
from tumourkit.utils.postprocessing import get_greatest_connected_component

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
RSWOOSH_DIR = parse_path(TEST_DIR) + 'rswoosh/'
THRESHOLD = 0.0001

@pytest.mark.parametrize("ids,max_id,out_l,out_r", [
    ([1,2,3,4,6,7,8], 8, 1, 4),
    ([1,4,3,7,8,6,2], 8, 1, 4),
    ([0,2,3,4,5,6,9,11], 11, 2, 6),
    ([3,2,0,4,11,5,6,9], 11, 2, 6),
    ([9,10,0,1,2,6,8], 10, 8, 2),
    ([0,1,2,3,4,5,6], 10, 0, 6),
    ([0,1,2,3,4,5,6], 6, 0, 6),
    ([0,1,2,3,4,5,6,8,9,10], 11, 0, 6),
    ([0,1,2,3,4,5,6,8,9,10], 10, 8, 6)
])
def test_conversion(ids, max_id, out_l, out_r):
    l, r = get_greatest_connected_component(ids, max_id)
    assert(l == out_l and r == out_r)
