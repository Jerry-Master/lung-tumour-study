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
import numpy as np

from tumourkit.utils.preprocessing import parse_path
from tumourkit.utils.postprocessing import get_N_closest_pairs_dists

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
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
