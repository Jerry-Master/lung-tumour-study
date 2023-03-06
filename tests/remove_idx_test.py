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

from tumourkit.utils.postprocessing import remove_idx



@pytest.mark.parametrize("a,a_idx,out", [
    ([0,0,[[0,0],[1,1],[2,2],[3,3]]], [2], (0,0,[[3,3],[0,0],[1,1]])),
    (
        [0,0,[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]], 
        [9,10,0,1,2,6,8], 
        (0,0,[[3,3],[4,4],[5,5],[6,6],[7,7]]),
    ),
    (
        [0,0,[[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],[11,11]]], 
        [3,2,0,4,11,5,6,9],
        (0,0,[[7,7],[8,8],[9,9],[10,10],[11,11],[0,0],[1,1]]), 
    )
])
def test_conversion(a,a_idx,out):
    pred = remove_idx(a, a_idx)
    assert(pred==out)
