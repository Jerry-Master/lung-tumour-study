"""
Copyright (C) 2025  Jose Pérez Cano

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
import numpy as np
import pytest
import tumourkit.utils.tda
from tumourkit.utils.tda import compute_matrix_persistence


def test_compute_matrix_persistence_rips():
    matrix = np.array([[1, 2], [3, 4]])
    if not tumourkit.utils.tda.HAS_DIONYSUS:
        with pytest.raises(RuntimeError, match="Rips persistence requires Dionysus"):
            compute_matrix_persistence(matrix, use_cubical=False)
    else:
        persistence = compute_matrix_persistence(matrix, use_cubical=False)
        assert persistence > 0.0
        assert isinstance(persistence, float)


def test_compute_matrix_persistence_cubical():
    matrix = np.array([
        [10, 10, 10, 10, 10],
        [10,  1, 10,  2, 10],
        [10, 10, 10, 10, 10],
        [10,  3, 10,  4, 10],
        [10, 10, 10, 10, 10],
    ])
    persistence = compute_matrix_persistence(matrix, use_cubical=True)
    assert persistence > 0.0
    assert isinstance(persistence, float)
