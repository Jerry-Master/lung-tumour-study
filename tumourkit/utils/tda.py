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
import math
from gudhi import RipsComplex, CubicalComplex


def compute_matrix_persistence(matrix: np.ndarray, use_cubical: bool = False) -> float:
    """
    Computes total 0-dimensional persistence for a single matrix.
    Uses either the Rips or Cubical complex from Gudhi.
    """
    assert len(matrix.shape) == 2, 'Matrix persistence can only be computed from a matrix.'
    if use_cubical:
        # Use Cubical complex directly on 2D matrix
        cc = CubicalComplex(top_dimensional_cells=np.abs(matrix))
        cc.persistence(homology_coeff_field=2)
        diag = cc.persistence()
    else:
        # Flatten to 1D point cloud for Rips
        points = np.expand_dims(np.abs(matrix).flatten(), axis=1)
        rc = RipsComplex(points=points)
        st = rc.create_simplex_tree(max_dimension=1)
        st.persistence(homology_coeff_field=2, persistence_dim_max=True)
        diag = st.persistence()

    # Total 0-dimensional persistence (finite deaths)
    h0_pairs = [pair for pair in diag if pair[0] == 0 and pair[1][1] != float('inf')]
    total_persistence = math.sqrt(sum((death - birth) ** 2 for _, (birth, death) in h0_pairs))
    return total_persistence