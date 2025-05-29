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
from gudhi import CubicalComplex


try:
    import dionysus as d
    HAS_DIONYSUS = True
except ImportError:
    HAS_DIONYSUS = False


def compute_matrix_persistence(matrix: np.ndarray, use_cubical: bool = False) -> float:
    """
    Computes total 0-dimensional persistence for a single matrix.
    Uses either the Rips or Cubical complex from Gudhi.
    """
    assert len(matrix.shape) == 2, 'Matrix persistence can only be computed from a matrix.'
    if use_cubical:
        # Use Cubical complex directly on 2D matrix
        cc = CubicalComplex(top_dimensional_cells=np.abs(matrix))
        diag = cc.persistence()
    else:
        # Check for Dionysus and platform compatibility
        if not HAS_DIONYSUS:
            raise RuntimeError(
                "Rips persistence requires Dionysus, which is not installed or not supported on Windows.\n"
                "To use this mode, install Dionysus on Linux/macOS or use WSL."
            )
        # Use Dionysus for Rips complex on flattened point cloud
        points = np.abs(matrix).flatten().astype(np.float32)
        points = np.column_stack((points, np.zeros_like(points)))
        f = d.fill_rips(points, 1, np.inf)
        p = d.homology_persistence(f)
        dgms = d.init_diagrams(p, f)
        diag = [(dim, (pt.birth, pt.death)) for dim, dgm in enumerate(dgms) for pt in dgm]

    # Total 0-dimensional persistence (finite deaths)
    h0_pairs = [pair for pair in diag if pair[0] == 0 and pair[1][1] != float('inf')]
    total_persistence = math.sqrt(sum((death - birth) ** 2 for _, (birth, death) in h0_pairs))
    return total_persistence