"""
Module with utility functions referring to finding nearest points in sets.

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
from scipy.spatial import KDTree
import numpy as np
from typing import List, Tuple, Optional

def generate_tree(centroids: List[Tuple[int,int,int]]) -> KDTree:
    """
    Input format: list of (x,y,class) tuples.
    """
    centroids_ = np.array(list(map(lambda x: (x[0], x[1]), centroids)))
    return KDTree(centroids_)

def find_nearest(a: Tuple[int,int,int], B: KDTree) -> int:
    """
    a: (x,y,class) tuple.
    B: KDTree to search for nearest point.
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return idx

def find_nearest_dist_idx(a: Tuple[int,int,int], B: KDTree) -> Tuple[float, int]:
    """
    a: (x,y,class) tuple.
    B: KDTree to search for nearest point.
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return dist, idx

Point = Tuple[float,float]
Contour = List[Point]
Cell = Tuple[int, int, Contour] # id, class

def get_N_closest_pairs_dists(a: Contour, b: Contour, N: int,
    threshold: Optional[float] = np.inf) -> List[float]:
    """
    Given two sets of points, 
    returns the N closests pairs distances.

    Prunes the search for distandes greater than threshold.
    Complexity: O((n+m)log(m)), being m = max(|a|,|b|) and n = min(|a|,|b|)
    """
    if len(a) > len(b):
        tree = KDTree(a)
        query_set = b
    else:
        tree = KDTree(b)
        query_set = a
    cpairs = []
    for point in query_set:
        dist, idx = tree.query(point, k=N, distance_upper_bound=threshold)
        cpairs.extend(dist)
        cpairs.sort()
        cpairs = cpairs[:N]
    return cpairs

def get_N_closest_pairs_idx(a: Contour, b: Contour, N: int,
    threshold: Optional[float] = np.inf) -> Tuple[List[int],List[int]]:
    """
    Given two sets of points, 
    returns the N closests pairs indices. 

    Prunes the search for distandes greater than threshold.
    Complexity: O((n+m)log(m)), being m = max(|a|,|b|) and n = min(|a|,|b|)
    """
    is_a_tree = len(a) > len(b)
    if is_a_tree:
        tree = KDTree(a)
        query_set = b
    else:
        tree = KDTree(b)
        query_set = a
    cpairs = []
    for idx2, point in enumerate(query_set):
        dist, idx = tree.query(point, k=N, distance_upper_bound=threshold)
        cpairs.extend(zip(dist, idx, [idx2 for _ in range(len(idx))]))
        cpairs.sort()
        cpairs = cpairs[:N]
    dists, idxa, idxb = list(zip(*cpairs))
    if not is_a_tree:
        idxa, idxb = idxb, idxa
    return list(idxa), list(idxb)