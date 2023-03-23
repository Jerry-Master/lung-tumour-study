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
    Returns a KDTree object from a list of (x, y, class) tuples.

    :param centroids: A list of (x, y, class) tuples representing the centroids.
    :type centroids: list[tuple(int, int, int)]
    :return: A KDTree object created from the x and y coordinates of the centroids.
    :rtype: KDTree
    """
    centroids_ = np.array(list(map(lambda x: (x[0], x[1]), centroids)))
    return KDTree(centroids_)

def find_nearest(a: Tuple[int,int,int], B: KDTree) -> int:
    """
    Finds the index of the nearest point in a KDTree to a given (x, y, class) tuple.

    :param a: The (x, y, class) tuple to search for the nearest point.
    :type a: tuple(int, int, int)
    :param B: The KDTree to search for the nearest point.
    :type B: KDTree
    :return: The index of the nearest point in the KDTree to the given tuple.
    :rtype: int
    """
    x, y = a[0], a[1]
    dist, idx = B.query([x,y], k=1)
    return idx

def find_nearest_dist_idx(a: Tuple[int,int,int], B: KDTree) -> Tuple[float, int]:
    """
    Finds the distance and index of the nearest point in a KDTree to a given (x, y, class) tuple.

    :param a: The (x, y, class) tuple to search for the nearest point.
    :type a: tuple(int, int, int)
    :param B: The KDTree to search for the nearest point.
    :type B: KDTree
    :return: A tuple containing the distance and index of the nearest point in the KDTree to the given tuple.
    :rtype: tuple(float, int)
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
    Returns the N closest pairs distances between two sets of points.

    :param a: The first set of points as a list of (x, y) tuples.
    :type a: Contour
    :param b: The second set of points as a list of (x, y) tuples.
    :type b: Contour
    :param N: The number of closest pairs to return.
    :type N: int
    :param threshold: An optional threshold to prune the search for distances greater than the threshold.
    :type threshold: float, optional
    :return: A list of the N closest pairs distances between the two sets of points.
    :rtype: List[float]

    The function has a complexity of O((n+m)log(m)), where m is the maximum of the lengths of the two sets of points and n is the minimum.
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
    Returns the indices of the N closest pairs of points between two sets of points.

    :param a: The first set of points as a list of (x, y) tuples.
    :type a: Contour
    :param b: The second set of points as a list of (x, y) tuples.
    :type b: Contour
    :param N: The number of closest pairs to return.
    :type N: int
    :param threshold: An optional threshold to prune the search for distances greater than the threshold.
    :type threshold: float, optional
    :return: A tuple containing two lists of indices, one for each set of points, of the N closest pairs of points between the two sets.
    :rtype: Tuple(List[int], List[int])

    The function has a complexity of O((n+m)log(m)), where m is the maximum of the lengths of the two sets of points and n is the minimum.
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