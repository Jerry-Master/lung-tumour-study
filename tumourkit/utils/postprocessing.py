"""
Module with utility functions for postprocessing.

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

from typing import Callable, Tuple, List
from scipy.spatial import KDTree
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.nearest import get_N_closest_pairs_dists, get_N_closest_pairs_idx

Point = Tuple[float,float]
Contour = List[Point]
Cell = Tuple[int, int, Contour] # id, class

def create_comparator(threshold: float, num_frontier: int) -> Callable[[Contour,Contour], bool]:
    """
    Returns comparator between two contours.
    Two contours are equal if their num_frontier closest pairs 
    are at a distance lower than threshold.
    """
    def is_equal(a: Contour, b: Contour) -> bool:
        """
        Two contours are equal if their num_frontier closest pairs 
        are at a distance lower than threshold.
        """
        cpairs_dists = get_N_closest_pairs_dists(a, b, num_frontier)
        for dist in cpairs_dists:
            if dist > threshold:
                return False
        return True
    return is_equal

def get_greatest_connected_component(idx: List[int], max_idx: int) -> Tuple[int,int]:
    """
    Given a list of indices, returns the left and right value of the 
    greatest connected component.
    Two indices are considered connected if the differ in one unit.
    0 and max_idx are considered connected.
    """
    idx.sort()
    l_final, l_aux, r_final, r_aux = 0, 0, 0, 0
    found_comp = False
    for k in range(1,len(idx)):
        if idx[k-1]+1 != idx[k]:
            found_comp = True
            # End of component
            r_aux = k-1
            if  r_aux - l_aux > r_final - l_final:
                # Save greatest component
                r_final, l_final = r_aux, l_aux
            # Restart left auxiliary index
            l_aux = k
    if not found_comp:
        return idx[0], idx[-1]
    if idx[0] == 0 and idx[-1] == max_idx:
        # Circular case
        r_aux = 1
        while r_aux < len(idx) and idx[r_aux-1] + 1 == idx[r_aux]:
            r_aux += 1
        r_aux -= 1
        l_aux = len(idx)-1
        while l_aux >= 0 and idx[l_aux-1] + 1 == idx[l_aux]:
            l_aux -= 1
        if (r_aux-0) + (len(idx)-l_aux) > r_final - l_final:
            r_final, l_final = r_aux, l_aux
    return idx[l_final], idx[r_final]

def remove_idx(a: Cell, a_idx: List[int]) -> Cell:
    """
    Removes all the indices in a's contour such that they are included
    in a_idx greatest connected component.
    """
    left, right = get_greatest_connected_component(a_idx, len(a[2])-1)
    res = []
    if left > right: # Contour is in third position
        res.extend(a[2][right+1:left])
    elif left <= right:
        res.extend(a[2][right+1:])
        res.extend(a[2][0:left]) 
    return (a[0], a[1], res)

def check_order(a: Cell, b: Cell) -> Cell:
    """NOT IMPLEMENTED"""
    # Checks if joining a and b creates a cross in the middle.
    return True

def join(a: Cell, b: Cell) -> Cell:
    """
    Appends all the points in b's contour to a's contour.
    Returns a new cell but it possibly modifies the input cells.
    """
    good_order = check_order(a, b)
    if not good_order:
        b[2].reverse()
    a[2].extend(b[2])
    return a

def merge_cells(a: Cell, b: Cell) -> Cell:
    """
    Joins two cells by removing their 30 closest pairs
    and connecting the endpoints.
    """
    a_idx, b_idx = get_N_closest_pairs_idx(a[2], b[2], 30)
    a = remove_idx(a, a_idx)
    b = remove_idx(b, b_idx)
    c = join(a, b)
    return c