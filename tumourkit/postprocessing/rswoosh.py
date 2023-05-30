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
from typing import Callable, Tuple, List


Point = Tuple[float, float]
Contour = List[Point]
Cell = Tuple[int, int, Contour]  # id, class


def rswoosh(
        I: List[Cell],
        isEqual: Callable[[Contour, Contour], bool],
        merge: Callable[[Cell, Cell], Cell]
        ) -> List[Cell]:
    """
    Given a list of cells, returns the same list without duplicates.

    The `rswoosh` function takes a list of cells `I`, a function `isEqual` to check equality between contours,
    and a function `merge` to merge two cells.

    :param I: The list of cells.
    :type I: List[Cell]
    :param isEqual: A function to check equality between contours.
    :type isEqual: Callable[[Contour, Contour], bool]
    :param merge: A function to merge two cells.
    :type merge: Callable[[Cell, Cell], Cell]

    :return: The list of cells without duplicates.
    :rtype: List[Cell]
    """
    out = {}  # Hash Table
    while (len(I) > 0):
        key_r, class_r, r = I.pop()  # O(1)
        exist_equal = False
        for key_s, (class_s, s) in out.items():
            if isEqual(r, s):
                del out[key_s]  # O(1)
                I.append(merge((key_r, class_r, r), (key_s, class_s, s)))  # O(1)
                exist_equal = True
                break
        if not exist_equal:
            out[key_r] = (class_r, r)  # O(1)
    return out.items()
