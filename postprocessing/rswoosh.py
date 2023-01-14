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

Point = Tuple[float,float]
Contour = List[Point]
Cell = Tuple[int, int, Contour] # id, class

def rswoosh(
    I: List[Cell], 
    isEqual: Callable[[Contour,Contour], bool], 
    merge: Callable[[Cell, Cell], Cell]
    ) -> List[Cell]:
    """
    Given a list of cells, returns the same list without duplicates.
    Must provide a function to check equality and another to merge cells.
    """
    O = {} # Hash Table
    while(len(I) > 0):
        key_r, class_r, r = I.pop() # O(1)
        exist_equal = False
        for key_s, (class_s, s) in O.items():
            if isEqual(r,s):
                del O[key_s] # O(1)
                I.append(merge((key_r, class_r, r),(key_s, class_s, s))) # O(1)
                exist_equal = True
                break
        if not exist_equal:
            O[key_r] = (class_r, r) # O(1)
    return O.items()