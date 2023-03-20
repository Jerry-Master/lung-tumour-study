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
import pandas as pd
import numpy as np

from tumourkit.preprocessing.centroidspng2csv import centroidspng2csv
from tumourkit.utils.preprocessing import get_names, parse_path, read_png, read_csv, read_centroids

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
THRESHOLD = 0.1


def check_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return np.abs(a.to_numpy() - b.to_numpy()).mean() < THRESHOLD


@pytest.mark.parametrize("name", get_names(PNGCSV_DIR, '.centroids.csv'))
def test_centroidspng2csv(name):
    centroids_file = read_centroids(name, PNGCSV_DIR)
    png_file = read_png(name, PNGCSV_DIR)
    csv_file = centroidspng2csv(centroids_file, png_file)
    ref_csv = read_csv(name, PNGCSV_DIR)
    if not check_equal(csv_file, ref_csv):
        assert False, name
    assert(True)
