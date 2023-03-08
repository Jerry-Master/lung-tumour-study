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
import pandas as pd
import os

from tumourkit.utils.preprocessing import get_names, parse_path
from tumourkit.segmentation.evaluate import get_confusion_matrix

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
CENTROIDS_DIR = parse_path(TEST_DIR) + 'centroids/'


"""
The format of the names should be [name].A.csv and [name].B.csv,
being A and B two files to compare the metric. And [name].result.csv
should be the expected result. 
Labels should be only 1 and 2.
CSV files must have headers, except for [name].result.csv which should
only be the confusion matrix.
"""
@pytest.mark.parametrize("name", get_names(CENTROIDS_DIR, '.result.csv'))
def test_conf_matrix_sum(name):
    A_centroids = pd.read_csv(CENTROIDS_DIR + name + '.A.csv').to_numpy()
    B_centroids = pd.read_csv(CENTROIDS_DIR + name + '.B.csv').to_numpy()
    confusion_matrix = get_confusion_matrix(A_centroids, B_centroids)
    N = len(confusion_matrix)
    for i in range(1,N):
        assert confusion_matrix[i,:].sum() == (A_centroids[:,2]==i).sum()
    for i in range(1,N):    
        assert confusion_matrix[:,i].sum() == (B_centroids[:,2]==i).sum()