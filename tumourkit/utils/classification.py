"""
Module with utility functions for the classification module.

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
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from classification.read_nodes import read_all_nodes

def normalize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes using the mean and deviation of the trainining dataset.
    """
    sc = StandardScaler()
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_val), sc.transform(X_test)

def fit_column_normalizer(node_dir: str, names: List[str]) -> StandardScaler:
    """
    Computes mean and standard deviation of features for later normalization.
    node_dir: Path to .nodes.csv files.
    names: Names of files to include. Must not have file extensions.
    """
    X, y = read_all_nodes(node_dir, [x+'.nodes.csv' for x in names])
    sc = StandardScaler()
    sc.fit(X)
    return sc