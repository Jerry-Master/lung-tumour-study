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
from ..classification.read_nodes import read_all_nodes

def normalize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes the input data using the mean and standard deviation of the training dataset.

    :param X_train: The input training data with shape (n_samples, n_features).
    :type X_train: numpy.ndarray
    :param X_val: The input validation data with shape (n_samples, n_features).
    :type X_val: numpy.ndarray
    :param X_test: The input test data with shape (n_samples, n_features).
    :type X_test: numpy.ndarray
    :return: A tuple containing three numpy.ndarrays:
             - The normalized X_train with shape (n_samples, n_features).
             - The normalized X_val with shape (n_samples, n_features).
             - The normalized X_test with shape (n_samples, n_features).
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    sc = StandardScaler()
    sc.fit(X_train)
    return sc.transform(X_train), sc.transform(X_val), sc.transform(X_test)

def fit_column_normalizer(node_dir: str, names: List[str]) -> StandardScaler:
    """
    Fits a StandardScaler object to the input data and returns it.

    :param node_dir: The directory containing node data files.
    :type node_dir: str
    :param names: A list of node names to read from the node directory.
    :type names: List[str]
    :return: A StandardScaler object fitted to the input data.
    :rtype: StandardScaler
    """
    X, y = read_all_nodes(node_dir, [x+'.nodes.csv' for x in names])
    sc = StandardScaler()
    sc.fit(X)
    return sc