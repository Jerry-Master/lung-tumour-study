"""
Module with utility functions for reading node attributes.
Contains functions to convert csv files into matrices.

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
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
from ..utils.preprocessing import *


def read_node_matrix(file: str, return_coordinates: Optional[bool] = False, return_class: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read csv and creates X and y matrices.
    Centroids coordinates are removed.
    Labels are subtracted 1 to be in 0-1 range.
    """
    df = pd.read_csv(file)
    if return_class:
        y = df['class'].to_numpy() - 1
        X = df.drop(['id', 'class', 'X', 'Y'], axis=1).to_numpy()
        if not return_coordinates:
            return X, y
        else:
            xx = df['X'].to_numpy()
            yy = df['Y'].to_numpy()
            return X, y, xx, yy
    else:
        X = df.drop(['id', 'X', 'Y'], axis=1).to_numpy()
        if not return_coordinates:
            return X, None
        else:
            xx = df['X'].to_numpy()
            yy = df['Y'].to_numpy()
            return X, None, xx, yy

def read_all_nodes(node_dir: str, names: List[str]) -> List[np.ndarray]:
    """
    Input
      node_dir: Path to folder with csv files containing node features.
      names: List of files to read. Must have file extension.
    Output
      X: Input data in array format.
      y: Labels in array format.
    """
    X, y = None, None
    for name in names:
        X_, y_ = read_node_matrix(os.path.join(node_dir, name))
        if X is None: 
            X = X_ # Shape (n_samples, n_features)
            y = y_ # Shape (n_samples,)
        else:
            X = np.vstack([X, X_])
            y = np.hstack([y, y_])
    return X, y

def create_node_splits(
    node_dir: str, val_size: float, test_size: float, 
    seed: Optional[int] = None,
    mode: Optional[str] = 'total') -> List[np.ndarray]:
    """
    Input
      node_dir: Path to folder with csv files containing node features.
      val_size: Percentage of data to use as validation.
      test_size: Percentage of data to use as test.
      seed: Seed for the random split.
      mode: Whether to mix images in the splits or not. It can be 'total' or 'by_img'. 
    Output
      X_train, X_val, X_test, y_train, y_val, y_test: Node features and labels.
    """
    ext = '.nodes.csv'
    names = get_names(node_dir, ext)
    names = [x+ext for x in names]
    if mode == 'total':
        X, y = read_all_nodes(node_dir, names)
        X_tr_val, X_test, y_tr_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr_val, y_tr_val, test_size=val_size / (1-test_size), 
            stratify=y_tr_val, random_state=seed
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    elif mode == 'by_img':
        random.seed(seed)
        random.shuffle(names)
        N = len(names)
        N_ts = int(N * test_size)
        N_val = int(N * val_size)
        N_tr = N - N_val - N_ts
        train_names = names[:N_tr]
        val_names = names[N_tr:N_val+N_tr]
        test_names = names[N_val+N_tr:]
        X_train, y_train = read_all_nodes(node_dir, train_names)
        X_val, y_val = read_all_nodes(node_dir, val_names)
        X_test, y_test = read_all_nodes(node_dir, test_names)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        assert False, 'Wrong mode.'