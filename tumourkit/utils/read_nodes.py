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
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import os
from .preprocessing import get_names


def read_node_matrix(
        file: str,
        return_coordinates: Optional[bool] = False,
        return_class: Optional[bool] = True,
        remove_prior: Optional[bool] = False,
        remove_morph: Optional[bool] = False,
        enable_background: Optional[bool] = False,
        ) -> List[np.ndarray]:
    """
    Read csv and creates X and y matrices.
    Centroids coordinates are removed.
    Labels are subtracted 1 to be in 0-1 range.
    """
    df = pd.read_csv(file)
    remove_vars = ['id', 'class', 'X', 'Y']
    if remove_morph:
        remove_vars.extend([
            'area', 'perimeter', 'std',
            'red0', 'red1', 'red2', 'red3', 'red4',
            'green0', 'green1', 'green2', 'green3', 'green4',
            'blue0', 'blue1', 'blue2', 'blue3', 'blue4'
        ])
    if remove_prior:
        remove_vars.extend(list(filter(lambda x: 'prob' in x, df.columns)))
    if enable_background:
        y_bkgr = df['background'].to_numpy()
    if 'background' in df.columns:
        df = df.drop(['background'], axis=1)
    if return_class:
        y = df['class'].to_numpy() - 1
        X = df.drop(remove_vars, axis=1).to_numpy()
        if remove_morph and remove_prior:
            X = np.zeros((len(y), 1))
        if not return_coordinates:
            if enable_background:
                return X, y, y_bkgr
            return X, y
        else:
            xx = df['X'].to_numpy()
            yy = df['Y'].to_numpy()
            if enable_background:
                return X, y, xx, yy, y_bkgr
            return X, y, xx, yy
    else:
        remove_vars.remove('class')
        X = df.drop(remove_vars, axis=1).to_numpy()
        if remove_morph and remove_prior:
            X = np.zeros((len(X), 1))
        if not return_coordinates:
            if enable_background:
                return X, None, None
            return X, None
        else:
            xx = df['X'].to_numpy()
            yy = df['Y'].to_numpy()
            if enable_background:
                return X, None, xx, yy, None
            return X, None, xx, yy


def _read_all_nodes(
        node_dir: str,
        names: List[str],
        remove_prior: Optional[bool] = False,
        remove_morph: Optional[bool] = False,
        enable_background: Optional[bool] = False,
        ) -> List[np.ndarray]:
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
        tmp = read_node_matrix(os.path.join(node_dir, name), remove_morph=remove_morph, remove_prior=remove_prior, enable_background=enable_background)
        if enable_background:
            X_, y_, y_bkgr = tmp
        else:
            X_, y_ = tmp
        if X is None:
            X = X_  # Shape (n_samples, n_features)
            y = y_  # Shape (n_samples,)
        else:
            X = np.vstack([X, X_])
            y = np.hstack([y, y_])
    return X, y


def read_all_nodes(
        node_dir: str,
        remove_prior: Optional[bool] = False,
        remove_morph: Optional[bool] = False,
        ) -> List[np.ndarray]:
    """
    Input
      node_dir: Path to folder with csv files containing node features.
    Output
      X: Input data in array format.
      y: Labels in array format.
    """
    ext = '.nodes.csv'
    names = get_names(node_dir, ext)
    names = [x + ext for x in names]
    X, y = _read_all_nodes(node_dir, names, remove_morph=remove_morph, remove_prior=remove_prior)
    return X, y


def create_node_splits(
        node_dir: str, val_size: float, test_size: float,
        seed: Optional[int] = None,
        mode: Optional[str] = 'total'
        ) -> List[np.ndarray]:
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
    names = [x + ext for x in names]
    if mode == 'total':
        X, y = _read_all_nodes(node_dir, names)
        X_tr_val, X_test, y_tr_val, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr_val, y_tr_val, test_size=val_size / (1 - test_size),
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
        val_names = names[N_tr:N_val + N_tr]
        test_names = names[N_val + N_tr:]
        X_train, y_train = _read_all_nodes(node_dir, train_names)
        X_val, y_val = _read_all_nodes(node_dir, val_names)
        X_test, y_test = _read_all_nodes(node_dir, test_names)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        assert False, 'Wrong mode.'
