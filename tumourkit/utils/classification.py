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
from typing import Tuple, List, Optional
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .read_nodes import _read_all_nodes
from .calibration_error import calibration_error

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

def fit_column_normalizer(
        node_dir: str,
        names: List[str],
        remove_prior: Optional[bool] = False,
        remove_morph: Optional[bool] = False,
        ) -> StandardScaler:
    """
    Fits a StandardScaler object to the input data and returns it.

    :param node_dir: The directory containing node data files.
    :type node_dir: str
    :param names: A list of node names to read from the node directory.
    :type names: List[str]
    :param remove_prior: If True, hovernet probabilites are ignored.
    :type remove_prior: Optional[bool]
    :param remove_morph: If True, morphological features are removed.
    :type remove_morph: Optional[bool]
    :return: A StandardScaler object fitted to the input data.
    :rtype: StandardScaler
    """
    X, y = _read_all_nodes(node_dir, [x+'.nodes.csv' for x in names], remove_morph=remove_morph, remove_prior=remove_prior)
    sc = StandardScaler()
    sc.fit(X)
    return sc


def check_imbalance(y_true: np.ndarray) -> bool:
    """
    Returns true if there are at least two classes, and false otherwise.
    """
    return len(np.unique(y_true)) > 1


def percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the deviation in the percentage of tumoral cells.
    """
    gt_perc = (y_true==1).sum() / len(y_true)
    pred_perc = (y_pred==1).sum() / len(y_pred)
    return abs(gt_perc - pred_perc)


def metrics_from_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        num_classes: Optional[int] = 2,
        ) -> List[float]:
    """
    Computes evaluation metrics using predictions and ground truth.
    For binary: Accuracy, F1 Score, ROC AUC, %ERR, ECE
    For multiclass: Micro F1, Macro F1, Weighted F1, ECE

    :param y_true: The ground truth to compare with, values start at 0. Shape: (N, 1).
    :type y_true: np.ndarray
    :param y_pred: Predictions with values starting at 0. Shape: (N, 1).
    :type y_pred: np.ndarray
    :param y_prob: Probabilities of the predictions. Shape: (N, num_classes) or (N, 1) if binary.
    :type y_prob: Optional[np.ndarray]
    :param num_classes: Number of classes to consider.
    :type num_classes: Optional[int]
    :return: Several metrics. Either 4 or 5 floats, depending on whether is binary or multiclass.
    :rtype: List[float]
    """
    if num_classes == 2:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if check_imbalance(y_true):
            if y_prob is not None:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = roc_auc_score(y_true, y_pred)
        else:
            auc = -1
        perc_error = percentage_error(y_true, y_pred)
        if y_prob is not None:
            ece = calibration_error(y_true, y_prob, norm='l1', reduce_bias=False)
        else:
            ece = -1
        return acc, f1, auc, perc_error, ece
    else:
        micro = f1_score(y_true, y_pred, average='micro')
        macro = f1_score(y_true, y_pred, average='macro')
        weighted = f1_score(y_true, y_pred, average='weighted')
        if y_prob is not None:
            ece = 0
            for i in range(num_classes):
                y_true_i = (y_true == i) * 1
                y_prob_i = y_prob[:, i]
                ece_i = calibration_error(y_true_i, y_prob_i, norm='l1', reduce_bias=False)
                ece += ece_i
            ece /= num_classes
        else:
            ece = -1
        return micro, macro, weighted, ece