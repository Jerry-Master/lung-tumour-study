"""
Evaluating script for cell predictions. 

Input format
------------
Centroids in a csv with columns X,Y and class. Both for prediction and GT.

Output
------
F1-score (binary, macro and weighted), accuracy, ROC AUC and error percentage between the prediction and the GT at cell-level.

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
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

from ..utils.preprocessing import read_names, read_centroids
from ..utils.nearest import generate_tree, find_nearest
from ..utils.classification import metrics_from_predictions

import argparse
from argparse import Namespace
import logging
from logging import Logger


def get_confusion_matrix(
    gt_centroids: List[Tuple[int,int,int]], 
    pred_centroids: List[Tuple[int,int,int]]
    ) -> np.ndarray:
    """
    Each centroid is represented by a 3-tuple with (X, Y, class).
    Matrix has (N+1)x(N+1) entries, one more for background when no match is found.
    N is the maximum value encountered for class.
    """
    if len(gt_centroids) == 0 and len(pred_centroids) == 0:
        return np.array([[0]])
    if len(gt_centroids) == 0:
        N = np.max(pred_centroids[:,2])
        M = np.zeros((N+1,N+1))
        classes, freqs = np.unique(pred_centroids[:,2], return_counts=True)
        M[0, classes] += freqs
        return M
    if len(pred_centroids) == 0:
        N = np.max(gt_centroids[:,2])
        M = np.zeros((N+1,N+1))
        classes, freqs = np.unique(gt_centroids[:,2], return_counts=True)
        M[classes, 0] += freqs
        return M
    if type(gt_centroids) == list:
        gt_centroids = np.array(gt_centroids)
    if type(pred_centroids) == list:
        pred_centroids = np.array(pred_centroids)
    N = int(max(np.max(gt_centroids[:,2]), np.max(pred_centroids[:,2])))
    assert min(np.min(gt_centroids[:,2]), np.min(pred_centroids[:,2])) > 0, 'Zero should not be a class.'
    gt_tree = generate_tree(gt_centroids[:,:2])
    pred_tree = generate_tree(pred_centroids[:,:2])
    M = np.zeros((N+1,N+1)) 
    for point_id, point in enumerate(gt_centroids):
        closest_id = find_nearest(point[:2], pred_tree)
        closest = pred_centroids[closest_id]
        if point_id == find_nearest(closest[:2], gt_tree):
            M[int(point[2])][int(closest[2])] += 1 # 1-1 matchings
        else:
            M[int(point[2])][0] += 1 # GT not matched in prediction
    for point_id, point in enumerate(pred_centroids):
        closest_id = find_nearest(point[:2], gt_tree)
        closest = gt_centroids[closest_id]
        if point_id != find_nearest(closest[:2], pred_tree):
            M[0][int(point[2])] += 1 # Prediction not matched in GT
    return M

def get_pairs(
    gt_centroids: List[Tuple[int,int,int]], 
    pred_centroids: List[Tuple[int,int,int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each centroid is represented by a 3-tuple with (X, Y, class).
    Class is 1=non-tumour, 2=tumour.
    Returns true and predicted labels ordered by their correspondences.
    Returned labels start at 0.
    """
    if len(gt_centroids) == 0:
        return None, None

    gt_tree = generate_tree(gt_centroids[:,:2])
    pred_tree = generate_tree(pred_centroids[:,:2])
    true_labels, pred_labels = [], []
    for point_id, point in enumerate(gt_centroids):
        closest_id = find_nearest(point[:2], pred_tree)
        closest = pred_centroids[closest_id]
        if closest[2] != -1 and point[2] != -1 and point_id == find_nearest(closest[:2], gt_tree):
            true_labels.append(point[2]-1)
            pred_labels.append(closest[2]-1)
    return np.array(true_labels), np.array(pred_labels)


def compute_f1_score_from_matrix(conf_mat: np.ndarray, cls: int) -> float:
    """
    Returns f1 score of given class against the rest.
    If no positive or predictive positive classes are found, None is returned.
    """
    if cls == 0:
        return None
    TP = conf_mat[cls, cls]
    PP = conf_mat[:, cls].sum()
    if PP == 0:
        return None
    P = conf_mat[cls, :].sum()
    if P == 0:
        return None
    precision = TP / PP
    recall = TP / P
    if TP == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def compute_metrics_from_matrix(conf_mat: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Given confusion matrix,
    returns F1 score, Accuracy, ROC AUC and percentage error.
    """
    total = np.sum(conf_mat)
    acc = np.matrix.trace(conf_mat) / total
    n_classes = len(conf_mat)
    macro, weighted = 0, 0
    real_n_classes = n_classes
    adjust_weighted = 1
    for k in range(n_classes):
        f1_class = compute_f1_score_from_matrix(conf_mat, cls=k)
        cls_supp = conf_mat[k,:].sum() / total
        if f1_class is not None:
            macro += f1_class
            weighted += f1_class * cls_supp
        else:
            real_n_classes -= 1
            adjust_weighted -= cls_supp
    macro /= real_n_classes
    weighted /= adjust_weighted
    return macro, weighted, acc


def add_matrices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Adds two matrices even if their shapes are different.
    If B is bigger than A, A is enlarged with zeros.
    And vice versa.
    """
    if A.shape == B.shape:
        return A + B
    n, m = A.shape[0], B.shape[0]
    if m > n:
        res = np.zeros((m,m))
        res[:n,:n] += A
        res += B
    else:
        res = np.zeros((n,n))
        res[:m,:m] += B
        res += A
    return res


def save_csv(
    metrics: Dict[str, List[float]], 
    save_path: str
    ) -> None:
    """
    Saves metrics in csv format for later use.
    Columns depend on dictionary keys.
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_path + '.csv', index=False)

def save_debug_matrix(
    mat: np.ndarray,
    save_path: str
    ) -> None:
    """
    Saves confusion matrices for debug purposes.
    """
    conf_mat_df = pd.DataFrame(mat)
    conf_mat_df.to_csv(save_path + '.csv')


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', type=str, required=True,
                        help='Path to txt file with names.')
    parser.add_argument('--gt-path', type=str, required=True,
                        help='Path to GT files.')
    parser.add_argument('--pred-path', type=str, required=True,
                        help='Path to prediction files.')
    parser.add_argument('--save-name', type=str, required=True,
                        help='Name to save the result, without file type.')
    parser.add_argument('--debug-path', type=str, default=None,
                        help='Name of file where to save confusion matrices (optional).')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args: Namespace, logger: Logger):
    names = read_names(args.names)
    if args.num_classes == 2:
        metrics = {
            'Name': [], 'F1': [], 'Accuracy': [], 'ROC_AUC': [], 'Perc_err': [], 'ECE': [],
            'Macro F1 (bkgr)': [], 'Weighted F1 (bkgr)': [], 'Micro F1 (bkgr)': []
        }
    else:
        metrics = {
            'Name': [], 'Macro F1': [], 'Weighted F1': [], 'Micro F1': [], 'ECE': [],
            'Macro F1 (bkgr)': [], 'Weighted F1 (bkgr)': [], 'Micro F1 (bkgr)': []
        }
    global_conf_mat = None
    global_pred, global_true = [], []
    for k, name in enumerate(names):
        logger.info('Progress: {:2d}/{}'.format(k+1, len(names)))
        metrics['Name'].append(name)
        # Read
        gt_centroids = read_centroids(name, args.gt_path)
        pred_centroids = read_centroids(name, args.pred_path)
        # Compute pairs and confusion matrix
        conf_mat = get_confusion_matrix(gt_centroids, pred_centroids)
        if args.debug_path is not None:
            save_debug_matrix(conf_mat, args.debug_path + '_' + name)
        try:
            true_labels, pred_labels = get_pairs(gt_centroids, pred_centroids)
        except Exception as e:
            logger.error(e)
            logger.error(name)
            _metrics = [-1] * 5 if args.num_classes == 2 else [-1] * 4
            macro_bkgr, weighted_bkgr, micro_bkgr = -1, -1, -1
            true_labels, pred_labels = None, None
        # Save for later
        if true_labels is not None and pred_labels is not None:
            global_true.extend(true_labels)
            global_pred.extend(pred_labels)
        if global_conf_mat is None:
            global_conf_mat = conf_mat
        else:
            global_conf_mat = add_matrices(global_conf_mat, conf_mat)
        # Compute per image scores and percentages
        try:
            _metrics = metrics_from_predictions(true_labels, pred_labels, None, args.num_classes)
            macro_bkgr, weighted_bkgr, micro_bkgr = compute_metrics_from_matrix(conf_mat)
        except Exception as e:
            logger.error(e)
            logger.error(name)
            _metrics = [-1] * 5 if args.num_classes == 2 else [-1] * 4
            macro_bkgr, weighted_bkgr, micro_bkgr = -1, -1, -1
        if args.num_classes == 2:
            acc, f1, auc, perc_error, ece = _metrics
            metrics['F1'].append(f1)
            metrics['Accuracy'].append(acc)
            metrics['ROC_AUC'].append(auc)
            metrics['Perc_err'].append(perc_error)
            metrics['ECE'].append(ece)
            metrics['Macro F1 (bkgr)'].append(macro_bkgr)
            metrics['Weighted F1 (bkgr)'].append(weighted_bkgr)
            metrics['Micro F1 (bkgr)'].append(micro_bkgr)
        else:
            micro, macro, weighted, ece = _metrics
            metrics['Macro F1'].append(macro)
            metrics['Weighted F1'].append(weighted)
            metrics['Micro F1'].append(micro)
            metrics['ECE'].append(ece)
            metrics['Macro F1 (bkgr)'].append(macro_bkgr)
            metrics['Weighted F1 (bkgr)'].append(weighted_bkgr)
            metrics['Micro F1 (bkgr)'].append(micro_bkgr)
    if args.debug_path is not None:
        save_debug_matrix(global_conf_mat, args.debug_path + '_global')
    save_csv(metrics, args.save_name)
    # Global scores and percentages
    _global_metrics = metrics_from_predictions(np.array(global_true), np.array(global_pred), None, args.num_classes)
    macro_bkgr, weighted_bkgr, micro_bkgr = compute_metrics_from_matrix(global_conf_mat)
    if args.num_classes == 2:
        acc, f1, auc, perc_error, ece = _global_metrics
        global_metrics = {
            'Name': ['All'], 'F1': [f1], 'Accuracy': [acc], 'ROC_AUC': [auc], 'Perc_err': [perc_error], 'ECE': [ece],
            'Macro F1 (bkgr)': [macro_bkgr], 'Weighted F1 (bkgr)': [weighted_bkgr], 'Micro F1 (bkgr)': [micro_bkgr]
        }
    else:
        micro, macro, weighted, ece = _global_metrics
        global_metrics = {
            'Name': ['All'], 'Macro F1': [macro], 'Weighted F1': [weighted], 'Micro F1': [micro], 'ECE': [ece],
            'Macro F1 (bkgr)': [macro_bkgr], 'Weighted F1 (bkgr)': [weighted_bkgr], 'Micro F1 (bkgr)': [micro_bkgr]
        }
    save_csv(global_metrics, args.save_name + '_all')

def main():
    parser = _create_parser()
    args = parser.parse_args()

    logger = logging.getLogger('eval_segment')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main(args, logger)