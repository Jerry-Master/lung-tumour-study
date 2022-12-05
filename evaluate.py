"""

Evaluating script for cell predictions. 

Input format
------------
Centroids in a csv with columns X,Y and class. Both for prediction and GT.

Output
------
F1-score, accuracy, ROC AUC and error percentage between the prediction and the GT at cell-level.

"""
import pandas as pd
import numpy as np
import argparse
import time
from utils.preprocessing import *
from utils.nearest import *
from typing import List, Tuple
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, required=True,
                    help='Path to txt file with names.')
parser.add_argument('--gt-path', type=str, required=True,
                    help='Path to GT files.')
parser.add_argument('--pred-path', type=str, required=True,
                    help='Path to prediction files.')
parser.add_argument('--save-name', type=str, required=True,
                    help='Name to save the result, without file type.')

def get_confusion_matrix(
    gt_centroids: List[Tuple[int,int,int]], 
    pred_centroids: List[Tuple[int,int,int]]
    ) -> np.ndarray:
    """
    Each centroid is represented by a 3-tuple with (X, Y, class).
    Class is 1=non-tumour, 2=tumour.
    """
    if len(gt_centroids) == 0:
        return None

    gt_tree = generate_tree(gt_centroids[:,:2])
    pred_tree = generate_tree(pred_centroids[:,:2])
    M = np.zeros((2,2)) 
    for point_id, point in enumerate(gt_centroids):
        closest_id = find_nearest(point[:2], pred_tree)
        closest = pred_centroids[closest_id]
        if closest[2] != -1 and point[2] != -1 and point_id == find_nearest(closest[:2], gt_tree):
            M[int(point[2]-1)][int(closest[2]-1)] += 1
    return M

def get_pairs(
    gt_centroids: List[Tuple[int,int,int]], 
    pred_centroids: List[Tuple[int,int,int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each centroid is represented by a 3-tuple with (X, Y, class).
    Class is 1=non-tumour, 2=tumour.
    Returns true and predicted labels ordered by their correspondences.
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

def compute_percentage(arr: np.ndarray) -> float:
    """
    arr is an array of integers representing classes
    It returns the the percentage of class 2: #2 / (#1 + #2)
    """
    n_one = np.sum(arr==1)
    n_two = np.sum(arr==2)
    return n_two / (n_one + n_two), n_one, n_two

def compute_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Given arrays of binary prediction and true labels,
    returns F1 score, Accuracy, ROC AUC and percentage error.
    """
    f1 = f1_score(true_labels, pred_labels, zero_division=1)
    acc = accuracy_score(true_labels, pred_labels)
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, pred_labels)
    else:
        auc = -1
    true_perc, _, _ = compute_percentage(true_labels+1)
    pred_perc, _, _ = compute_percentage(pred_labels+1)
    err = abs(true_perc - pred_perc)
    return f1, acc, auc, err

def save_csv(
    metrics: List[Tuple[str,float,float,float,float,float,float]], 
    save_path: str
    ) -> None:
    """
    Saves metrics in csv format for later use.
    Columns: 'name', 'F1', 'Accuracy', 'ROC_AUC', 'Perc_err'
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_path + '.csv', index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    names = read_names(args.names)
    metrics = {'Name': [], 'F1': [], 'Accuracy': [], 'ROC_AUC': [], 'Perc_err': []}
    global_conf_mat = None
    global_gt_one, global_gt_two, global_pred_one, global_pred_two = 0,0,0,0
    global_pred, global_true = [], []
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        metrics['Name'].append(name)
        # Read
        gt_centroids = read_centroids(name, args.gt_path)
        pred_centroids = read_centroids(name, args.pred_path)
        pred_centroids[pred_centroids[:,2]==0] = 1
        # Make pairs
        true_labels, pred_labels = get_pairs(gt_centroids, pred_centroids)
        if true_labels is None:
            continue
        # Save for later
        global_true.extend(true_labels)
        global_pred.extend(pred_labels)
        # Compute per image scores
        try:
            f1, acc, auc, err = compute_metrics(true_labels, pred_labels)
        except Exception as e:
            print(e)
            print(name)
        metrics['F1'].append(f1)
        metrics['Accuracy'].append(acc)
        metrics['ROC_AUC'].append(auc)
        metrics['Perc_err'].append(err)
    save_csv(metrics, args.save_name)
    # Global scores and percentages
    f1, acc, auc, err = compute_metrics(np.array(global_true), np.array(global_pred))
    global_metrics = {'Name': ['All'], 'F1': [f1], 'Accuracy': [acc], 'ROC_AUC': [auc], 'Perc_err': [err]}
    save_csv(global_metrics, args.save_name + '_all')

