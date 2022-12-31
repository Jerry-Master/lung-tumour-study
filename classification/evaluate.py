"""

Script to evaluate predictions.

"""
from typing import Dict, List
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--node-dir', type=str, required=True,
                     help='Folder with .nodes.csv prediction files.')
parser.add_argument('--save-file', type=str, required=True,
                     help='Name to file where to save the results. Must not contain extension.')
parser.add_argument('--by-img', action='store_true',
                     help='Whether to separate images in the split. Default: False.')

def check_imbalance(labels: np.ndarray) -> bool:
    """
    Returns true if there are at least two classes, and false otherwise.
    """
    return len(np.unique(labels)) > 1

def compute_metrics(nodes_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes F1-score, Accuracy, ROC AUC, Expected Calibration Error and percentage error.
    Dataframe must contain columns class and prob1.
    Class columns must have 1 for negative and 2 for positive.
    """
    labels = nodes_df['class'].to_numpy()-1
    probs = nodes_df['prob1'].to_numpy()
    preds = (probs > 0.5) * 1

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    if check_imbalance(labels):
        auc = roc_auc_score(labels, probs)
    else:
        auc = -1
    return {'Accuracy': acc, 'F1-score': f1, 'ROC AUC': auc}

def save_metrics(metrics: Dict[str, float], save_name):
    """
    Saves metrics into csv file.
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_name + '.csv', index=False)
    
def join_dictionaries(dict1: Dict[str, List[float]], dict2: Dict[str, float]) -> None:
    """
    Joins dict2 into dict1, modifying dict1.
    For each key it adds the value in dict2 to the list in dict1,
    if dict1 doesn't have such key, it is created.
    """
    for key in dict2.keys():
        if key not in dict1:
            dict1[key] = []
        dict1[key].append(dict2[key])

def main(args):
    node_files = os.listdir(args.node_dir)
    node_paths = [parse_path(args.node_dir) + x for x in node_files]
    metrics = {}
    if args.by_img:
        for node_path, node_file in zip(node_paths, node_files):
            nodes_df = pd.read_csv(node_path)
            aux_metrics = compute_metrics(nodes_df)
            aux_metrics['name'] = node_file[:-10]
            join_dictionaries(metrics, aux_metrics)
    else:
        total_nodes_df = pd.DataFrame()
        for node_path in node_paths:
            nodes_df = pd.read_csv(node_path)
            total_nodes_df = pd.concat([total_nodes_df,nodes_df])
        total_nodes_df.reset_index(inplace=True)
        total_nodes_df.drop('index', axis=1, inplace=True)
        aux_metrics = compute_metrics(total_nodes_df)
        join_dictionaries(metrics, aux_metrics)
    save_metrics(metrics, args.save_file)

if __name__=='__main__':
    args = parser.parse_args()
    main(args)
