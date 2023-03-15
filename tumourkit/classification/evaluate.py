"""
Script to evaluate predictions.

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
from typing import Dict, List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..utils.preprocessing import parse_path, create_dir
from .calibration_error import calibration_error

import argparse


def check_imbalance(labels: np.ndarray) -> bool:
    """
    Returns true if there are at least two classes, and false otherwise.
    """
    return len(np.unique(labels)) > 1

def percentage_error(labels: np.ndarray, preds: np.ndarray) -> float:
    """
    Computes the deviation in the percentage of tumoral cells.
    """
    gt_perc = (labels==1).sum() / len(labels)
    pred_perc = (preds==1).sum() / len(preds)
    return abs(gt_perc - pred_perc)

def abline(slope: float, intercept: float, axes: Axes) -> None:
    """
    Plot a line from slope and intercept on current axis.
    """
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    axes.plot(x_vals, y_vals, linestyle='dotted', color='k')

def draw_reliability_diagram(
    y_true: np.ndarray, 
    y_probs: np.ndarray, 
    save_path: str, 
    method_name: str
    ) -> None:
    """
    Draws reliability diagram into save_path.
    save_path must not contain extension.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=20)

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].plot(prob_pred, prob_true, marker='s', color='forestgreen')
    abline(1, 0, ax[0])
    ax[0].set(
        title='Reliability diagram', 
        xlabel='Mean predicted probability', 
        ylabel='Fraction of positives'
    )
    ax[0].legend([method_name, 'Perfectly calibrated'])

    ax[1].hist(
        y_probs,
        range=(0, 1),
        bins=20,
        label=method_name,
        color='forestgreen',
    )
    ax[1].set(title=method_name, xlabel="Mean predicted probability", ylabel="Count")

    fig.savefig(save_path + '.png')
    plt.close(fig)

def compute_metrics(
    nodes_df: pd.DataFrame, 
    draw_on: Optional[str] = None, 
    method_name: Optional[str] = 'Method'
    ) -> Dict[str, float]:
    """
    Computes F1-score, Accuracy, ROC AUC, Expected Calibration Error and percentage error.
    Dataframe must contain columns class and prob1.
    Class columns must have 1 for negative and 2 for positive.
    """
    labels = nodes_df['class'].to_numpy()-1
    probs = nodes_df['prob1'].to_numpy()
    preds = (probs > 0.5) * 1

    if draw_on is not None:
        draw_reliability_diagram(labels, probs, draw_on, method_name)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)
    if check_imbalance(labels):
        auc = roc_auc_score(labels, probs)
    else:
        auc = -1
    perc_error = percentage_error(labels, preds)
    ece = calibration_error(labels, probs, norm='l1', reduce_bias=False)
    return {'Accuracy': acc, 'F1-score': f1, 'ROC AUC': auc, 'Error percentage': perc_error, 'ECE': ece}

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


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node-dir', type=str, required=True,
                        help='Folder with .nodes.csv prediction files.')
    parser.add_argument('--save-file', type=str, required=True,
                        help='Name to file where to save the results. Must not contain extension.')
    parser.add_argument('--by-img', action='store_true',
                        help='Whether to evaluate images separately or totally.')
    parser.add_argument('--draw', action='store_true',
                        help='Whether to draw reliability diagrams.')
    parser.add_argument('--draw-dir', type=str, 
                        help='Folder to save reliability diagram on by-img mode.')
    parser.add_argument('--name', type=str, default='Method', 
                        help='The name of the model used. Default: Method.')
    return parser


def main_with_args(args):
    node_files = os.listdir(args.node_dir)
    node_paths = [parse_path(args.node_dir) + x for x in node_files]
    metrics = {}
    if args.by_img:
        for node_path, node_file in zip(node_paths, node_files):
            nodes_df = pd.read_csv(node_path)
            if args.draw:
                assert args.draw_dir is not None, 'In by-img mode you must provide a folder for saving drawings.'
                draw_dir = parse_path(args.draw_dir)
                create_dir(draw_dir)
                aux_metrics = compute_metrics(nodes_df, draw_on=draw_dir + node_file, method_name=args.name)
            else:
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
        if args.draw:
            aux_metrics = compute_metrics(total_nodes_df, draw_on=args.save_file, method_name=args.name)
        else:
            aux_metrics = compute_metrics(total_nodes_df)
        join_dictionaries(metrics, aux_metrics)
    save_metrics(metrics, args.save_file)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
