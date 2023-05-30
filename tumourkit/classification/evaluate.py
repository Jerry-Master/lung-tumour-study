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
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..utils.preprocessing import parse_path, create_dir
from ..utils.classification import metrics_from_predictions

import argparse


def abline(slope: float, intercept: float, axes: Axes) -> None:
    """
    Plot a line on the current axis given a slope and intercept.

    :param slope: The slope of the line.
    :type slope: float
    :param intercept: The y-intercept of the line.
    :type intercept: float
    :param axes: The matplotlib axes object on which to plot the line.
    :type axes: matplotlib.axes.Axes
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
    Draws a reliability diagram and saves it to the specified path.

    :param y_true: The true labels.
    :type y_true: np.ndarray
    :param y_probs: The predicted probabilities.
    :type y_probs: np.ndarray
    :param save_path: The path to save the reliability diagram (without extension).
    :type save_path: str
    :param method_name: The name of the method for the legend.
    :type method_name: str
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=20)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
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
    Computes various evaluation metrics based on the provided predictions and ground truth labels.

    This function computes the following metrics:
        - Accuracy
        - F1-score
        - ROC AUC
        - Expected Calibration Error (ECE)
        - Percentage error

    The input DataFrame must contain the following columns:
        - 'class': The ground truth labels (1 for negative, 2 for positive).
        - 'prob1': The predicted probabilities for the positive class.

    Optionally, a reliability diagram can be generated and saved to a file specified by 'draw_on'.
    The 'method_name' parameter is used for the legend in the reliability diagram.

    :param nodes_df: The DataFrame containing the predictions and ground truth labels.
    :type nodes_df: pd.DataFrame
    :param draw_on: The file path to save the reliability diagram (without extension), defaults to None.
    :type draw_on: Optional[str]
    :param method_name: The name of the method for the reliability diagram legend, defaults to 'Method'.
    :type method_name: Optional[str]

    :return: A dictionary containing the computed metrics.
    :rtype: Dict[str, float]
    """
    labels = nodes_df['class'].to_numpy() - 1
    probs = nodes_df['prob1'].to_numpy()
    preds = (probs > 0.5) * 1

    acc, f1, auc, perc_error, ece = metrics_from_predictions(labels, preds, probs, 2)

    if draw_on is not None:
        draw_reliability_diagram(labels, probs, draw_on, method_name)

    return {'Accuracy': acc, 'F1-score': f1, 'ROC AUC': auc, 'Error percentage': perc_error, 'ECE': ece}


def save_metrics(metrics: Dict[str, float], save_name):
    """
    Saves the computed metrics into a CSV file.

    :param metrics: A dictionary containing the computed metrics.
    :type metrics: Dict[str, float]
    :param save_name: The file path to save the metrics (without extension).
    :type save_name: str
    """
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(save_name + '.csv', index=False)


def join_dictionaries(dict1: Dict[str, List[float]], dict2: Dict[str, float]) -> None:
    """
    Joins the values from dict2 into dict1, modifying dict1.

    For each key in dict2, the corresponding value is added to the list in dict1
    with the same key. If dict1 does not have the key, it is created with an empty list.

    :param dict1: The dictionary to be modified.
    :type dict1: Dict[str, List[float]]
    :param dict2: The dictionary containing the values to be joined into dict1.
    :type dict2: Dict[str, float]
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
            total_nodes_df = pd.concat([total_nodes_df, nodes_df])
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
