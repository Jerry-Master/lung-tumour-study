"""

Script to train xgboost models.
Right now only supports classification over nodes, without edges.

Copyright (C) 2022  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
import argparse
from read_nodes import create_node_splits
import xgboost as xgb
import os
import pickle
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedKFold

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--graph-dir', type=str, required=True,
                     help='Folder containing .graph.csv files.')
parser.add_argument('--val-size', type=float, default=0.2,
                     help='Percentage of nodes given to validation set. Default: 0.2')
parser.add_argument('--test-size', type=float, default=0.2,
                     help='Percentage of nodes given to test set. Default: 0.2')
parser.add_argument('--seed', type=int, default=None,
                     help='Seed for random split. Default: None')
parser.add_argument('--by-img', action='store_true',
                     help='Whether to separate images in the split. Default: False.')
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--cv-folds', type=int, default=10)

def logprob2prob(predt: np.ndarray) -> np.ndarray:
    """
    Converts log probabilities to probabilities.
    """
    exp_predt = np.exp(predt)
    return exp_predt / (1 + exp_predt)

def binarize(predt: np.ndarray, threshold: float) -> None:
    """
    Applies thresholding. [Inplace]
    """
    predt[predt >= threshold] = 1
    predt[predt < threshold] = 0

def my_f1_score(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    predt = logprob2prob(predt)
    binarize(predt, 0.5)
    f1 = f1_score(y, predt)
    return 'F1_Score', f1 if f1 is not None else 0

def my_accuracy(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    predt = logprob2prob(predt)
    binarize(predt, 0.5)
    acc = accuracy_score(y, predt)
    return 'Accuracy', acc if acc is not None else 0

if __name__=='__main__':
    args = parser.parse_args()
    GRAPH_DIR = args.graph_dir

    X_train, X_val, X_test, y_train, y_val, y_test = \
        create_node_splits(
            GRAPH_DIR, args.val_size, args.test_size, args.seed, 
            'by_img' if args.by_img else 'total'
        )

    X = np.vstack((X_train, X_val))
    y = np.hstack((y_train, y_val))
    skf = StratifiedKFold(n_splits=args.cv_folds)
    skf.get_n_splits(X, y)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ## Train

        ## Save metrics


    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, 'train'), (dval, 'eval')]

    param = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic'}
    param['nthread'] = args.num_workers
    param['eval_metric'] = 'auc'

    num_round = 100
    evals_result = {}
    bst = xgb.train(
        param, dtrain, num_round, evals=evallist, feval=my_f1_score,
        early_stopping_rounds=10, evals_result=evals_result
    )

    with open(os.path.join(FILE_DIR, 'logs/0000.pickle'), 'wb') as f:
        pickle.dump(evals_result, f)

    bst.save_model(os.path.join(FILE_DIR,'weights/0000.model'))

    