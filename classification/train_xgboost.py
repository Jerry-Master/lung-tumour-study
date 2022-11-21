"""

Script to train classification models.
Right now only supports classification over nodes, without edges.

"""
import argparse
from read_nodes import create_node_splits
import xgboost as xgb
import os
import pickle
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from typing import Tuple

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--graph_dir', type=str, required=True,
                     help='Folder containing .graph.csv files.')
parser.add_argument('--val_size', type=float, default=0.2,
                     help='Percentage of nodes given to validation set. Default: 0.2')
parser.add_argument('--test_size', type=float, default=0.2,
                     help='Percentage of nodes given to test set. Default: 0.2')
parser.add_argument('--seed', type=int, default=None,
                     help='Seed for random split. Default: None')
parser.add_argument('--by_img', action='store_true',
                     help='Whether to separate images in the split. Default: False.')
parser.add_argument('--num_workers', type=int, default=1)

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

    