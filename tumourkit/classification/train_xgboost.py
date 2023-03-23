"""
Script to train xgboost models.
Right now only supports classification over nodes, without edges.

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
import argparse
from .read_nodes import create_node_splits
from xgboost import XGBClassifier
import os
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--graph-dir', type=str, required=True,
                     help='Folder containing .graph.csv files.')
parser.add_argument('--val-size', type=float, default=0.2,
                     help='Validation size used for early stopping. Default: 0.2')
parser.add_argument('--seed', type=int, default=None,
                     help='Seed for random split. Default: None')
parser.add_argument('--num-workers', type=int, default=1, 
                     help='Number of processors to use. Default: 1.')
parser.add_argument('--cv-folds', type=int, default=10, 
                     help='Number of CV folds. Default: 10.')
parser.add_argument('--save-name', type=str, required=True,
                    help='Name to save the result, without file type.')


def train(
        conf: Dict[str, Any], 
        X_tr: np.ndarray, 
        y_tr: np.ndarray, 
        val_size: float,
        seed: int
        ) -> XGBClassifier:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed
    )
    model = XGBClassifier(
        n_estimators=conf['n_estimators'],
        learning_rate=conf['learning_rate'], # 0.05
        max_depth=conf['max_depth'],
        colsample_bytree=conf['colsample_bytree'],
        eval_metric='logloss',
        early_stopping_rounds=10
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)], verbose=False
    )
    return model

def evaluate(
        model: XGBClassifier, 
        X_val: np.ndarray,
        y_val: np.ndarray
        ) -> Tuple[float, float, float]:
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1]
    f1 = f1_score(y_val, preds)
    acc = accuracy_score(y_val, preds)
    auc = roc_auc_score(y_val, probs)
    return f1, acc, auc

def save(metrics: Dict[str,Tuple[float,float,float]], path: str) -> None:
    metrics.to_csv(path, index=False)

def cross_validate(args, conf, skf, X, y):
    f1_mean, acc_mean, auc_mean = 0, 0, 0
    for train_index, val_index in skf.split(X, y):
        X_train, X_cval = X[train_index], X[val_index]
        y_train, y_cval = y[train_index], y[val_index]
        ## Train
        model = train(conf, X_train, y_train, args.val_size, args.seed)
        ## Save metrics
        f1, acc, auc = evaluate(model, X_cval, y_cval)
        f1_mean += f1; acc_mean += acc; auc_mean += auc
    f1_mean /= args.cv_folds; acc_mean /= args.cv_folds; auc_mean /= args.cv_folds
    tmp = pd.DataFrame(
        [(*conf.values(), f1_mean, acc_mean, auc_mean)], 
        columns=[
            'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
            'f1', 'accuracy', 'auc'
            ]
    )
    return tmp

def create_confs():
    confs = [
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 8,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 8,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 8,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 8,'colsample_bytree': 0.5},
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 100,'learning_rate': 0.05,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.05,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 4,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 8,'colsample_bytree': 0},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 8,'colsample_bytree': 0},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 8,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 8,'colsample_bytree': 0.5},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 100,'learning_rate': 0.005,'max_depth': 16,'colsample_bytree': 0.5},
        {'n_estimators': 500,'learning_rate': 0.005,'max_depth': 16,'colsample_bytree': 0.5}
    ]
    return confs

if __name__=='__main__':
    args = parser.parse_args()
    GRAPH_DIR = args.graph_dir

    X_train, X_val, X_test, y_train, y_val, y_test = \
        create_node_splits(
            GRAPH_DIR, 0.2, 0.2, 0, 'total'
        )

    X = np.vstack((X_train, X_val, X_test))
    y = np.hstack((y_train, y_val, y_test))
    y = np.array(y, dtype=np.int32)
    skf = StratifiedKFold(n_splits=args.cv_folds)
    skf.get_n_splits(X, y)

    metrics = pd.DataFrame(
        {}, 
        columns=[
            'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
            'f1', 'accuracy', 'auc'
        ]
    )
    confs = create_confs()
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for conf in confs:
            future = executor.submit(cross_validate, args, conf, skf, X, y)
            futures.append(future)
        for k, future in enumerate(futures):
            print('Configuration {:2}/{:2}'.format(k, len(confs)), end='\r')
            tmp = future.result()
            metrics = pd.concat((metrics, tmp))
            save(metrics, args.save_name + '.csv')
    save(metrics, args.save_name + '.csv')
    