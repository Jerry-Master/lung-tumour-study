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
from argparse import Namespace
from ..utils.read_nodes import read_all_nodes
from xgboost import XGBClassifier
from logging import Logger
import logging
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from ..utils.classification import metrics_from_predictions


def train(
        conf: Dict[str, Any], 
        X_tr: np.ndarray, 
        y_tr: np.ndarray, 
        val_size: float,
        seed: int,
        num_classes: Optional[int] = 2
        ) -> XGBClassifier:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed
    )
    if num_classes == 2:
        model = XGBClassifier(
            n_estimators=conf['n_estimators'],
            learning_rate=conf['learning_rate'],
            max_depth=conf['max_depth'],
            colsample_bytree=conf['colsample_bytree'],
            eval_metric='logloss',
            early_stopping_rounds=10
        )
    else:
        model = XGBClassifier(
            n_estimators=conf['n_estimators'],
            learning_rate=conf['learning_rate'],
            max_depth=conf['max_depth'],
            colsample_bytree=conf['colsample_bytree'],
            eval_metric='mlogloss',
            objective='multi:softmax',
            num_class=num_classes,
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
        y_val: np.ndarray,
        num_classes: Optional[int] = 2
        ) -> Tuple[float, float, float]:
    preds = model.predict(X_val)
    if num_classes == 2:
        probs = model.predict_proba(X_val)[:,1]
        acc, f1, auc, perc_err, ece = metrics_from_predictions(y_val, preds, probs, 2)
        return f1, acc, auc, perc_err, ece
    else:
        probs = model.predict_proba(X_val)
        micro, macro, weighted, ece = metrics_from_predictions(y_val, preds, probs, num_classes)
        return micro, macro, weighted, ece


def save(metrics: Dict[str,Tuple[float,float,float]], path: str) -> None:
    metrics.to_csv(path, index=False)


def cross_validate(args, conf, skf, X, y):
    if args.num_classes == 2:
        f1_mean, acc_mean, auc_mean, perc_err_mean, ece_mean = 0, 0, 0, 0, 0
    else:
        micro_mean, macro_mean, weighted_mean, ece_mean = 0, 0, 0, 0
    for train_index, val_index in skf.split(X, y):
        X_train, X_cval = X[train_index], X[val_index]
        y_train, y_cval = y[train_index], y[val_index]
        ## Train
        model = train(conf, X_train, y_train, args.val_size, args.seed, args.num_classes)
        ## Save metrics
        val_metrics = evaluate(model, X_cval, y_cval, args.num_classes)
        if args.num_classes == 2:
            f1, acc, auc, perc_err, ece = val_metrics
            f1_mean += f1; acc_mean += acc; auc_mean += auc
            perc_err_mean += perc_err; ece_mean += ece
        else:
            micro, macro, weighted, ece = val_metrics
            micro_mean += micro; macro_mean += macro; weighted_mean += weighted
            ece_mean += ece
    if args.num_classes == 2:
        f1_mean /= args.cv_folds; acc_mean /= args.cv_folds; auc_mean /= args.cv_folds
        perc_err_mean /= args.cv_folds; ece_mean /= args.cv_folds
        tmp = pd.DataFrame(
            [(*conf.values(), f1_mean, acc_mean, auc_mean, perc_err_mean, ece_mean)], 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'f1', 'accuracy', 'auc', 'perc_err', 'ece'
                ]
        )
    else:
        micro_mean /= args.cv_folds; macro_mean /= args.cv_folds; weighted_mean /= args.cv_folds
        ece_mean /= args.cv_folds
        tmp = pd.DataFrame(
            [(*conf.values(), micro_mean, macro_mean, weighted_mean, ece_mean)], 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'micro', 'macro', 'weighted', 'ece'
                ]
        )
    return tmp


def create_confs():
    confs = [{'n_estimators': n, 'learning_rate': l, 'max_depth': d, 'colsample_bytree': c}
            for n in [500] 
            for l in [0.05, 0.005] 
            for d in [8, 16] 
            for c in [0, 0.5]]
    return confs


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph-dir', type=str, required=True,
                        help='Folder containing .graph.csv files.')
    parser.add_argument('--test-graph-dir', type=str, required=True,
                        help='Folder containing .graph.csv files to evaluate at the end.')
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
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes to consider for classification (background not included).')
    return parser


def main_with_args(args: Namespace, logger: Logger):
    X, y = read_all_nodes(args.graph_dir)
    y = np.array(y, dtype=np.int32)
    skf = StratifiedKFold(n_splits=args.cv_folds)
    skf.get_n_splits(X, y)

    if args.num_classes == 2:
        metrics = pd.DataFrame(
            {}, 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'f1', 'accuracy', 'auc', 'perc_err', 'ece'
            ]
        )
    else:
        metrics = pd.DataFrame(
            {}, 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'micro', 'macro', 'weighted', 'ece'
            ]
        )
    confs = create_confs()
    logger.info('Training various XGBoost configurations')
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for conf in confs:
                future = executor.submit(cross_validate, args, conf, skf, X, y)
                futures.append(future)
            for k, future in enumerate(futures):
                tmp = future.result()
                logger.info('Configuration {:2}/{:2}'.format(k + 1, len(confs)))
                metrics = pd.concat((metrics, tmp))
                save(metrics, args.save_name + '.csv')
    else:
        for k, conf in enumerate(confs):
            logger.info('Configuration {:2}/{:2}'.format(k + 1, len(confs)))
            tmp = cross_validate(args, conf, skf, X, y)
            metrics = pd.concat((metrics, tmp))
            save(metrics, args.save_name + '.csv')
    save(metrics, args.save_name + '.csv')
    metrics = pd.read_csv(args.save_name + '.csv')
    logger.info('Selecting best XGBoost configuration.')
    if args.num_classes == 2:
        best_conf = metrics.sort_values(by='f1', ascending=False).iloc[0]
    else:
        best_conf = metrics.sort_values(by='weighted', ascending=False).iloc[0]
    tmp = {}
    tmp['n_estimators'] = int(best_conf['n_estimators'])
    tmp['learning_rate'] = float(best_conf['learning_rate'])
    tmp['max_depth'] = int(best_conf['max_depth'])
    tmp['colsample_bytree'] = float(best_conf['colsample_bytree'])
    best_conf = tmp
    logger.info('Retraining with best configuration.')
    model = train(best_conf, X, y, args.val_size, args.seed, args.num_classes)
    logger.info('Computing test metrics.')
    X_test, y_test = read_all_nodes(args.test_graph_dir)
    y_test = np.array(y_test, dtype=np.int32)
    test_metrics = evaluate(model, X_test, y_test, args.num_classes)
    if args.num_classes == 2:
        f1, acc, auc, perc_err, ece = test_metrics
        test_metrics = pd.DataFrame(
            [(*best_conf.values(), f1, acc, auc, perc_err, ece)], 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'f1', 'accuracy', 'auc', 'perc_err', 'ece'
                ]
        )
    else:
        micro, macro, weighted, ece = test_metrics
        test_metrics = pd.DataFrame(
            [(*best_conf.values(), micro, macro, weighted, ece)], 
            columns=[
                'n_estimators', 'learning_rate', 'max_depth', 'colsample_bytree',
                'micro', 'macro', 'weighted', 'ece'
                ]
        )
    save(test_metrics, args.save_name + '_test.csv')


def main():
    parser = _create_parser()
    args = parser.parse_args()

    logger = logging.getLogger('train_xgboost')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    main_with_args(args, logger)
    
    