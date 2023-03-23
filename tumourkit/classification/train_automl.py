"""
Script to train classification models.
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
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
import numpy as np
import pickle
from datetime import datetime
from ..utils.preprocessing import create_dir, parse_path

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
parser.add_argument('--total-limit', type=int, default=1, 
                    help='Limit in time (min) to find a model. Default: 1.')
parser.add_argument('--per-model-limit', type=int, default=30, 
                    help='Limit in time (sec) to search for hyperparameters for each given model. Default: 30')
parser.add_argument('--log-dir', type=str, required=True, 
                    help='Folder to save logs.')

def read_data(GRAPH_DIR, args):
    X_train, X_val, X_test, y_train, y_val, y_test = \
        create_node_splits(
            GRAPH_DIR, args.val_size, args.test_size, args.seed, 
            'by_img' if args.by_img else 'total'
        )
    X_train = np.vstack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))
    return X_train, X_test, y_train, y_test

def train(X_train, y_train, args):
    model = AutoSklearnClassifier(
        time_left_for_this_task=args.total_limit*60, 
        per_run_time_limit=args.per_model_limit, 
        n_jobs=args.num_workers, memory_limit=1_000_000_000,
        metric=autosklearn.metrics.f1,
        scoring_functions=[
            autosklearn.metrics.f1, 
            autosklearn.metrics.accuracy, 
            autosklearn.metrics.roc_auc
        ]
    )
    # perform the search
    model.fit(X_train, y_train)
    return model

def save_model(model):
    with open(os.path.join(LOG_DIR, 'best.pkl'), 'wb') as f:
        pickle.dump(model, f)
    x = model.show_models()
    results = {"ensemble": x}
    with open(os.path.join(LOG_DIR, 'fname.pickle'),'wb') as f:
        pickle.dump(results, f)


def summarize(model, X_test, y_test):
    now = datetime.now()
    date = now.strftime("%m-%d-%Y-%H-%M-%S")
    # Standard statistics
    print(model.sprint_statistics())
    with open(os.path.join(LOG_DIR, date + '-automl.txt'), 'w') as f:
        print(model.sprint_statistics(), file = f)
    # Leaderboard of all models
    results = model.leaderboard(detailed=True)
    results.to_csv(os.path.join(LOG_DIR, date + '-results.csv'))
    # Evaluate best model
    y_hat = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    f1 = f1_score(y_test, y_hat)
    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_proba)
    print("F1 Score: %.3f" % f1)
    print("Accuracy: %.3f" % acc)
    print("ROC AUC: %.3f" % auc)
    with open(os.path.join(LOG_DIR, date + '-automl.txt'), 'a') as f:
        print("F1 Score: %.3f" % f1, file=f)
        print("Accuracy: %.3f" % acc, file=f)
        print("ROC AUC: %.3f" % auc, file=f)

def main(args):
    X_train, X_test, y_train, y_test = read_data(GRAPH_DIR, args)   
    model = train(X_train, y_train, args)
    save_model(model)
    summarize(model, X_test, y_test)

if __name__=='__main__':
    args = parser.parse_args()
    GRAPH_DIR = parse_path(args.graph_dir)
    LOG_DIR = parse_path(args.log_dir)
    create_dir(LOG_DIR)
    main(args)