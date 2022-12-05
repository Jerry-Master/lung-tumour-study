"""

Script to train classification models.
Right now only supports classification over nodes, without edges.

"""
import argparse
from read_nodes import create_node_splits
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from autosklearn.classification import AutoSklearnClassifier
import numpy as np

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
parser.add_argument('--total_limit', type=int, default=1, 
                    help='Limit in time to find a model.')
parser.add_argument('--per_model_limit', type=int, default=1, 
                    help='Limit in time to search for hyperparameters for each given model.')

if __name__=='__main__':
    args = parser.parse_args()
    GRAPH_DIR = args.graph_dir

    X_train, X_val, X_test, y_train, y_val, y_test = \
        create_node_splits(
            GRAPH_DIR, args.val_size, args.test_size, args.seed, 
            'by_img' if args.by_img else 'total'
        )
    X_train = np.vstack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))
    # define search
    model = AutoSklearnClassifier(
        time_left_for_this_task=args.total_limit*60, 
        per_run_time_limit=args.per_model_limit, 
        n_jobs=args.num_workers
    )
    # perform the search
    model.fit(X_train, y_train)
    # summarize
    print(model.sprint_statistics())
    # evaluate best model
    y_hat = model.predict(X_test)
    f1 = f1_score(y_test, y_hat)
    acc = accuracy_score(y_test, y_hat)
    auc = roc_auc_score(y_test, y_hat)
    print("F1 Score: %.3f" % f1)
    print("Accuracy: %.3f" % acc)
    print("ROC AUC: %.3f" % auc)