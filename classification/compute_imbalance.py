"""

Script to train classification models.
Right now only supports classification over nodes, without edges.

"""
import argparse
from read_nodes import read_all_nodes
import os
import numpy as np

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--graph-dir', type=str, required=True,
                     help='Folder containing .graph.csv files.')

if __name__=='__main__':
    args = parser.parse_args()
    GRAPH_DIR = args.graph_dir

    X, y = read_all_nodes(GRAPH_DIR, os.listdir(GRAPH_DIR))
    print('Number of tumoral cells:', np.sum(y==1))
    print('Number of non-tumoral cells:', np.sum(y==0))