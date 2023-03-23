"""
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
import shutil
import os
from argparse import Namespace
import pandas as pd

from tumourkit.utils.preprocessing import parse_path, create_dir
from tumourkit.postprocessing import join_graph_gt
from tumourkit.postprocessing import join_hovprob_graph
from tumourkit import eval_class

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = parse_path(TEST_DIR)
JSON_DIR = TEST_DIR + 'json/'
GRAPHS_DIR = TEST_DIR + 'pipe_graphs/'
CENTROIDS_DIR = TEST_DIR + 'pipe_centroids/'
TMP_DIR = TEST_DIR + 'tmp/'


def test_hov_prob_pipe():
    create_dir(TMP_DIR)
    args = Namespace()
    args.graph_dir = GRAPHS_DIR
    args.centroids_dir = CENTROIDS_DIR
    args.output_dir = TMP_DIR
    join_graph_gt(args)
    args.node_dir = TMP_DIR
    args.save_file = TEST_DIR + 'tmp'
    args.by_img = False
    args.draw = False
    eval_class(args)
    args.json_dir = JSON_DIR
    args.graph_dir = TMP_DIR
    args.output_dir = TMP_DIR
    args.num_classes = 2
    join_hovprob_graph(args)
    args.node_dir = TMP_DIR
    args.save_file = TEST_DIR + 'tmp2'
    eval_class(args)
    res1 = pd.read_csv(TEST_DIR + 'tmp.csv')
    res2 = pd.read_csv(TEST_DIR + 'tmp2.csv')
    assert (abs(res1['Accuracy'] - res2['Accuracy']) < 0.01)[0]
    assert (abs(res1['F1-score'] - res2['F1-score']) < 0.01)[0]
    assert (abs(res1['Error percentage'] - res2['Error percentage']) < 0.01)[0]
    shutil.rmtree(TMP_DIR)
    os.remove(TEST_DIR + 'tmp.csv')
    os.remove(TEST_DIR + 'tmp2.csv')
