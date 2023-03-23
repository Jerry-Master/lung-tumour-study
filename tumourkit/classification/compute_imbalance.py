"""
Auxiliar script to compute how imbalanced the dataset is.

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
from .read_nodes import read_all_nodes
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