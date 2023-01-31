"""
Extract HoVer-Net probabilities from json and
concatenates the result as new column (prob1) to .nodes.csv.

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
from tqdm import tqdm
from typing import Dict, Any
import pandas as pd
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import create_dir, parse_path, get_names, read_json, save_graph

parser = argparse.ArgumentParser()
parser.add_argument(
    '--json-dir', type=str, required=True,
    help='Path to folder containing HoVer-Net json outputs.'
)
parser.add_argument(
    '--graph-dir', type=str, required=True,
    help='Path to directory to .nodes.csv containing graph information.'
)
parser.add_argument(
    '--output-dir', type=str, required=True,
    help='Path where to save new .nodes.csv. If same as --graph-dir, overwrites its content.'
)


def add_probability(graph: pd.DataFrame, hov_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Extracts type_prob from json and adds it as column prob1.
    Makes the join based on id.
    """
    graph = graph.copy()
    n_cols = len(graph.columns)
    if not 'prob1' in graph.columns:
        graph.insert(n_cols, 'prob1', [-1] * len(graph))
    for i in range(len(graph)):
        idx = graph.loc[i, 'id']
        cell = hov_json[str(idx)]
        graph.loc[i, 'prob1'] = cell['prob1']
    assert graph['prob1'].min() != -1
    return graph


def main() -> None:
    names = get_names(GRAPH_DIR, '.nodes.csv')
    for name in tqdm(names):
        graph = pd.read_csv(os.path.join(GRAPH_DIR, name + '.nodes.csv'))
        hov_json = read_json(os.path.join(JSON_DIR, name + '.json'))
        graph = add_probability(graph, hov_json)
        save_graph(graph, os.path.join(OUTPUT_DIR, name + '.nodes.csv'))
    return


if __name__=='__main__':
    args = parser.parse_args()
    JSON_DIR = parse_path(args.json_dir)
    GRAPH_DIR = parse_path(args.graph_dir)
    OUTPUT_DIR = parse_path(args.output_dir)
    create_dir(OUTPUT_DIR)
    main()