"""
Script to parse the json format of the centroids of Hovernet
into a CSV format with columns X,Y,class
The class output format is 0=non-tumour, 1=tumour.

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
from typing import Dict, Any, List, Tuple
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from ..utils.preprocessing import create_dir, read_json, parse_path, get_names


def parse_centroids(nuc: Dict[str, Any]) -> List[Tuple[int,int,int]]:
    """
    Input: Hovernet json nuclei dictionary as given by read_json.
    Output: List of (X,Y,class) tuples representing centroids.
    """
    centroids_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_centroid = inst_info['centroid']
        inst_type = inst_info['type']
        if inst_type == 0:
            logging.warning('Found cell with class 0, removing it.')
        else:
            centroids_.append((inst_centroid[1], inst_centroid[0], inst_type)) 
    return centroids_


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-dir', type=str, required=True,
                        help='Path to json files.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    OUTPUT_PATH = parse_path(args.output_path)
    JSON_DIR = parse_path(args.json_dir)
    create_dir(OUTPUT_PATH)
    names = get_names(JSON_DIR, '.json')
    for name in tqdm(names):
        json_path = JSON_DIR + name + '.json'
        nuc = read_json(json_path)
        centroids = parse_centroids(nuc)
        df = pd.DataFrame(centroids, columns=['X','Y','class'])
        df.to_csv(OUTPUT_PATH + name + '.centroids.csv', index=False)
