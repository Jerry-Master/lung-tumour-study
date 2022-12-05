"""

Script to parse the json format of the centroids of Hovernet
into a CSV format with columns X,Y,class
The class output format is 0=non-tumour, 1=tumour.

Copyright (C) 2022  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>
"""
import pandas as pd
import argparse
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--json-dir', type=str, required=True,
                    help='Path to json files.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to save files.')


def parse_centroids(nuc: Dict[str, Any]) -> list[tuple[int,int,int]]:
    """
    Input: Hovernet json nuclei dictionary as given by read_json.
    Output: List of (X,Y,class) tuples representing centroids.
    """
    centroids_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_centroid = inst_info['centroid']
        inst_type = inst_info['type']
        centroids_.append((inst_centroid[1], inst_centroid[0], inst_type)) 
    return centroids_

if __name__ == '__main__':
    args = parser.parse_args()
    names = get_names(args.json_dir, '.json')
    create_dir(parse_path(args.output_path))
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        json_path = args.json_dir + name + '.json'
        nuc = read_json(json_path)
        centroids = parse_centroids(nuc)
        df = pd.DataFrame(centroids, columns=['X','Y','class'])
        df.to_csv(args.output_path + name + '.centroids.csv', index=False)
            
