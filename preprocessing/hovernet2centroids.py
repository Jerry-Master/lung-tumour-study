"""

Script to parse the json format of the centroids of Hovernet
into a CSV format with columns X,Y,class
The class output format is 0=non-tumour, 1=tumour.

"""
import json
import pandas as pd
import argparse
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, required=True,
                    help='Path to json files.')
parser.add_argument('--output_path', type=str, required=True,
                    help='Path to save files.')

def read_json(json_dir):
    with open(json_dir) as json_file:
        data = json.load(json_file)
        nuc_info = data['nuc']
    return nuc_info

def parse_centroids(nuc):
    centroids_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_centroid = inst_info['centroid']
        inst_type = inst_info['type']
        centroids_.append((inst_centroid[1], inst_centroid[0], inst_type-1)) # Minus 1 because initial range is [1,2]
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
            
