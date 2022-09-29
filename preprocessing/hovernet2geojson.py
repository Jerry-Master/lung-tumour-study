"""

Converts from HoVernet json to QuPath geojson

"""
import json
import pandas as pd
import argparse
import geojson
from geojson import Feature, Polygon
import sys
import os

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import *

parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str, default='./',
                    help='Path to json files.')
parser.add_argument('--gson_dir', type=str, default='./',
                    help='Path to save files.')

def parse_contours(nuc):
    contours_ = []
    for inst in nuc:
        inst_info = nuc[inst]
        inst_type = inst_info['type']
        if inst_type == 1 or inst_type == 2:
            inst_contour = inst_info['contour']
            inst_contour.append(inst_contour[0])
            contours_.append((inst_contour, inst_type)) 
    return contours_


def save_contours(out_dir, name, contours):
    create_dir(parse_path(out_dir))
    with open(out_dir + name + '.geojson', 'w') as f:
        geojson.dump(contours, f, sort_keys=True, indent=2)


if __name__ == '__main__':
    args = parser.parse_args()
    names = get_names(args.json_dir, '.json')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        json_path = args.json_dir + name + '.json'
        nuc = read_json(json_path)
        contours = parse_contours(nuc)
        features = create_geojson(contours)
        save_contours(args.gson_dir, name, features)
