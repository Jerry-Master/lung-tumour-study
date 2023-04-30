import os
from typing import List, Dict, Any
from tqdm import tqdm
import argparse
from argparse import Namespace
from ..utils.preprocessing import get_names, read_gson, save_geojson


def process_gson(gson: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(filter(lambda x: x['properties']['classification']['name'] != 'uncertain', gson))


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    names = get_names(args.input_dir, '.geojson')
    for name in tqdm(names):
        gson = read_gson(name, args.input_dir)
        out_gson = process_gson(gson)
        save_geojson(out_gson, name, args.output_dir)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Folder where geojsons are with label uncertain.')
    parser.add_argument('--output-dir', type=str, help='Folder where to save geojsons without label uncertain.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)