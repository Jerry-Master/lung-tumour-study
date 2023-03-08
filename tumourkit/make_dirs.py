import os
import argparse
import logging
from typing import Union, Dict, List


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    return parser


def create_subfolders(node: Union[Dict, List], current_folder: str) -> None:
    if isinstance(node, Dict):
        for key, value in node.items():
            subfolder = os.path.join(current_folder, key)
            os.mkdir(subfolder)
            create_subfolders(value, subfolder)
    elif isinstance(node, List):
        for value in node:
            subfolder = os.path.join(current_folder, value)
            os.mkdir(subfolder)
    else:
        assert False, 'Wrong folder structure format.'


def main():
    parser = _create_parser()
    args = parser.parse_args()
    if os.path.exists(args.root_dir):
        logging.warning('Root folder already exists, aborting.')
    else:
        structure = {
            'data': {
                'train': {
                    'png': [], 'csv': [], 'gson': [], 'json': [],
                    'graphs': ['raw', 'preds', 'GT'], 'npy': []
                },
                'validation': {
                    'png': [], 'csv': [], 'gson': [], 'json': [],
                    'graphs': ['raw', 'preds', 'GT'], 'npy': []
                },
                'test': {
                    'png': [], 'csv': [], 'gson': [], 'json': [],
                    'graphs': ['raw', 'preds', 'GT'], 'npy': []
                },
                'orig': []
            },
            'weights': {
                'segmentation': {
                    'hovernet': [], 'cellnet': ['count', 'segment']
                },
                'classification': {
                    'automl': [], 'xgb': [],
                    'gnn': ['confs', 'normalizers', 'weights']
                }
            }
        }
        os.mkdir(args.root_dir)
        create_subfolders(structure, args.root_dir)