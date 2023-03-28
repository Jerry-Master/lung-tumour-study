import argparse
from argparse import Namespace
import logging
from logging import Logger
from tumourkit import eval_segment
import os
from .preprocessing import hovernet2centroids
from .utils.preprocessing import get_names


class HovernetNotFoundError(Exception):
    pass


def run_preprocessing(args: Namespace, logger : Logger) -> None:
    """
    Converts the gson format to the rest of formats.
    """
    logger.info('Extracting Hovernet centroids from training output.')
    if not os.path.isdir(os.path.join(args.root_dir, 'data', 'tmp_hov', 'json')):
            raise HovernetNotFoundError('Please, train again or extract hovernet outputs.')
    newargs = Namespace(
        json_dir = os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'),
        output_path = os.path.join(args.root_dir, 'data', 'tmp_hov', 'centroids_hov'),
    )
    hovernet2centroids(newargs)
    for split in ['train', 'validation', 'test']:
        split_names = get_names(os.path.join(args.root_dir, 'data', split, 'gson'), '.geojson')
        with open(os.path.join(args.root_dir, 'data', split, 'names.txt'), 'w') as f:
            for name in split_names:
                print(name, file=f)
    return


def run_evaluation(args: Namespace, logger: Logger) -> None:
    logger.info('Starting evaluation of Hovernet output.')
    for split in ['train', 'validation', 'test']:
        logger.info(f'    Evaluating {split} split')
        newargs = Namespace(
            names = os.path.join(args.root_dir, 'data', split, 'names.txt'),
            gt_path = os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path = os.path.join(args.root_dir, 'data', split, 'centroids_hov'),
            save_name = args.save_name + '_' + split,
            debug_path = None,
        )
        eval_segment(newargs)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--save-name', type=str, required=True, help='Name to save the result, without file type.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    logger = logging.getLogger('eval_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_preprocessing(args, logger)
    run_evaluation(args, logger)