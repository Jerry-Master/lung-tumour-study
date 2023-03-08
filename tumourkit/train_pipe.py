import argparse
from argparse import Namespace
from .preprocessing import geojson2pngcsv
from .segmentation import pngcsv2npy
import os
import logging


def run_preproc_pipe(args: Namespace) -> None:
    """
    Converts the gson format to the rest of formats.
    """
    for split in ['train', 'validation', 'test']:
        newargs = Namespace(
            gson_dir = os.path.join(args.root_dir, 'data', split, 'gson'),
            png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv')
        )
        geojson2pngcsv(newargs)
        newargs = Namespace(
            orig_dir = os.path.join(args.root_dir, 'data', 'orig'),
            png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
            out_dir = os.path.join(args.root_dir, 'data', split, 'npy'),
            save_example = False, use_labels = True, split = False,
            shape = '518'
        )
        pngcsv2npy(newargs)
    return


def run_hov_pipe(args: Namespace) -> None:
    """
    Trains hovernet and predicts cell contours on json format.
    """
    return


def run_postproc_pipe(args: Namespace) -> None:
    """
    Converts the json format to the graph format containing GT and preds.
    """
    return


def run_graph_pipe(args: Namespace) -> None:
    """
    Trains the graph models.
    """
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    logger = logging.getLogger('train_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Starting preprocessing pipeline.')
    run_preproc_pipe(args)
    logger.info('Finished preprocessing pipeline.')
    logger.info('Starting Hovernet pipeline.')
    run_hov_pipe(args)
    logger.info('Finished Hovernet pipeline.')
    logger.info('Starting postprocessing pipeline.')
    run_postproc_pipe(args)
    logger.info('Finished postprocessing pipeline.')
    logger.info('Starting graph pipeline.')
    run_graph_pipe(args)
    logger.info('Finished graph pipeline.')