import argparse
from argparse import Namespace
import logging
from logging import Logger
from . import eval_segment
import os
from .preprocessing import hovernet2centroids, geojson2pngcsv, pngcsv2centroids
from .utils.preprocessing import get_names
from .utils.pipes import HovernetNotFoundError, check_void


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
        if check_void(os.path.join(args.root_dir, 'data', split, 'names.txt')):
            logger.info(f'Preprocessing split {split}.')
            split_names = get_names(os.path.join(args.root_dir, 'data', split, 'gson'), '.geojson')
            with open(os.path.join(args.root_dir, 'data', split, 'names.txt'), 'w') as f:
                for name in split_names:
                    print(name, file=f)
        if check_void(os.path.join(args.root_dir, 'data', split, 'png')) or check_void(os.path.join(args.root_dir, 'data', split, 'csv')):
            logger.info('   From geojson to pngcsv.')
            newargs = Namespace(
                gson_dir = os.path.join(args.root_dir, 'data', split, 'gson'),
                png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
                num_classes = args.num_classes,
            )
            geojson2pngcsv(newargs)
        if check_void(os.path.join(args.root_dir, 'data', split, 'centroids')):
            logger.info('   Extracting centroids from GT.')
            newargs = Namespace(
                png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
                output_path = os.path.join(args.root_dir, 'data', split, 'centroids')
            )
            pngcsv2centroids(newargs)
    return


def run_evaluation(args: Namespace, logger: Logger) -> None:
    logger.info('Starting evaluation of Hovernet output.')
    for split in ['train', 'validation', 'test']:
        logger.info(f'    Evaluating {split} split')
        newargs = Namespace(
            names = os.path.join(args.root_dir, 'data', split, 'names.txt'),
            gt_path = os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path = os.path.join(args.root_dir, 'data', 'tmp_hov', 'centroids_hov'),
            save_name = args.save_name + '_' + split,
            debug_path = None,
            num_classes = args.num_classes
        )
        eval_segment(newargs, logger)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--save-name', type=str, required=True, help='Name to save the result, without file type.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
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