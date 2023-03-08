import argparse
from argparse import Namespace
from .preprocessing import geojson2pngcsv
from .segmentation import pngcsv2npy
import os
import logging
from .segmentation import hov_train, hov_infer


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
    newargs = Namespace(
        gpu = args.gpu, view = None, save_name = None,
        log_dir = os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'),
        train_dir = os.path.join(args.root_dir, 'data', 'train', 'npy'),
        valid_dir = os.path.join(args.root_dir, 'data', 'validation', 'npy'),
        pretrained_path = args.pretrained_path,
        shape = '518'
    )
    # hov_train(newargs)
    newargs = {
        'nr_types': '3',
        'type_info_path': os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', 'type_info.json'),
        'gpu': args.gpu,
        'nr_inference_workers': '0',
        'model_path': os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', '01', 'net_epoch=50.tar'),
        'batch_size': '10',
        'shape': '518',
        'nr_post_proc_workers': '0',
        'model_mode': 'original',
        'help': False
    }
    newsubargs = {
        'input_dir': os.path.join(args.root_dir, 'data', 'orig'),
        'output_dir': os.path.join(args.root_dir, 'data', 'tmp_hov'),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')
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
    parser.add_argument('--pretrained-path', type=str, help='Path to initial weights.')
    parser.add_argument('--gpu', type=str, default='')
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
    # run_preproc_pipe(args)
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