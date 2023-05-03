import shutil
import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
import pandas as pd
from .segmentation import hov_train, hov_infer
from .utils.pipes import check_training, WrongConfigurationError, HovernetNotFoundError, check_void
from .preprocessing import hovernet2centroids, geojson2pngcsv, pngcsv2centroids
from .segmentation import pngcsv2npy
from .utils.preprocessing import get_names
from . import eval_segment
from .classification import train_xgb, train_gnn


def run_preprocessing(args: Namespace, logger: Logger) -> None:
    if args.experiment == 'cnn-gnn' or args.experiment == 'xgb-gnn' or args.experiment == 'void-gnn':
        check_training(args, logger)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.experiment == 'scaling':
        if args.pretrained_path is None:
            raise WrongConfigurationError('You must provide a path to pretrained Hovernet weights for the scaling experiment.')
    return


def hovernet_preproc_with_shape(shape: str, args: Namespace, logger : Logger) -> None:
    """
    Converts the gson format to the rest of formats.
    """
    real_shape = shape[:-2] if 'FT' in shape else shape
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'data', real_shape), exist_ok=True)
    for split in ['train', 'validation', 'test']:
        logger.info(f'Parsing {split} split')
        if check_void(os.path.join(args.root_dir, 'data', split, 'png')) or check_void(os.path.join(args.root_dir, 'data', split, 'csv')):
            logger.info('   From geojson to pngcsv.')
            newargs = Namespace(
                gson_dir = os.path.join(args.root_dir, 'data', split, 'gson'),
                png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
                num_classes = args.num_classes,
            )
            geojson2pngcsv(newargs)
        os.makedirs(os.path.join(args.output_dir, 'hovernet', 'data', real_shape, split), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'hovernet', 'data', real_shape, split, 'npy'), exist_ok=True)
        if check_void(os.path.join(args.output_dir, 'hovernet', 'data', real_shape, split, 'npy')):
            newargs = Namespace(
                orig_dir = os.path.join(args.root_dir, 'data', 'orig'),
                png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
                out_dir = os.path.join(args.output_dir, 'hovernet', 'data', real_shape, split, 'npy'),
                save_example = False, use_labels = True, split = False,
                shape = real_shape
            )
            pngcsv2npy(newargs)
    return


def train_hovernet_with_shape(shape: str, args: Namespace, logger: Logger) -> None:
    logger.info(f'Starting hovernet preprocessing of {shape}.')
    hovernet_preproc_with_shape(shape, args, logger)
    logger.info(f'Starting training of {shape}.')
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'weights', shape), exist_ok=True)
    real_shape = shape[:-2] if 'FT' in shape else shape
    newargs = Namespace(
        gpu = args.gpu, view = None, save_name = None,
        log_dir = os.path.join(args.output_dir, 'hovernet', 'weights', shape),
        train_dir = os.path.join(args.output_dir, 'hovernet', 'data', real_shape, 'train', 'npy'),
        valid_dir = os.path.join(args.output_dir, 'hovernet', 'data', real_shape, 'validation', 'npy'),
        pretrained_path = args.pretrained_path if 'FT' in shape else None,
        shape = real_shape,
        num_classes = args.num_classes,
    )
    hov_train(newargs)


def infer_hovernet_with_shape(shape: str, args: Namespace, logger: Logger) -> None:
    logger.info(f'Starting inference of {shape}.')
    newargs = {
        'nr_types': str(args.num_classes + 1),
        'type_info_path': os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet', 'type_info.json'),
        'gpu': args.gpu,
        'nr_inference_workers': '0',
        'model_path': os.path.join(args.output_dir, 'hovernet', 'weights', shape, '01', 'net_epoch=50.tar'),
        'batch_size': '10',
        'shape': shape[:-2] if 'FT' in shape else shape,
        'nr_post_proc_workers': '0',
        'model_mode': 'original',
        'help': False
    }
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'output', shape), exist_ok=True)
    newsubargs = {
        'input_dir': os.path.join(args.root_dir, 'data', 'orig'),
        'output_dir': os.path.join(args.output_dir, 'hovernet', 'output', shape),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')


def run_postprocessing_with_shape(shape: str, args: Namespace, logger : Logger) -> None:
    """
    Converts the gson format to the rest of formats.
    """
    logger.info('Extracting Hovernet centroids from training output.')
    if not os.path.isdir(os.path.join(args.output_dir, 'hovernet', 'output', shape, 'json')):
        raise HovernetNotFoundError('Please, train again or extract hovernet outputs.')
    newargs = Namespace(
        json_dir = os.path.join(args.output_dir, 'hovernet', 'output', shape, 'json'),
        output_path = os.path.join(args.output_dir, 'hovernet', 'output', shape, 'centroids_hov'),
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


def evaluate_hovernet_with_shape(shape: str, args: Namespace, logger: Logger) -> None:
    logger.info(f'Starting evaluation of {shape}.')
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'output', shape, 'results'), exist_ok=True)
    for split in ['train', 'validation', 'test']:
        logger.info(f'    Evaluating {split} split')
        newargs = Namespace(
            names = os.path.join(args.root_dir, 'data', split, 'names.txt'),
            gt_path = os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path = os.path.join(args.output_dir, 'hovernet', 'output', shape, 'centroids_hov'),
            save_name = os.path.join(args.output_dir, 'hovernet', 'output', shape, 'results', shape + '_' + split),
            debug_path = None,
            num_classes = args.num_classes
        )
        eval_segment(newargs, logger)
    

def run_scaling(args: Namespace, logger: Logger) -> None:
    os.makedirs(os.path.join(args.output_dir, 'hovernet'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'weights'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'data'), exist_ok=True)
    train_hovernet_with_shape('270', args, logger)
    train_hovernet_with_shape('270FT', args, logger)
    train_hovernet_with_shape('518', args, logger)
    train_hovernet_with_shape('518FT', args, logger)

    os.makedirs(os.path.join(args.output_dir, 'hovernet', 'output'), exist_ok=True)
    infer_hovernet_with_shape('270', args, logger)
    infer_hovernet_with_shape('270FT', args, logger)
    infer_hovernet_with_shape('518', args, logger)
    infer_hovernet_with_shape('518FT', args, logger)

    run_postprocessing_with_shape('270', args, logger)
    run_postprocessing_with_shape('270FT', args, logger)
    run_postprocessing_with_shape('518', args, logger)
    run_postprocessing_with_shape('518FT', args, logger)

    evaluate_hovernet_with_shape('270', args, logger)
    evaluate_hovernet_with_shape('270FT', args, logger)
    evaluate_hovernet_with_shape('518', args, logger)
    evaluate_hovernet_with_shape('518FT', args, logger)


def run_xgb(args: Namespace, logger: Logger) -> None:
    logger.info('Moving graphs files into one single folder.')
    all_graphs_dir = os.path.join(args.root_dir, 'data', 'train_validation', 'graphs', 'preds')
    os.makedirs(all_graphs_dir, exist_ok=True)
    tr_graphs_dir = os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds')
    val_graphs_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds')
    files = [os.path.join(tr_graphs_dir, f) for f in os.listdir(tr_graphs_dir)] + \
            [os.path.join(val_graphs_dir, f) for f in os.listdir(val_graphs_dir)]
    for file in files:
        shutil.copy(file, all_graphs_dir)
    logger.info('Starting XGBoost training.')
    os.makedirs(os.path.join(args.output_dir, 'xgb'), exist_ok=True)
    newargs = Namespace(
        graph_dir = all_graphs_dir,
        test_graph_dir = os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        val_size = 0.2,
        seed = 0,
        num_workers = args.num_workers,
        cv_folds = 5,
        save_name = os.path.join(args.output_dir, 'xgb', 'cv_results'),
        num_classes = args.num_classes,
    )
    train_xgb(newargs, logger)


def run_void(args: Namespace, logger: Logger) -> None:
    logger.info('Training GNN without prior.')
    newargs = Namespace(
        train_node_dir = os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        test_node_dir = os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        log_dir = os.path.join(args.output_dir, 'gnn_no_prior_logs'),
        early_stopping_rounds = 10,
        batch_size = 10,
        model_name = 'GCN',
        save_file = os.path.join(args.output_dir, 'gnn_no_prior_logs', 'gnn_results'),
        num_confs = 32,
        save_dir = os.path.join(args.root_dir, 'weights', 'classification', 'gnn_no_prior'),
        device = 'cpu' if args.gpu == '' else 'cuda',
        num_workers = args.num_workers,
        checkpoint_iters = -1,
        num_classes = args.num_classes,
        disable_prior = True,
        disable_morph_feats = False,
    )
    train_gnn(newargs)

    logger.info('Training GNN without morphological features.')
    newargs = Namespace(
        train_node_dir = os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        test_node_dir = os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        log_dir = os.path.join(args.output_dir, 'gnn_no_morph_logs'),
        early_stopping_rounds = 10,
        batch_size = 10,
        model_name = 'GCN',
        save_file = os.path.join(args.output_dir, 'gnn_no_morph_logs', 'gnn_results'),
        num_confs = 32,
        save_dir = os.path.join(args.root_dir, 'weights', 'classification', 'gnn_no_morph'),
        device = 'cpu' if args.gpu == '' else 'cuda',
        num_workers = args.num_workers,
        checkpoint_iters = -1,
        num_classes = args.num_classes,
        disable_prior = False,
        disable_morph_feats = True,
    )
    train_gnn(newargs)

    logger.info('Training void GNN.')
    newargs = Namespace(
        train_node_dir = os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        test_node_dir = os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        log_dir = os.path.join(args.output_dir, 'gnn_void_logs'),
        early_stopping_rounds = 10,
        batch_size = 10,
        model_name = 'GCN',
        save_file = os.path.join(args.output_dir, 'gnn_void_logs', 'gnn_results'),
        num_confs = 32,
        save_dir = os.path.join(args.root_dir, 'weights', 'classification', 'gnn_void'),
        device = 'cpu' if args.gpu == '' else 'cuda',
        num_workers = args.num_workers,
        checkpoint_iters = -1,
        num_classes = args.num_classes,
        disable_prior = True,
        disable_morph_feats = True,
    )
    train_gnn(newargs)


def run_cnn(args: Namespace, logger: Logger) -> None:
    raise NotImplementedError


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help='Internal data and weights directory.', default='./.internals/')
    parser.add_argument('--output-dir', type=str, help='Folder where to save all the results.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--experiment', type=str, choices=['scaling', 'xgb-gnn', 'void-gnn', 'cnn-gnn'])
    parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    
    logger = logging.getLogger('research_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_preprocessing(args, logger)
    if args.experiment == 'scaling':
        run_scaling(args, logger)
    if args.experiment == 'xgb-gnn':
        run_xgb(args, logger)
    if args.experiment == 'void-gnn':
        run_void(args, logger)
    if args.experiment == 'cnn-gnn':
        run_cnn(args, logger)
