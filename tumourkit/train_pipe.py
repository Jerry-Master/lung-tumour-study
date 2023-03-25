import argparse
from argparse import Namespace
from .preprocessing import geojson2pngcsv, pngcsv2graph, hovernet2geojson, pngcsv2centroids
from .segmentation import pngcsv2npy
import os
import logging
from logging import Logger
from .segmentation import hov_train, hov_infer
from .utils.preprocessing import get_names
import shutil
from .postprocessing import join_graph_gt, join_hovprob_graph
from .classification import train_gnn


def run_preproc_pipe(args: Namespace, logger : Logger) -> None:
    """
    Converts the gson format to the rest of formats.
    """
    for split in ['train', 'validation', 'test']:
        logger.info(f'Parsing {split} split')
        newargs = Namespace(
            gson_dir = os.path.join(args.root_dir, 'data', split, 'gson'),
            png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
            num_classes = args.num_classes,
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


def run_hov_pipe(args: Namespace, logger : Logger) -> None:
    """
    Trains hovernet and predicts cell contours on json format.
    """
    logger.info('Starting training.')
    newargs = Namespace(
        gpu = args.gpu, view = None, save_name = None,
        log_dir = os.path.join(args.root_dir, 'weights', 'segmentation', 'hovernet'),
        train_dir = os.path.join(args.root_dir, 'data', 'train', 'npy'),
        valid_dir = os.path.join(args.root_dir, 'data', 'validation', 'npy'),
        pretrained_path = args.pretrained_path,
        shape = '518',
        num_classes = args.num_classes,
    )
    hov_train(newargs)
    logger.info('Starting inference.')
    newargs = {
        'nr_types': str(args.num_classes + 1),
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


def run_postproc_pipe(args: Namespace, logger : Logger) -> None:
    """
    Converts the json format to the graph format containing GT and preds.
    """
    logger.info('Moving json files to corresponding folders.')
    tr_files = set(get_names(os.path.join(args.root_dir, 'data', 'train', 'gson'), '.geojson'))
    val_files = set(get_names(os.path.join(args.root_dir, 'data', 'validation', 'gson'), '.geojson'))
    ts_files = set(get_names(os.path.join(args.root_dir, 'data', 'test', 'gson'), '.geojson'))
    json_files = set(get_names(os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'), '.json'))
    for folder_name, split_files in zip(['train', 'validation', 'test'], [tr_files, val_files, ts_files]):
        for file in json_files.intersection(split_files):
            shutil.copy(
                os.path.join(args.root_dir, 'data', 'tmp_hov', 'json', file + '.json'),
                os.path.join(args.root_dir, 'data', folder_name, 'json')
            )
    for split in ['train', 'validation', 'test']:
        logger.info(f'Parsing {split} split')
        logger.info('   From json to geojson.')
        newargs = Namespace(
            json_dir = os.path.join(args.root_dir, 'data', split, 'json'),
            gson_dir = os.path.join(args.root_dir, 'data', split, 'gson_hov'),
            num_classes = args.num_classes
        )
        hovernet2geojson(newargs)
        logger.info('   From geojson to pngcsv.')
        newargs = Namespace(
            gson_dir = os.path.join(args.root_dir, 'data', split, 'gson_hov'),
            png_dir = os.path.join(args.root_dir, 'data', split, 'png_hov'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv_hov'),
            num_classes = args.num_classes,
        )
        geojson2pngcsv(newargs)
        logger.info('   From pngcsv to nodes.csv.')
        newargs = Namespace(
            png_dir = os.path.join(args.root_dir, 'data', split, 'png_hov'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv_hov'),
            orig_dir = os.path.join(args.root_dir, 'data', 'orig'),
            output_path = os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            num_workers = args.num_workers
        )
        pngcsv2graph(newargs)
        logger.info('   Extracting centroids from GT.')
        newargs = Namespace(
            png_dir = os.path.join(args.root_dir, 'data', split, 'png'),
            csv_dir = os.path.join(args.root_dir, 'data', split, 'csv'),
            output_path = os.path.join(args.root_dir, 'data', split, 'centroids')
        )
        pngcsv2centroids(newargs)
        logger.info('   Adding GT labels to .nodes.csv.')
        newargs = Namespace(
            graph_dir = os.path.join(args.root_dir, 'data', split, 'graphs', 'raw'),
            centroids_dir = os.path.join(args.root_dir, 'data', split, 'centroids'),
            output_dir = os.path.join(args.root_dir, 'data', split, 'graphs', 'GT')
        )
        join_graph_gt(newargs)
        logger.info('   Adding hovernet predictions to .nodes.csv.')
        newargs = Namespace(
            json_dir = os.path.join(args.root_dir, 'data', split, 'json'),
            graph_dir = os.path.join(args.root_dir, 'data', split, 'graphs', 'GT'),
            output_dir = os.path.join(args.root_dir, 'data', split, 'graphs', 'preds'),
            num_classes = args.num_classes,
        )
        join_hovprob_graph(newargs, logger)
    return


def run_graph_pipe(args: Namespace, logger : Logger) -> None:
    """
    Trains the graph models.
    """
    newargs = Namespace(
        train_node_dir = os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        validation_node_dir = os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        test_node_dir = os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
        log_dir = os.path.join(args.root_dir, 'gnn_logs'),
        early_stopping_rounds = 10,
        batch_size = 10,
        model_name = 'GCN',
        save_file = os.path.join(args.root_dir, 'gnn_logs', 'gnn_results'),
        num_confs = 32,
        save_dir = os.path.join(args.root_dir, 'weights', 'classification', 'gnn'),
        device = 'cpu' if args.gpu == '' else 'cuda',
        num_workers = args.num_workers,
        checkpoint_iters = -1,
        num_classes = args.num_classes
    )
    train_gnn(newargs)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--pretrained-path', type=str, help='Path to initial Hovernet weights.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    return parser


def main():
    # TODO: Run each subpipe independently and measure time and memory requirements.

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
    run_preproc_pipe(args, logger)
    logger.info('Finished preprocessing pipeline.')
    logger.info('Starting Hovernet pipeline.')
    run_hov_pipe(args, logger)
    logger.info('Finished Hovernet pipeline.')
    logger.info('Starting postprocessing pipeline.')
    run_postproc_pipe(args, logger)
    logger.info('Finished postprocessing pipeline.')
    logger.info('Starting graph pipeline.')
    run_graph_pipe(args, logger)
    logger.info('Finished graph pipeline.')
