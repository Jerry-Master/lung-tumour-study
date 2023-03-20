import argparse
from argparse import Namespace
import logging
from logging import Logger
from .segmentation import hov_infer
import os
from .preprocessing import geojson2pngcsv, pngcsv2graph, hovernet2geojson
from .postprocessing import join_hovprob_graph
from .utils.preprocessing import create_dir
from .classification import infer_gnn


def run_hovernet(args: Namespace, logger: Logger) -> None:
    logger.info('Starting hovernet inference.')
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
        'input_dir': args.input_dir,
        'output_dir': os.path.join(args.output_dir, 'tmp_hov'),
        'draw_dot': False,
        'save_qupath': False,
        'save_raw_map': False,
        'mem_usage': '0.2'
    }
    hov_infer(newargs, newsubargs, 'tile')
    return


def run_posthov(args: Namespace, logger: Logger) -> None:
    logger.info(f'Parsing Hovernet output')
    logger.info('   From json to geojson.')
    newargs = Namespace(
        json_dir = os.path.join(args.output_dir, 'tmp_hov', 'json'),
        gson_dir = os.path.join(args.output_dir, 'gson_hov')
    )
    hovernet2geojson(newargs)
    logger.info('   From geojson to pngcsv.')
    newargs = Namespace(
        gson_dir = os.path.join(args.output_dir, 'gson_hov'),
        png_dir = os.path.join(args.output_dir, 'png_hov'),
        csv_dir = os.path.join(args.output_dir, 'csv_hov')
    )
    geojson2pngcsv(newargs)
    logger.info('   From pngcsv to nodes.csv.')
    create_dir(os.path.join(args.output_dir, 'graphs'))
    newargs = Namespace(
        png_dir = os.path.join(args.output_dir, 'png_hov'),
        csv_dir = os.path.join(args.output_dir, 'csv_hov'),
        orig_dir = args.input_dir,
        output_path = os.path.join(args.output_dir, 'graphs', 'raw'),
        num_workers = 0
    )
    pngcsv2graph(newargs)
    logger.info('   Adding hovernet predictions to .nodes.csv.')
    newargs = Namespace(
        json_dir = os.path.join(args.output_dir, 'tmp_hov', 'json'),
        graph_dir = os.path.join(args.output_dir, 'graphs', 'raw'),
        output_dir = os.path.join(args.output_dir, 'graphs', 'hovpreds')
    )
    join_hovprob_graph(newargs, logger)
    return


def run_graphs(args: Namespace, logger: Logger) -> None:
    logger.info('Starting graph inference.')
    model_name = 'best_' + args.best_arch + '_' + args.best_num_layers + '_' \
        + args.best_dropout + '_' + args.best_norm_type
    newargs = Namespace(
        node_dir = os.path.join(args.output_dir, 'graphs', 'hovpreds'),
        output_dir = os.path.join(args.output_dir, 'gnn_preds'),
        weights = os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'weights', model_name + '.pth'),
        conf = os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'confs', model_name + '.json'),
        normalizers = os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'normalizers', model_name + '.pkl'),
    )
    infer_gnn(newargs)
    return


def run_postgraphs(args: Namespace, logger: Logger) -> None:
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help='Internal data and weights directory.', default='./internals/')
    parser.add_argument('--input-dir', type=str, help='Folder containing patches to process.', required=True)
    parser.add_argument('--output-dir', type=str, help='Folder where to save results. Additional subfolder will be created.', required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--best-num-layers', type=str, help='Optimal number of layers when training GNNs.', required=True)
    parser.add_argument('--best-dropout', type=str, help='Optimal dropout rate when training GNNs', required=True)
    parser.add_argument('--best-norm-type', type=str, help='Optimal type of normalization layers when training GNNs', required=True)
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', required=True)
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    create_dir(args.output_dir)

    logger = logging.getLogger('infer_pipe')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    run_hovernet(args, logger)
    run_posthov(args, logger)
    run_graphs(args, logger)
    run_postgraphs(args, logger)