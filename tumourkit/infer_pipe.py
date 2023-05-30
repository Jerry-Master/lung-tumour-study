"""
Pipeline for inference.

Copyright (C) 2023  Jose Pérez Cano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact information: joseperez2000@hotmail.es
"""
import argparse
from argparse import Namespace
import logging
from logging import Logger
from .segmentation import hov_infer
import os
from .preprocessing import geojson2pngcsv_main, png2graph_main, hovernet2geojson_main, graph2centroids_main, centroidspng2csv_main, pngcsv2geojson_main
from .postprocessing import join_hovprob_graph_main
from .utils.preprocessing import create_dir
from .classification import infer_gnn
import pandas as pd


def set_best_configuration(args: Namespace, logger: Logger) -> None:
    """
    Sets the best configuration from training based on the F1 score.

    :param args: The arguments for setting the best configuration.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Configuration not provided, using best configuration from training based on F1 score.')
    if args.best_arch == 'GCN':
        save_file = os.path.join(args.root_dir, 'gnn_logs', 'gcn_results.csv')
    elif args.best_arch == 'ATT':
        save_file = os.path.join(args.root_dir, 'gnn_logs', 'gat_results.csv')
    else:
        assert False, 'Architecture not supported'
    gnn_results = pd.read_csv(save_file)
    if args.num_classes == 2:
        best_conf = gnn_results.sort_values(by='F1 Score', ascending=False).iloc[0]
    else:
        best_conf = gnn_results.sort_values(by='Weighted F1', ascending=False).iloc[0]
    args.best_num_layers = str(best_conf['NUM_LAYERS'])
    args.best_dropout = str(best_conf['DROPOUT'])
    args.best_norm_type = str(best_conf['NORM_TYPE'])
    return


def run_hovernet(args: Namespace, logger: Logger) -> None:
    """
    Runs the Hovernet inference.

    :param args: The arguments for running Hovernet inference.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Starting hovernet inference.')
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
    """
    Performs post-processing steps on Hovernet output.

    :param args: The arguments for running post-processing on Hovernet output.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Parsing Hovernet output')
    logger.info('   From json to geojson.')
    newargs = Namespace(
        json_dir=os.path.join(args.output_dir, 'tmp_hov', 'json'),
        gson_dir=os.path.join(args.output_dir, 'gson_hov'),
        num_classes=args.num_classes
    )
    hovernet2geojson_main(newargs)
    logger.info('   From geojson to pngcsv.')
    newargs = Namespace(
        gson_dir=os.path.join(args.output_dir, 'gson_hov'),
        png_dir=os.path.join(args.output_dir, 'png_hov'),
        csv_dir=os.path.join(args.output_dir, 'csv_hov'),
        num_classes=args.num_classes
    )
    geojson2pngcsv_main(newargs)
    logger.info('   From pngcsv to nodes.csv.')
    create_dir(os.path.join(args.output_dir, 'graphs'))
    newargs = Namespace(
        png_dir=os.path.join(args.output_dir, 'png_hov'),
        orig_dir=args.input_dir,
        output_path=os.path.join(args.output_dir, 'graphs', 'raw'),
        num_workers=args.num_workers
    )
    png2graph_main(newargs)
    logger.info('   Adding hovernet predictions to .nodes.csv.')
    newargs = Namespace(
        json_dir=os.path.join(args.output_dir, 'tmp_hov', 'json'),
        graph_dir=os.path.join(args.output_dir, 'graphs', 'raw'),
        output_dir=os.path.join(args.output_dir, 'graphs', 'hovpreds'),
        num_classes=args.num_classes
    )
    join_hovprob_graph_main(newargs, logger)
    return


def run_graphs(args: Namespace, logger: Logger) -> None:
    """
    Runs the graph inference.

    :param args: The arguments for running graph inference.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Starting graph inference.')
    model_name = 'best_' + args.best_arch + '_' + args.best_num_layers + '_' \
        + args.best_dropout + '_' + args.best_norm_type
    newargs = Namespace(
        node_dir=os.path.join(args.output_dir, 'graphs', 'hovpreds'),
        output_dir=os.path.join(args.output_dir, 'gnn_preds'),
        weights=os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'weights', model_name + '.pth'),
        conf=os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'confs', model_name + '.json'),
        normalizers=os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'normalizers', model_name + '.pkl'),
        num_classes=args.num_classes,
        disable_prior=False,
        disable_morph_feats=False,
    )
    infer_gnn(newargs)
    return


def run_postgraphs(args: Namespace, logger: Logger) -> None:
    """
    Performs post-processing steps on GNN output.

    :param args: The arguments for running post-processing on GNN output.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Parsing gnn output.')
    logger.info('   Converting .nodes.csv to .centroids.csv.')
    newargs = Namespace(
        graph_dir=os.path.join(args.output_dir, 'gnn_preds'),
        centroids_dir=os.path.join(args.output_dir, 'centroids'),
        num_classes=args.num_classes,
    )
    graph2centroids_main(newargs)
    logger.info('   Converting .centroids.csv and .GT_cells.png to .class.csv.')
    newargs = Namespace(
        centroids_dir=os.path.join(args.output_dir, 'centroids'),
        png_dir=os.path.join(args.output_dir, 'png_hov'),
        csv_dir=os.path.join(args.output_dir, 'csv_gnn'),
    )
    centroidspng2csv_main(newargs)
    logger.info('   Converting png/csv to geojson.')
    newargs = Namespace(
        png_dir=os.path.join(args.output_dir, 'png_hov'),
        csv_dir=os.path.join(args.output_dir, 'csv_gnn'),
        gson_dir=os.path.join(args.output_dir, 'gson_gnn'),
        num_classes=args.num_classes
    )
    pngcsv2geojson_main(newargs)
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help='Internal data and weights directory.', default='./.internals/')
    parser.add_argument('--input-dir', type=str, help='Folder containing patches to process.', required=True)
    parser.add_argument('--output-dir', type=str, help='Folder where to save results. Additional subfolder will be created.', required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--best-num-layers', type=str, help='Optimal number of layers when training GNNs.')
    parser.add_argument('--best-dropout', type=str, help='Optimal dropout rate when training GNNs')
    parser.add_argument('--best-norm-type', type=str, help='Optimal type of normalization layers when training GNNs')
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', required=True, choices=['GCN', 'ATT', 'HATT', 'SAGE', 'BOOST'])
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
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

    if args.best_num_layers is None or args.best_dropout is None or args.best_norm_type is None:
        set_best_configuration(args, logger)
    run_hovernet(args, logger)
    run_posthov(args, logger)
    run_graphs(args, logger)
    run_postgraphs(args, logger)
