"""
Pipeline for evaluation of models.

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
import numpy as np
import pandas as pd
from . import eval_segment
import os
import shutil
from .postprocessing import join_hovprob_graph_main, join_graph_gt_main
from .preprocessing import hovernet2centroids_main, geojson2pngcsv_main, pngcsv2centroids_main, graph2centroids_main
from .utils.preprocessing import get_names
from .utils.pipes import HovernetNotFoundError, check_void
from .utils.classification import metrics_from_predictions
from .classification import infer_gnn
from .infer_pipe import set_best_configuration


def run_preprocessing(args: Namespace, logger: Logger) -> None:
    """
    Runs the preprocessing steps to convert the gson format to other formats.

    :param args: The arguments for the preprocessing.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger

    :raises HovernetNotFoundError: If the Hovernet centroids are not found in the training output.
    """
    logger.info('Extracting Hovernet centroids from training output.')
    if not os.path.isdir(os.path.join(args.root_dir, 'data', 'tmp_hov', 'json')):
        raise HovernetNotFoundError('Please, train again or extract hovernet outputs.')
    if check_void(os.path.join(args.root_dir, 'data', 'tmp_hov', 'centroids_hov')):
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'),
            output_path=os.path.join(args.root_dir, 'data', 'tmp_hov', 'centroids_hov'),
        )
        hovernet2centroids_main(newargs)
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
                gson_dir=os.path.join(args.root_dir, 'data', split, 'gson'),
                png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
                num_classes=args.num_classes,
            )
            geojson2pngcsv_main(newargs)
        if check_void(os.path.join(args.root_dir, 'data', split, 'centroids')):
            logger.info('   Extracting centroids from GT.')
            newargs = Namespace(
                png_dir=os.path.join(args.root_dir, 'data', split, 'png'),
                csv_dir=os.path.join(args.root_dir, 'data', split, 'csv'),
                output_path=os.path.join(args.root_dir, 'data', split, 'centroids')
            )
            pngcsv2centroids_main(newargs)

    if args.best_arch is None:
        args.best_arch = 'GCN'
        set_best_configuration(args, logger)
    # Perform inference steps from infer_pipe
    if check_void(os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds')):
        logger.info('Starting graph inference.')
        os.makedirs(os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds'), exist_ok=True)
        os.makedirs(os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'raw'), exist_ok=True)
        os.makedirs(os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds'), exist_ok=True)
        for split in ['train', 'validation', 'test']: # Move raw morphological features
            for file in os.listdir(os.path.join(args.root_dir, 'data', split, 'graphs', 'raw')):
                shutil.copy(
                    os.path.join(args.root_dir, 'data', split, 'graphs', 'raw', file),
                    os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'raw')
                )
        # Add hov probabilities
        newargs = Namespace(
            json_dir=os.path.join(args.root_dir, 'data', 'tmp_hov', 'json'),
            graph_dir=os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'raw'),
            output_dir=os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds'),
            num_classes=args.num_classes,
        )
        join_hovprob_graph_main(newargs, logger)
        for file in os.listdir(os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds')):
            df = pd.read_csv(os.path.join(os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds'), file))
            assert (df['class'].to_numpy() == ((df['prob1'].to_numpy() > 0.5) * 1 + 1)).all(), f"Probs and prediction of Hovernet don't coincide. {file}"
            df.drop('class', axis=1).to_csv(os.path.join(os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds'), file), index=False)
        model_name = 'best_' + args.best_arch + '_' + args.best_num_layers + '_' \
            + args.best_dropout + '_' + args.best_norm_type
        if args.gnn_dir is None:
            args.gnn_dir = os.path.join(args.root_dir, 'weights', 'classification', 'gnn')
        newargs = Namespace(
            node_dir=os.path.join(args.root_dir, 'tmp_graphs', 'graphs', 'hovpreds'),
            output_dir=os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds'),
            weights=os.path.join(args.gnn_dir, 'weights', model_name + '.pth'),
            conf=os.path.join(args.gnn_dir, 'confs', model_name + '.json'),
            normalizers=os.path.join(args.gnn_dir, 'normalizers', model_name + '.pkl'),
            num_classes=args.num_classes,
            disable_prior=args.disable_prior,
            disable_morph_feats=args.disable_morph,
        )
        infer_gnn(newargs)
    # Postproc to obtain centroids
    if check_void(os.path.join(args.root_dir, 'tmp_graphs', 'centroids')):
        logger.info('Parsing gnn output.')
        logger.info('   Converting .nodes.csv to .centroids.csv.')
        newargs = Namespace(
            graph_dir=os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds'),
            centroids_dir=os.path.join(args.root_dir, 'tmp_graphs', 'centroids'),
            num_classes=args.num_classes,
        )
        graph2centroids_main(newargs)

    logger.info('Creating folders.')
    os.makedirs(os.path.join(args.save_dir, 'conf-matrices', 'gnn_individual'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'conf-matrices', 'hov_individual'), exist_ok=True)
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(args.save_dir, split), exist_ok=True)
    return


def compute_ece(args: Namespace, split: str, is_hov: bool) -> None:
    """
    Computes Expected Calibration Error (ECE) and other metrics for the given split.

    :param args: The arguments for computing ECE and metrics.
    :type args: Namespace

    :param split: The name of the split (e.g., 'train', 'validation', 'test').
    :type split: str

    :param is_hov: Whether we are computing for hovernet or gnns. The directories have different structure, so we need to know.
    :type is_hov: bool
    """
    if is_hov:
        folder = os.path.join(args.root_dir, 'data', split, 'graphs', 'preds')
    else:
        folder = os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds_gt', split)
    trues, probs = None, None
    for file in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, file))
        _trues = df['class'].to_numpy() - 1
        if trues is None:
            trues = _trues.reshape((-1, 1))
        else:
            trues = np.vstack((trues, _trues.reshape((-1, 1))))
        if args.num_classes == 2:
            _probs = df['prob1'].to_numpy().reshape((-1, 1))
            _probs = np.hstack((1 - _probs, _probs))
        else:
            cols = ['prob' + str(k) for k in range(1, args.num_classes + 1)]
            _probs = df[cols].to_numpy()
        if probs is None:
            probs = _probs
        else:
            probs = np.vstack((probs, _probs))
        if pd.isna(trues).any():
            print(file)
    preds = np.argmax(probs, axis=1).reshape((-1, 1))
    metrics = metrics_from_predictions(trues, preds, probs[:, 1], args.num_classes)
    if args.num_classes == 2:
        acc, f1, auc, perc_error, ece = metrics
        dic_metrics = {
            'F1': [f1], 'Accuracy': [acc], 'ROC_AUC': [auc], 'Perc_err': [perc_error], 'ECE': [ece]
        }
    else:
        micro, macro, weighted, ece = metrics
        dic_metrics = {
            'Macro F1': [macro], 'Weighted F1': [weighted], 'Micro F1': [micro], 'ECE': [ece]
        }
    metrics_df = pd.DataFrame(dic_metrics)
    metrics_df.to_csv(os.path.join(args.save_dir, split, ('hov' if is_hov else 'gnn') + '_ece.csv'), index=False)


def run_evaluation(args: Namespace, logger: Logger) -> None:
    """
    Runs the evaluation of Hovernet output for different splits.

    :param args: The arguments for the evaluation.
    :type args: Namespace

    :param logger: The logger object used for logging messages.
    :type logger: Logger
    """
    logger.info('Starting evaluation of Hovernet output.')
    for split in ['train', 'validation', 'test']:
        logger.info(f'    Evaluating {split} split')
        newargs = Namespace(
            names=os.path.join(args.root_dir, 'data', split, 'names.txt'),
            gt_path=os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path=os.path.join(args.root_dir, 'data', 'tmp_hov', 'centroids_hov'),
            save_name=os.path.join(args.save_dir, split, 'hov'),
            debug_path=os.path.join(args.save_dir, 'conf-matrices', 'hov_individual', 'debug_hov') if args.debug else None,
            num_classes=args.num_classes
        )
        eval_segment(newargs, logger)
        compute_ece(args, split, is_hov=True)
        if args.debug:
            shutil.move(
                os.path.join(args.save_dir, 'conf-matrices', 'hov_individual', 'debug_hov_global.csv'),
                os.path.join(args.save_dir, 'conf-matrices', 'debug_hov_global_' + split + '.csv'))
    # Evaluate graph output same as hov output
    logger.info('Starting evaluation of Hovernet output.')
    for split in ['train', 'validation', 'test']:
        logger.info(f'    Evaluating {split} split (with background)')
        newargs = Namespace(
            names=os.path.join(args.root_dir, 'data', split, 'names.txt'),
            gt_path=os.path.join(args.root_dir, 'data', split, 'centroids'),
            pred_path=os.path.join(args.root_dir, 'tmp_graphs', 'centroids'),
            save_name=os.path.join(args.save_dir, split, 'gnn'),
            debug_path=os.path.join(args.save_dir, 'conf-matrices', 'gnn_individual', 'debug_gnn') if args.debug else None,
            num_classes=args.num_classes
        )
        eval_segment(newargs, logger)
        if args.debug:
            shutil.move(
                os.path.join(args.save_dir, 'conf-matrices', 'gnn_individual', 'debug_gnn_global.csv'),
                os.path.join(args.save_dir, 'conf-matrices', 'debug_gnn_global_' + split + '.csv'))
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds_gt', split), exist_ok=True)
        for file in get_names(os.path.join(args.root_dir, 'data', split, 'graphs', 'preds'), '.nodes.csv'):
            df_gt = pd.read_csv(os.path.join(args.root_dir, 'data', split, 'graphs', 'preds', file + '.nodes.csv'))
            df = pd.read_csv(os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds', file + '.nodes.csv'))
            # import pdb;pdb.set_trace()
            df = df[df.id.isin(df_gt.id)]
            df = df.set_index(df.id)
            df_gt = df_gt.set_index(df_gt.id)
            df['class'] = df_gt['class']
            assert not df['class'].isna().any()
            df.to_csv(os.path.join(args.root_dir, 'tmp_graphs', 'gnn_preds_gt', split, file + '.nodes.csv'), index=False)
        compute_ece(args, split, is_hov=False)
    shutil.rmtree(os.path.join(args.root_dir, 'tmp_graphs'))
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./.internals/', help='Root folder to save data and models.')
    parser.add_argument('--save-dir', type=str, required=True, help='Folder to save the results, without file type.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
    parser.add_argument('--best-num-layers', type=str, help='Optimal number of layers when training GNNs.')
    parser.add_argument('--best-dropout', type=str, help='Optimal dropout rate when training GNNs')
    parser.add_argument('--best-norm-type', type=str, help='Optimal type of normalization layers when training GNNs')
    parser.add_argument('--best-arch', type=str, help='Best architecture (convolutional, attention, ...) when training GNNs', choices=['GCN', 'ATT'])
    parser.add_argument('--debug', action='store_true', help='Whether to save confusion matrices.')
    parser.add_argument('--gnn-dir', type=str, help='Path where the gnn is saved, otherwise the default in root-dir will be used. It must have three folder inside: confs, weights and normalizers.')
    parser.add_argument('--disable-prior', action='store_true', help='Whether to disable the use of Hovernet probabilities.')
    parser.add_argument('--disable-morph', action='store_true', help='Whether to disable the use of morphological properties.')
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
