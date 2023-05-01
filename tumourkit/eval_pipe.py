import argparse
from argparse import Namespace
import logging
from logging import Logger
import numpy as np
import pandas as pd
from . import eval_segment
import os
from .preprocessing import hovernet2centroids, geojson2pngcsv, pngcsv2centroids
from .utils.preprocessing import get_names
from .utils.pipes import HovernetNotFoundError, check_void
from .utils.classification import metrics_from_predictions


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


def compute_ece(args: Namespace, logger: Logger, split: str) -> None:
    folder = os.path.join(args.root_dir, 'data', split, 'graphs', 'preds')
    trues, probs = None, None
    for file in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder, file))
        _trues = df['class'].to_numpy()-1
        if trues is None:
            trues = _trues.reshape((-1,1))
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
    metrics_df.to_csv(args.save_name + '_' + split + '_ece.csv', index=False)


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

        compute_ece(args, logger, split)
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