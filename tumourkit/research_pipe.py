import argparse
from argparse import Namespace
import logging
from logging import Logger
import os
from typing import List
import pandas as pd


class WrongConfigurationError(Exception):
    pass


def check_void(path: str) -> bool:
    """Tells if a directory if void or if file exists. True if void, False otherwise."""
    if os.path.isdir(path):
        files = os.listdir(path)
        return len(files) == 0
    return not os.path.exists(path)


def check_same_num(dir_list: List[str]) -> bool:
    """Tells if all the folders in the list contain the same number of files."""
    num_files = [len([os.listdir(dir) for dir in dir_list])]
    if len(num_files) == 0:
        return True
    num_file = num_files[0]
    for element in num_files:
        if element != num_file:
            return False
    return True


def run_preprocessing(args: Namespace, logger: Logger) -> None:
    logger.info('Checking training pipeline was run correctly.')
    check_list = [
        os.path.join(args.root_dir, 'gnn_logs', 'gnn_results.csv'),
        os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'confs'),
        os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'normalizers'),
        os.path.join(args.root_dir, 'weights', 'classification', 'gnn', 'weights'),
        os.path.join(args.root_dir, 'data', 'train', 'graphs', 'preds'),
        os.path.join(args.root_dir, 'data', 'validation', 'graphs', 'preds'),
        os.path.join(args.root_dir, 'data', 'test', 'graphs', 'preds'),
    ]
    for element in check_list:
        if check_void(element):
            raise WrongConfigurationError(f'The following folder / file was not found: {element}. Try running the training pipeline again.')
        
    dir_list = [os.path.join(args.root_dir, 'weights', 'classification', 'gnn', dir) for dir in ['confs', 'normalizers', 'weights']]
    if not check_same_num(dir_list):
        raise WrongConfigurationError('Graph weights folders contain different number of files. Try running the training pipeline again')
    
    results = pd.read_csv(os.path.join(args.root_dir, 'gnn_logs', 'gnn_results.csv'))
    if 'F1 Score' in results.columns and args.num_classes != 2:
        raise WrongConfigurationError('You have trained the GNN models for binary problem but are asking to run research for multiclass.')
    elif 'Micro F1' in results.columns and args.num_classes == 2:
        raise WrongConfigurationError('You have trained the GNN models for multiclass problem but are asking to run research for binary.')

    return


def run_models(args: Namespace, logger: Logger) -> None:
    return


def run_evaluation(args: Namespace, logger: Logger) -> None:
    return


def run_visualizations(args: Namespace, logger: Logger) -> None:
    return


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, help='Internal data and weights directory.', default='./.internals/')
    parser.add_argument('--output-dir', type=str, help='Folder where to save all the results.')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes to consider for classification (background not included).')
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

    if args.num_classes != 2:
        raise NotImplementedError('This pipe only supports the binary case.')

    run_preprocessing(args, logger)
    run_models(args, logger)
    run_evaluation(args, logger)
    run_visualizations(args, logger)
