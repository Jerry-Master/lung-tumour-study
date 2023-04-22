from argparse import Namespace
from logging import Logger
import os
import pandas as pd
from typing import List


class WrongConfigurationError(Exception):
    pass


class HovernetNotFoundError(Exception):
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


def check_training(args: Namespace, logger: Logger) -> None:
    logger.info('Checking training pipeline was run correctly.')
    check_list = [
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
    graph_check_list = [
        os.path.join(args.root_dir, 'gnn_logs', 'gnn_results.csv'),
        os.path.join(args.root_dir, 'gnn_logs', 'gcn_results.csv'),
        os.path.join(args.root_dir, 'gnn_logs', 'gat_results.csv'),
    ]
    exist_result_file = False
    graph_result_file = None
    for element in graph_check_list:
        if not check_void(element):
            exist_result_file = True
            graph_result_file = element
    if not exist_result_file:
        graph_folder = os.path.join(args.root_dir, 'gnn_logs')
        raise WrongConfigurationError(f'There is not graph result csv file under {graph_folder}')
        
    dir_list = [os.path.join(args.root_dir, 'weights', 'classification', 'gnn', dir) for dir in ['confs', 'normalizers', 'weights']]
    if not check_same_num(dir_list):
        raise WrongConfigurationError('Graph weights folders contain different number of files. Try running the training pipeline again')
    
    results = pd.read_csv(graph_result_file)
    if 'F1 Score' in results.columns and args.num_classes != 2:
        raise WrongConfigurationError('You have trained the GNN models for binary problem but are asking to run research for multiclass.')
    elif 'Micro F1' in results.columns and args.num_classes == 2:
        raise WrongConfigurationError('You have trained the GNN models for multiclass problem but are asking to run research for binary.')
        