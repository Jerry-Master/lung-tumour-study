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
    """
    Checks if a directory is void (empty) or if a file exists..

    :param path: The path to the directory or file.
    :type path: str

    :return: True if the directory is void (empty) or the file exists, False otherwise.
    :rtype: bool
    """
    if os.path.isdir(path):
        files = os.listdir(path)
        return len(files) == 0
    return not os.path.exists(path)


def check_same_num(dir_list: List[str]) -> bool:
    """
    Checks if all the folders in the list contain the same number of files.

    This function takes a list of folder paths and checks if all the folders contain the same number of files. It counts
    the number of files in each folder and compares them. If the number of files is the same for all folders, the
    function returns True. Otherwise, it returns False.

    :param dir_list: The list of folder paths.
    :type dir_list: List[str]

    :return: True if all folders contain the same number of files, False otherwise.
    :rtype: bool
    """
    num_files = [len([os.listdir(dir) for dir in dir_list])]
    if len(num_files) == 0:
        return True
    num_file = num_files[0]
    for element in num_files:
        if element != num_file:
            return False
    return True


def check_training(args: Namespace, logger: Logger) -> None:
    """
    Checks if the training pipeline was run correctly.

    This function checks if the necessary folders and files from the training pipeline exist. It verifies the presence
    of specific folders and files in the expected locations. If any required folder or file is missing, or if there are
    inconsistencies in the training results, an exception is raised.

    :param args: The command-line arguments.
    :type args: Namespace

    :param logger: The logger object for logging messages.
    :type logger: Logger

    :raises WrongConfigurationError: If any required folder or file is missing, or if there are inconsistencies in the training results.
    """
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
