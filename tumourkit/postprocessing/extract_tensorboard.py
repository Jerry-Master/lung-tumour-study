"""
Script to retrieve the data from tensorboard logs.
It computes the maximum or minimum per each time series and saves it in a file.
It is intended to be used to process the gnn results.


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
import os
import numpy as np
import pandas as pd
from typing import Dict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_scalar_from_tb(path: str) -> Dict[str, float]:
    """
    Extracts the best values of all the time series.

    :param path: Path to tensorboard file.
    :type path: str

    :return: Dictionary with metrics as keys and maximums as values.
    :rtype: Dict[str, float]
    """
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    res = {}
    for scalar_key in event_acc.scalars.Keys():
        scalar_values = list(map(lambda x: x.value, event_acc.Scalars(scalar_key)))
        if 'ECE' in scalar_key or 'Perc' in scalar_key:
            scalar_best = np.min(scalar_values)
        else:
            scalar_best = np.max(scalar_values)
        assert scalar_key not in res, 'More than one line per metric is not supported.'
        res[scalar_key] = scalar_best
    return res


def extract_all_scalars(path: str) -> Dict[str, Dict[str, float]]:
    """
    Extracts all the best values of all metrics and files.
    Tensorboard files must be under a folder.
    Folders' names will be used as dictionary keys.

    :param path: Path to folder containing folders containing tensorboard files.
    :type path: str

    :return: Dictionary with folders' names as keys, and another dictionary as values. That dictionary contains metrics and keys and maximums as values.
    :rtype: Dict[str, Dict[str, float]]
    """
    res = {}
    for tb_input in filter(
            lambda x: os.path.isdir(os.path.join(path, x)),
            os.listdir(path)
            ):
        tb_files = os.listdir(os.path.join(path, tb_input))
        assert len(tb_files) == 1, 'There must be only one tensorboard file per folder.'
        tb_file = os.path.join(path, tb_input, tb_files[0])
        scalar_values = extract_scalar_from_tb(tb_file)
        res[tb_input] = scalar_values
    return res


def to_long_format(values: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the initial dataframe into a more useful one that is in long format.
    New columns: arch, bn, dropout, layers, metric, value

    :param values: The initial dataframe from extract_tensorboard.
    :type values: pd.DataFrame

    :return: DataFrame in long format.
    :rtype: pd.DataFrame
    """
    data = values.copy()
    data = data.reset_index()
    data[['arch', 'layers', 'dropout', 'bn']] = data['index'].str.split('_', expand=True)
    data['layers'] = pd.to_numeric(data['layers'])
    data['dropout'] = pd.to_numeric(data['dropout'])
    data['bn'] = data['bn'].apply(lambda x: {'bn': 'Yes', 'None': 'No'}.get(x, 'Unknown'))
    data = data.drop(columns=['index'])
    data = data.melt(id_vars=['arch', 'layers', 'dropout', 'bn'], var_name='metric')
    data[['metric', 'split']] = data['metric'].str.split('/', expand=True)
    return data


def write_to_file(path: str, values: Dict[str, Dict[str, float]]) -> None:
    """
    Saves output to file. Dictionary keys are used as index.
    Inner dictionary keys are used as columns.

    :param path: Path to save file, without extension, .csv appended.
    :type path: str
    :param values: Values to save.
    :type values: Dict[str, Dict[str, float]]
    """
    values_df = pd.DataFrame(values).transpose()
    values_df = to_long_format(values_df)
    values_df.to_csv(path + '.csv', index=False)


def _create_parser():
    parser = argparse.ArgumentParser(description='Process tensorboard gnn logs.')
    parser.add_argument('--logs-dir', required=True, help='Path to gnn logs dir with tensorboard files.')
    parser.add_argument('--output-path', required=True, help='Output file path, must not have extension.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    metrics = extract_all_scalars(args.logs_dir)
    write_to_file(args.output_path, metrics)
