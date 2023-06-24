"""
Script to visualize the result from extract_tensorboard.
It provides several plots about how the hyperparameter affects the metrics.


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
import pandas as pd
import os
import altair as alt


def plot_vis(values: pd.DataFrame, path: str) -> None:
    """
    Creates interesting altair plots.

    :param values: Dataframe with hyperparams in index and metrics in columns.
    :type values: pd.DataFrame
    :param path: Folder where to save plots, must exists already.
    :type path: str
    """
    metrics = values['metric'].unique()
    archs = values['arch'].unique()
    for arch in archs:
        os.makedirs(os.path.join(path, 'batch-norm', arch), exist_ok=True)
        os.makedirs(os.path.join(path, 'dropout', arch), exist_ok=True)
        for metric in metrics:
            # Is batch normalization useful?
            x_var = 'layers'
            y_var = 'max(value)'
            box_var = 'value'
            color_var = 'bn'
            data = values[(values.split == 'validation') & (values.metric == metric) & (values.arch == arch)]
            lines = alt.Chart(data).mark_line().encode(
                x=alt.X(x_var, title='Number of layers'),
                y=alt.Y(y_var, title=metric),
                color=alt.Color(color_var, legend=alt.Legend(title="BatchNorm")),
                tooltip=[x_var, y_var, color_var]
            )
            lines.save(os.path.join(path, 'batch-norm', arch, metric + '-' + arch + '-bn-line.svg'))
            lines.save(os.path.join(path, 'batch-norm', arch, metric + '-' + arch + '-bn-line.png'), scale_factor=2.0)
            # How does dropout affect the result?
            x_var = 'dropout'
            boxplot = alt.Chart(data).mark_boxplot().encode(
                x=alt.X(x_var, title='Percentage of dropout'),
                y=alt.Y(box_var, title=metric),
            )
            boxplot.save(os.path.join(path, 'dropout', arch, metric + '-' + arch + '-drop-box.svg'))
            boxplot.save(os.path.join(path, 'dropout', arch, metric + '-' + arch + '-drop-box.png'), scale_factor=2.0)

    return


def _create_parser():
    parser = argparse.ArgumentParser('Plot tensorboard logs in a better way.')
    parser.add_argument('--input-path', type=str, required=True, help='Path to the .csv from extract_tensorboard.')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to folder where to save plots.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    values = pd.read_csv(args.input_path)
    plot_vis(values, args.output_dir)