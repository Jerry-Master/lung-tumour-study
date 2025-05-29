"""
Script to read the tensorboard logs and extract the last ten
validation points of each configuration to select the best.


Copyright (C) 2025  Jose Pérez Cano

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
import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main_with_args(args: Namespace, logger: Logger) -> None:
    rows = [("configuration", "metric", "value")]
    # Iterate through subdirectories (one per line)
    for line_name in os.listdir(args.input_dir):
        log_dir = os.path.join(args.input_dir, line_name)
        if not os.path.isdir(log_dir):
            continue
        # Load TensorBoard logs
        try:
            ea = EventAccumulator(log_dir)
            ea.Reload()
        except Exception as e:
            logger.warning(f"Skipping {log_dir}: {e}")
            continue
        # Filter scalar tags that contain "validation"
        for tag in ea.Tags().get("scalars", []):
            if "validation" not in tag:
                continue
            events = ea.Scalars(tag)
            if len(events) < 10:
                logger.warning(f"Not enough data points for {line_name} - {tag}")
                continue
            # Get the best value
            value = events[-args.patience].value
            rows.append((line_name, tag, value))
    # Save in CSV
    with open(args.output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _create_parser():
    parser = argparse.ArgumentParser('Gets the best validation value of each configuration.')
    parser.add_argument('--input-dir', type=str, required=True, help='Path to the tensorboard directory, it should contain one subfolder with one logfile per configuration.')
    parser.add_argument('--output-path', type=str, required=True, help='Path where to save csv.')
    parser.add_argument('--patience', type=int, default=10, help='Number of steps that were checked for improvement in validation.')
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

    main_with_args(args, logger)