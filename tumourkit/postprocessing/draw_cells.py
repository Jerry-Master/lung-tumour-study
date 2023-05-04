"""
Draws the cells into an image.
Input format: PNG / CSV
Output format: PNG

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
from tqdm import tqdm
import os
import cv2
import numpy as np
import pandas as pd
import json
from typing import Dict, Tuple, List
from ..utils.preprocessing import get_names, read_labels


def draw_cells(orig: np.ndarray, png: np.ndarray, csv: pd.DataFrame, type_info: Dict[str, Tuple[str, List[int]]]) -> np.ndarray:
    blend = orig.copy()
    for i, (idx, cell_label) in csv.iterrows():
        blend[png==idx] = 0.3 * np.array(type_info[str(cell_label)][1]) \
                       + 0.7 * blend[png==idx]
    return blend


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig-dir', type=str, help='Path to base images folder. Must be in .png format.')
    parser.add_argument('--png-dir', type=str, help='Path to folder with png of the labels.')
    parser.add_argument('--csv-dir', type=str, help='Path to folder with csv of the labels.')
    parser.add_argument('--output-dir', type=str, help='Path to folder where to save results.')
    parser.add_argument('--type-info', type=str, help='Path to type_info.json.')
    return parser


def main_with_args(args: Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    names = get_names(args.orig_dir, '.png')
    for name in tqdm(names):
        orig = cv2.imread(os.path.join(args.orig_dir, name + '.png'), cv2.IMREAD_COLOR)[:, :, ::-1]
        png, csv = read_labels(name, args.png_dir, args.csv_dir)
        with open(args.type_info, 'r') as f:
            type_info = json.load(f)
        out = draw_cells(orig, png, csv, type_info)
        cv2.imwrite(os.path.join(args.output_dir, name + '.overlay.png'), out[:, :, ::-1])
    return


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)