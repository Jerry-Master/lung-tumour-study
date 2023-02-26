"""
Converts our png <-> csv format into QuPath geojson format.

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
from typing import List, Dict, Any
import argparse
import pandas as pd
import cv2
import numpy as np
import geojson
from tqdm import tqdm
from ..utils.preprocessing import (
    get_names, create_dir, parse_path,
    format_contour, create_geojson, read_labels
)


def save_geojson(gson: List[Dict[str, Any]], name: str, path: str) -> None:
    """
    Save geojson to file path + name.
    """
    with open(path + name + '.geojson', 'w') as f:
        geojson.dump(gson, f)


def create_mask(png: np.ndarray, csv: pd.DataFrame, label: int) -> np.ndarray:
    """
    Returns the image with only the pixels of the class given in label.
    The pixel values are truncated to uint8.
    """
    mask = png.copy()
    for i, (idx, cell_label) in csv.iterrows():
        if cell_label != label:
            mask[mask==idx] = 0
    return np.array(mask, dtype=np.uint8)


def pngcsv2features(png: np.ndarray, csv: pd.DataFrame, label: int) -> List[Dict[str, Any]]:
    """
    Computes geojson features of contours of a given class.
    """
    mask = create_mask(png, csv, label)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c), label) for c in contours])
    return create_geojson(contours)


def pngcsv2geojson(png: np.ndarray, csv: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes geojson as list of features representing contours.
    Contours are approximated by method cv2.CHAIN_APPROX_SIMPLE.
    """
    total_contours = []
    for _, (idx, cell_label) in csv.iterrows():
        mask = png.copy()
        mask[mask != idx] = 0
        mask[mask == idx] = 1
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c), cell_label) for c in contours])
        total_contours.extend(create_geojson(contours))
        del mask
    return total_contours


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv files.')
    parser.add_argument('--gson-dir', type=str, required=True,
                        help='Path to save files.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)
    OUTPUT_PATH = parse_path(args.gson_dir)
    create_dir(OUTPUT_PATH)

    names = get_names(PNG_DIR, '.GT_cells.png')
    for name in tqdm(names):
        png, csv = read_labels(name, PNG_DIR, CSV_DIR)
        if png is None or png.max() == 0:
            continue
        gson = pngcsv2geojson(png, csv)
        save_geojson(gson, name, OUTPUT_PATH)
