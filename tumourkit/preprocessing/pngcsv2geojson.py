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
from typing import List, Dict, Any, Optional
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
    Save a list of geojson features to a file with the given name at the
    specified path.

    :param gson: A list of geojson features to save.
    :type gson: List[Dict[str, Any]]
    :param name: The name of the file to save.
    :type name: str
    :param path: The path to save the file at.
    :type path: str
    """
    with open(path + name + '.geojson', 'w') as f:
        geojson.dump(gson, f)


def create_mask(png: np.ndarray, csv: pd.DataFrame, label: int) -> np.ndarray:
    """
    Returns an image with only the pixels of the class given in the label.
    
    :param png: Segmentation mask with indices as pixel values.
    :type png: np.ndarray
    :param csv: DataFrame containing the class labels for each cell in the image.
    :type csv: pd.DataFrame
    :param label: Class label to extract from the image.
    :type label: int
    :return: Binary mask where pixels of the specified class have value 1 and all others have value 0.
    :rtype: np.ndarray
    """
    mask = png.copy()
    for i, (idx, cell_label) in csv.iterrows():
        if cell_label != label:
            mask[mask==idx] = 0
    return np.array(mask, dtype=np.uint8)


def pngcsv2features(png: np.ndarray, csv: pd.DataFrame, label: int, num_classes: Optional[int] = 2) -> List[Dict[str, Any]]:
    """
    Computes geojson features of contours of a given class.

    :param png: Segmentation mask with indices as pixel values.
    :type png: np.ndarray
    :param csv: DataFrame with the classes of the cells in the image.
    :type csv: pd.DataFrame
    :param label: The label of the class of cells to extract features for.
    :type label: int
    :param num_classes: The number of classes in the segmentation mask, defaults to 2.
    :type num_classes: Optional[int], optional
    :return: A list of dictionaries with the geojson features of the contours of the specified class of cells.
    :rtype: List[Dict[str, Any]]
    """
    mask = create_mask(png, csv, label)
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c), label) for c in contours])
    return create_geojson(contours, num_classes)


def pngcsv2geojson(png: np.ndarray, csv: pd.DataFrame, num_classes: Optional[int] = 2) -> List[Dict[str, Any]]:
    """
    Computes the GeoJSON representation of contours in the given PNG image based on the provided CSV labels.

    :param png: A NumPy array representing the PNG image.
    :type png: np.ndarray
    :param csv: A Pandas DataFrame with the CSV labels of each cell.
    :type csv: pd.DataFrame
    :param num_classes: The number of classes. Default is 2.
    :type num_classes: Optional[int]
    :return: A list of features representing contours.
    :rtype: List[Dict[str, Any]]
    """
    total_contours = []
    for _, (idx, cell_label) in csv.iterrows():
        mask = png.copy()
        mask[mask != idx] = 0
        mask[mask == idx] = 1
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda x: len(x[0]) >= 3, [(format_contour(c), cell_label) for c in contours])
        total_contours.extend(create_geojson(contours, num_classes))
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
    parser.add_argument('--num-classes', type=int, default=2)
    return parser


def main_with_args(args):
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    output_path = parse_path(args.gson_dir)
    create_dir(output_path)

    names = get_names(png_dir, '.GT_cells.png')
    for name in tqdm(names):
        png, csv = read_labels(name, png_dir, csv_dir)
        if png is None or png.max() == 0:
            continue
        gson = pngcsv2geojson(png, csv, args.num_classes)
        save_geojson(gson, name, output_path)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
