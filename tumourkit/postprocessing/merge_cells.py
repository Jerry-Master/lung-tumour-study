"""
Merge broken cells using a morphological algorithm
to detect touching frontiers.

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
import os
from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd
import skimage
from ..utils.preprocessing import (
    parse_path, create_dir, get_names, save_pngcsv, read_labels
)
import argparse
from tqdm import tqdm


MAX_CELLS = 1500


def create_id_map() -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[int, Tuple[int, int]]]:
    """
    Creates an index mapping function and its inverse mapping based on a unique difference per pair.
    The index mapping function `f` is defined as `f(x) = x^5`, ensuring that the difference between function values for each pair of indices is unique.
    Extra information: https://math.stackexchange.com/questions/4565014/injectivity-of-given-integer-function/4567383#4567383

    :return: A tuple containing the index mapping function `f` and the inverse mapping of the differences.
    :rtype: Tuple[Callable[[np.ndarray], np.ndarray], Dict[int, Tuple[int, int]]]
    """
    def f(x: int) -> int:
        return x**5
    sq_dif = []
    inv_diff_mapping = {}
    for i in range(MAX_CELLS):
        for j in range(i + 1, MAX_CELLS):
            sq_dif.append(f(j) - f(i))
            inv_diff_mapping[f(j) - f(i)] = [j, i]
    assert (len(sq_dif) - len(set(sq_dif))) == 0, 'Defined function is not injective.'

    vec_mapping = np.vectorize(f)
    return vec_mapping, inv_diff_mapping


def get_gradient(png: np.ndarray) -> np.ndarray:
    """"
    Computes the morphological gradient of a binary image.

    :param png: The binary image for which the gradient is computed.
    :type png: np.ndarray

    :return: The gradient image.
    :rtype: np.ndarray
    """
    mask = png.copy()
    mask[mask > 0] = 1
    dilation = skimage.morphology.dilation(png, np.ones((3, 3)))
    grad = dilation - png
    grad_bkgr = grad * mask
    return grad_bkgr


def merge_cells(
        png: np.ndarray,
        vec_mapping: Callable[[np.ndarray], np.ndarray],
        inv_diff_mapping: Dict[int, Tuple[int, int]]
        ) -> np.ndarray:
    """
    Merges cells in a binary image based on the shared frontier length.

    :param png: The binary image containing cells.
    :type png: np.ndarray
    :param vec_mapping: The vectorized mapping function to map indices.
    :type vec_mapping: Callable[[np.ndarray], np.ndarray]
    :param inv_diff_mapping: The inverse mapping of differences between indices.
    :type inv_diff_mapping: Dict[int, Tuple[int, int]]

    :return: The merged binary image.
    :rtype: np.ndarray
    """
    png_cp = png.copy()
    png_cp = vec_mapping(png_cp)  # Indices mapping
    grad_bkgr = get_gradient(png_cp)  # Morphological gradient
    # Select pairs with long frontier
    unique, counts = np.unique(grad_bkgr, return_counts=True)
    component = [-1 for _ in range(MAX_CELLS)]
    for el, freq in zip(unique, counts):
        if el > 0 and freq > 13:  # More than 13 points in frontier
            pair = inv_diff_mapping[el]
            # Get component of each index
            if component[pair[0]] != -1:
                pair[0] = component[pair[0]]
            if component[pair[1]] != -1:
                pair[1] = component[pair[1]]
            M = min(*pair)  # Merge operation is to set both indices to the minimum
            # Update component of each index
            for k, el in enumerate(component):
                if el == pair[0] or el == pair[1]:
                    component[k] = M
            component[pair[0]] = M
            component[pair[1]] = M
            # Map indices to new component
            png[png == pair[0]] = M
            png[png == pair[1]] = M
    return png


def remove_lost_ids(png: np.ndarray, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Removes identifiers in csv that are not present in the png.

    :param png: The binary image containing cell identifiers.
    :type png: np.ndarray
    :param csv: The DataFrame containing cell information with identifiers.
    :type csv: pd.DataFrame

    :return: The updated DataFrame with removed identifiers.
    :rtype: pd.DataFrame
    """
    next_ids = set(np.unique(png))
    curr_ids = set(csv.id)
    assert next_ids.difference(curr_ids) == set([0]), "Indices doesn't match"
    remove_ids = curr_ids.difference(next_ids)
    csv.set_index('id', inplace=True)
    csv.drop(remove_ids, axis='index', inplace=True)
    csv.reset_index(inplace=True)
    return csv


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv files.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    output_path = parse_path(args.output_path)
    png_out_path = os.path.join(output_path, 'postPNG')
    csv_out_path = os.path.join(output_path, 'postCSV')
    create_dir(output_path)
    create_dir(png_out_path)
    create_dir(csv_out_path)

    names = get_names(png_dir, '.GT_cells.png')
    vec_mapping, inv_diff_mapping = create_id_map()
    for k, name in tqdm(enumerate(names)):
        png, csv = read_labels(name, png_dir, csv_dir)
        if png is None or csv is None:
            continue
        assert (np.max(csv.id) < MAX_CELLS), "Exceeded maximum number of cells."
        png = merge_cells(png, vec_mapping, inv_diff_mapping)
        csv = remove_lost_ids(png, csv)
        save_pngcsv(png, csv, png_out_path, csv_out_path, name)
