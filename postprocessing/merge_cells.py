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
from typing import Callable, Dict, Tuple
import numpy as np
import pandas as pd
import skimage
import os
import sys

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PKG_DIR)

from utils.preprocessing import (
    parse_path, create_dir, get_names, save_pngcsv, read_labels
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--png-dir', type=str, required=True,
                    help='Path to png files.')
parser.add_argument('--csv-dir', type=str, required=True,
                    help='Path to csv files.')
parser.add_argument('--output-path', type=str, required=True,
                    help='Path to save files.')

MAX_CELLS = 1500

def create_id_map() -> Tuple[Callable[[np.ndarray], np.ndarray], Dict[int, Tuple[int,int]]]:
    """
    Map index to some function so that difference is unique per pair.
    Also returns the inverse mapping of the differences and the 
    inverse mapping of the function itself.
    Extra information: https://math.stackexchange.com/questions/4565014/injectivity-of-given-integer-function/4567383#4567383
    """
    f = lambda x: x**5
    sq_dif = []
    inv_diff_mapping = {}
    for i in range(MAX_CELLS):
        for j in range(i+1,MAX_CELLS):
            sq_dif.append(f(j)-f(i))
            inv_diff_mapping[f(j)-f(i)] = [j,i]
    assert (len(sq_dif) - len(set(sq_dif)))==0, 'Defined function is not injective.'

    vec_mapping = np.vectorize(f)
    return vec_mapping, inv_diff_mapping

def get_gradient(png: np.ndarray) -> np.ndarray:
    """
    Apply a dilation, subtract the image and remove pixels in background.
    """
    mask = png.copy()
    mask[mask>0] = 1
    dilation = skimage.morphology.dilation(png, np.ones((3,3)))
    grad = dilation - png
    grad_bkgr = grad * mask
    return grad_bkgr

def merge_cells(
    png: np.ndarray, 
    vec_mapping: Callable[[np.ndarray], np.ndarray], 
    inv_diff_mapping: Dict[int, Tuple[int,int]]
    ) -> np.ndarray:
    """
    Merges all the cells that share a frontier of more than 13 pixels.
    """
    png_cp = png.copy()
    png_cp = vec_mapping(png_cp) # Indices mapping
    grad_bkgr = get_gradient(png_cp) # Morphological gradient
    # Select pairs with long frontier
    unique, counts = np.unique(grad_bkgr, return_counts=True)
    component = [-1 for _ in range(MAX_CELLS)]
    for el, freq in zip(unique, counts):
        if el > 0 and freq > 13: # More than 13 points in frontier
            pair = inv_diff_mapping[el]
            # Get component of each index
            if component[pair[0]] != -1:
                pair[0] = component[pair[0]]
            if component[pair[1]] != -1:
                pair[1] = component[pair[1]]
            M = min(*pair) # Merge operation is to set both indices to the minimum
            # Update component of each index
            for k, el in enumerate(component):
                if el == pair[0] or el == pair[1]:
                    component[k] = M
            component[pair[0]] = M
            component[pair[1]] = M
            # Map indices to new component
            png[png==pair[0]] = M
            png[png==pair[1]] = M
    return png

def remove_lost_ids(png: np.ndarray, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Removes identifiers in csv that are not in png.
    """
    next_ids = set(np.unique(png))
    curr_ids = set(csv.id)
    assert next_ids.difference(curr_ids) == set([0]), "Indices doesn't match"
    remove_ids = curr_ids.difference(next_ids)
    csv.set_index('id', inplace=True)
    csv.drop(remove_ids, axis='index', inplace=True)
    csv.reset_index(inplace=True)
    return csv

if __name__=='__main__':
    args = parser.parse_args()
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)
    OUTPUT_PATH = parse_path(args.output_path)
    PNG_OUT_PATH = OUTPUT_PATH + '/postPNG/'
    CSV_OUT_PATH = OUTPUT_PATH + '/postCSV/'
    create_dir(OUTPUT_PATH)
    create_dir(PNG_OUT_PATH)
    create_dir(CSV_OUT_PATH)

    names = get_names(PNG_DIR, '.GT_cells.png')
    vec_mapping, inv_diff_mapping = create_id_map()
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        png, csv = read_labels(name, PNG_DIR, CSV_DIR)
        if png is None or csv is None: continue
        assert(np.max(csv.id) < MAX_CELLS), "Exceeded maximum number of cells."
        png = merge_cells(png, vec_mapping, inv_diff_mapping)
        csv = remove_lost_ids(png, csv)
        save_pngcsv(png, csv, PNG_OUT_PATH, CSV_OUT_PATH, name)
