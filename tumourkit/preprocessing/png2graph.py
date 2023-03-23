"""
Computes a graph representation from the images and pngcsv labels.

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
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
from ..utils.preprocessing import read_png, save_graph, parse_path, create_dir, get_names, get_mask, apply_mask, add_node, extract_features, read_image





def png2graph(img: np.ndarray, png: np.ndarray) -> pd.DataFrame:
    """
    Given an original image and a segmentation mask in PNG format, this function extracts the nodes and their attributes
    and returns them in a pandas DataFrame. The following attributes are computed for each node:
    
    * X: The X-coordinate of the centroid.
    * Y: The Y-coordinate of the centroid.
    * Area: The area of the cell.
    * Perimeter: The perimeter of the cell.
    * Variance: The variance of the grayscale values inside the cell.
    * Histogram: The normalized histogram of the grayscale values inside the cell (5 bins).
    
    :param img: The original image as a numpy array.
    :type img: np.ndarray
    :param png: The segmentation mask in PNG format as a numpy array.
    :type png: np.ndarray
    :return: A pandas DataFrame containing the extracted nodes and their attributes.
    :rtype: pd.DataFrame
    """
    graph = {}
    for idx in np.unique(png):
        if idx == 0:
            continue
        mask = get_mask(png, idx)
        msk_img, msk, X, Y  = apply_mask(img, mask)
        try:
            feats = extract_features(msk_img, msk)
        except:
            feats = extract_features(msk_img, msk, debug=True)
        if len(feats) > 0:
            feats['id'] = idx
            feats['X'] = X
            feats['Y'] = Y
        add_node(graph, feats)
    return pd.DataFrame(graph)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--orig-dir', type=str, required=True,
                        help='Path to original images.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    parser.add_argument('--num-workers', type=int, default=1)
    return parser


def main_subthread(
        name: str,
        png_dir: str,
        orig_dir: str,
        output_path: str,
        pbar: tqdm,
        )-> None:
    """
    A wrapper function to use multiprocessing.

    :param name: The name of the image file (without extension).
    :type name: str
    :param png_dir: The directory path containing PNG segmentation masks.
    :type png_dir: str
    :param orig_dir: The directory path containing original RGB images.
    :type orig_dir: str
    :param output_path: The directory path to save output files.
    :type output_path: str
    :param pbar: A progress bar object to track the progress of the computation.
    :type pbar: tqdm.tqdm
    """
    try:
        png = read_png(name, png_dir)
        img = read_image(name, orig_dir)
        graph = png2graph(img, png)
        save_graph(graph, os.path.join(output_path, name+'.nodes.csv'))
    except Exception as e:
        logging.warning(e)
        logging.warning('Failed at:', name)
    finally:
        pbar.update(1)


def main_with_args(args):
    png_dir = parse_path(args.png_dir)
    orig_dir = parse_path(args.orig_dir)
    output_path = parse_path(args.output_path)
    create_dir(output_path)

    names = get_names(png_dir, '.GT_cells.png')
    pbar = tqdm(total=len(names))
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for name in names:
                executor.submit(main_subthread, name, png_dir, orig_dir, output_path, pbar) 
    else:
        for name in names:
            main_subthread(name, png_dir, orig_dir, output_path, pbar)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
