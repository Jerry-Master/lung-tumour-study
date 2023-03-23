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
from ..utils.preprocessing import get_mask, read_labels, parse_path, create_dir, save_graph, get_names, apply_mask, extract_features, add_node, read_image


def pngcsv2graph(img: np.ndarray, png: np.ndarray, csv: pd.DataFrame) -> pd.DataFrame:
    """
    Given original image and pngcsv labels, 
    returns nodes with extracted attributes in a DataFrame.
    Current attributes are:
        - X, Y of centroid
        - Area
        - Perimeter
        - Variance
        - Regularity (not yet)
        - Histogram (5 bins)
    """
    graph = {}
    for idx, cls in csv.itertuples(index=False, name=None):
        mask = get_mask(png, idx)
        msk_img, msk, X, Y  = apply_mask(img, mask)
        try:
            feats = extract_features(msk_img, msk)
        except:
            feats = extract_features(msk_img, msk, debug=True)
        if len(feats) > 0:
            feats['class'] = cls
            feats['id'] = idx
            feats['X'] = X
            feats['Y'] = Y
        add_node(graph, feats)
    return pd.DataFrame(graph)


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to png files.')
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to csv files.')
    parser.add_argument('--orig-dir', type=str, required=True,
                        help='Path to original images.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save files.')
    parser.add_argument('--num-workers', type=int, default=1)
    return parser


def main_subthread(
        name: str,
        png_dir: str,
        csv_dir: str,
        orig_dir: str,
        output_path: str,
        pbar: tqdm,
        )-> None:
    """
    Wrapper to use multiprocessing
    """
    try:
        png, csv = read_labels(name, png_dir, csv_dir)
        img = read_image(name, orig_dir)
        graph = pngcsv2graph(img, png, csv)
        save_graph(graph, os.path.join(output_path, name+'.nodes.csv'))
    except Exception as e:
        logging.warning(e)
        logging.warning('Failed at:', name)
    finally:
        pbar.update(1)


def main_with_args(args):
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    orig_dir = parse_path(args.orig_dir)
    output_path = parse_path(args.output_path)
    create_dir(output_path)

    names = get_names(png_dir, '.GT_cells.png')
    pbar = tqdm(total=len(names))
    if args.num_workers > 0:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for name in names:
                executor.submit(main_subthread, name, png_dir, csv_dir, orig_dir, output_path, pbar) 
    else:
        for name in names:
            main_subthread(name, png_dir, csv_dir, orig_dir, output_path, pbar)


def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
