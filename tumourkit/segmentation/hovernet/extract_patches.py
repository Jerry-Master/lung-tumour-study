"""
Preprocess data into hovernet input format.

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
import cv2
import numpy as np
import pandas as pd
import random
from typing import Optional, List, Tuple, Set, Callable
import argparse
from tqdm import tqdm
import os
from tumourkit.utils.preprocessing import parse_path, create_dir, get_names


def to4dim(orig_dir: str, png_dir: str, csv_dir: str, name: str) -> np.ndarray:
    """
    Generates 3D numpy array where dimensions are:
    Height x Width x Channels
    Channels are: R, G, B, Ground Truth (segmentation)
    csv_dir is only needed for compatibility, it is not used.
    """
    gt_path = png_dir + name + '.GT_cells.png'
    img_path = orig_dir + name + '.png'

    gt = cv2.imread(gt_path, -1)
    gt = np.reshape(gt, (*gt.shape, 1))
    img = cv2.imread(img_path, -1)[:,:,::-1]
    return np.concatenate([img, gt], axis=-1)


def generate_labels(
        png_dir: str, csv_dir: str, name: str,
        return_segm: Optional[bool]=True) -> np.ndarray:
    """
    Generates class GT. Pixels have values:
    0-> Background
    1-> Non-tumour
    2-> Tumour
    Return shape: Height x Width x 1
    """
    gt_path = png_dir + name + '.GT_cells.png'
    gt = cv2.imread(gt_path, -1)
    gt = np.reshape(gt, (*gt.shape, 1))
    
    gt_labels_path = csv_dir + name + '.class.csv'
    parser = pd.read_csv(gt_labels_path, header=None)
    parser = dict(zip(parser.iloc[:,0], parser.iloc[:,1]))

    def aux(x):
        if x>1:
            return parser[x]
        return x
    vec_parser = np.vectorize(aux)
    
    if return_segm:
        return gt, vec_parser(gt)
    return vec_parser(gt)


def to5dim(orig_dir: str, png_dir: str, csv_dir: str, name: str) -> np.ndarray:
    """
    Generates 3D numpy array where dimensions are:
    Height x Width x Channels
    Channels are: R, G, B, GT (segmentation), GT (class)
    """
    gt, labels = generate_labels(png_dir, csv_dir, name)

    img_path = orig_dir + name + '.png'
    img = cv2.imread(img_path, -1)[:,:,::-1]

    return np.concatenate([img, gt, labels], axis=-1)


def create_splits(
        img_list: List[str], out_dir: str
        ) -> Tuple[Tuple[Set[str], Set[str], Set[str]], List[str]]:
    """
    img_list: List of image names to split.
    out_dir: Parent directory to create subfolders train, validation and test.

    Returns sets of strings as splits, and creates subfolders
    for saving data on them later.
    """
    split_dirs = [out_dir + x for x in ['train/', 'validation/', 'test/']]
    for dir_ in split_dirs:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
    n = len(img_list)
    n_train = int(n * 0.7)
    n_val = int(n * 0.25)
    n_test = n - n_train - n_val
    print('Train samples:', n_train)
    print('Validation samples:', n_val)
    print('Test samples:', n_test)

    aux = set(img_list)
    train_split = set(random.sample(aux, n_train))
    aux = aux.difference(train_split)
    val_split = set(random.sample(aux, n_val))
    test_split = aux.difference(val_split)

    splits = [train_split, val_split, test_split]
    return splits, split_dirs


def save_batch(
        name: str, save_dir: str,
        processing_function: Callable[[str, str, str, str], np.ndarray],
        orig_dir: str, png_dir: str, csv_dir: str,
        shape: str) -> None:
    """
    Loads data from paths and saves the joined array in npy format.
    Three output shapes are supported:
    'full': The whole image is saved.
    '518': Image is pieced into 518x518 patches.
    '270': Image is pieced into 270x270 patches.
    """
    arr = processing_function(orig_dir, png_dir, csv_dir, name)
    if shape == 'full':
        np.save(save_dir + name + '.npy', arr)
    elif shape == '270':
        for i in [0, 252, 503, 754]:
            for j in [0, 252, 503, 754]:
                crop = arr[i:i+270, j:j+270,]
                np.save(save_dir + name + '(' + str(i) + ',' + str(j) + ').npy', crop)
    elif shape == '518':
        for i in [0, 506]:
            for j in [0, 506]:
                crop = arr[i:i+518, j:j+518,]
                np.save(save_dir + name + '(' + str(i) + ',' + str(j) + ').npy', crop)
    else:
        assert False, 'Something wrong happened.'


def save_npy(
        img_list: str, out_dir: str,
        processing_function: Callable[[str, str, str, str], np.ndarray],
        split: bool, orig_dir: str, png_dir: str, csv_dir: str,
        shape: bool) -> None:
    """
    Saves data in the npy required format.
    Internally calls save_batch for each split if wanted.
    """
    if split:
        splits, split_dirs = create_splits(img_list, out_dir)
        for split_, split_dir in zip(splits, split_dirs):
            print('Split path: ', split_dir)
            for name in tqdm(split_):
                save_batch(name, split_dir, processing_function, orig_dir, png_dir, csv_dir, shape)

    else:
        for name in tqdm(img_list):
            save_batch(name, out_dir, processing_function, orig_dir, png_dir, csv_dir, shape)


def save_imgs(
        orig_dir: str, png_dir: str, csv_dir: str,
        name: str, out_dir: str) -> None:
    """
    Saves RGB image, GT segmentation image and GT class image
    in png format.
    """
    arr = to5dim(orig_dir, png_dir, csv_dir, name)
    img = np.array(arr[:,:,:3][:,:,::-1], dtype=np.uint8)
    gt = arr[:,:,3]
    labels = arr[:,:,4]

    save_dir = out_dir+'samples/'
    create_dir(save_dir)
    cv2.imwrite(save_dir + name + '.png', img)
    cv2.imwrite(save_dir + name + '.GT_cells.png', gt)
    cv2.imwrite(save_dir + name + '.class.png', labels)


def _create_parser():
    parser = argparse.ArgumentParser(description='Generates .npy files for HoVer-net from the current data images in .png and .csv.')
    parser.add_argument('--orig-dir', type=str, required=True, help='Directory to read the original png files. Must end with a slash /.')
    parser.add_argument('--png-dir', type=str, required=True, help='Directory to read the label png files. Must end with a slash /.')
    parser.add_argument('--csv-dir', type=str, required=True, help='Directory to read the label csv files. Must end with a slash /.')
    parser.add_argument('--out-dir', type=str, required=True, help='Directory to save the .npy files. Must end with a slash /.')
    parser.add_argument('--save-example', action='store_true', help="Save one sample as images to visualize.")
    parser.add_argument('--use-labels', action='store_true', help="Whether to generate the label image or only segmentation.")
    parser.add_argument('--split', action='store_true', help="Whether to split into train and validation folders.")
    parser.add_argument('--shape', type=str, choices=['270', '518', 'full'], required=True, help='Height and width of the saved arrays. If less than full, patching is applied.')
    return parser


def main_with_args(args):
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)
    orig_dir = parse_path(args.orig_dir)
    out_dir = parse_path(args.out_dir)
    create_dir(out_dir)

    img_list = get_names(png_dir, '.GT_cells.png')

    if args.save_example:
        name = random.sample(img_list, 1)[0]
        save_imgs(orig_dir, png_dir, csv_dir, name, out_dir)

    if args.use_labels:
        save_npy(img_list, out_dir, to5dim, args.split, orig_dir, png_dir, csv_dir, args.shape)
    else:
        save_npy(img_list, out_dir, to4dim, args.split, orig_dir, png_dir, csv_dir, args.shape)

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)

