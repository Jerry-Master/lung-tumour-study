"""
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
import pytest
import os
import shutil
import numpy as np

from tumourkit.utils.preprocessing import parse_path, get_names
from tumourkit.segmentation.hovernet.extract_patches import save_npy, to4dim, to5dim, create_dir

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PNGCSV_DIR = parse_path(TEST_DIR) + 'pngcsv/'
ORIG_DIR = parse_path(TEST_DIR) + 'tiles/'
OUT_DIR = parse_path(TEST_DIR) + 'npy_files/'

@pytest.mark.parametrize('shape', ['518', '270'])
def test_hovernet_patches_5dim(shape):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    create_dir(OUT_DIR)
    img_list = get_names(PNGCSV_DIR, '.GT_cells.png')[:5]
    save_npy(
        img_list, OUT_DIR, to5dim, False,
        ORIG_DIR, PNGCSV_DIR, PNGCSV_DIR,
        shape)
    if shape == '518':
        check_shape = (518,518,5)
    elif shape == '270':
        check_shape = (270,270,5)
    npy_list = get_names(OUT_DIR, '.npy')
    for file in npy_list:
        arr = np.load(OUT_DIR + file + '.npy')
        if arr.shape != check_shape:
            assert False
    assert True
    shutil.rmtree(OUT_DIR)

@pytest.mark.parametrize('shape', ['518', '270'])
def test_hovernet_patches_4dim(shape):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    create_dir(OUT_DIR)
    img_list = get_names(PNGCSV_DIR, '.GT_cells.png')[:5]
    save_npy(
        img_list, OUT_DIR, to4dim, False,
        ORIG_DIR, PNGCSV_DIR, PNGCSV_DIR,
        shape)
    if shape == '518':
        check_shape = (518,518,4)
    elif shape == '270':
        check_shape = (270,270,4)
    npy_list = get_names(OUT_DIR, '.npy')
    for file in npy_list:
        arr = np.load(OUT_DIR + file + '.npy')
        if arr.shape != check_shape:
            assert False
    assert True
    shutil.rmtree(OUT_DIR)