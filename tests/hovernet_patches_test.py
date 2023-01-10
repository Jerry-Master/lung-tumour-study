import pytest
import sys
import os
import shutil
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TEST_DIR)
sys.path.append(PKG_DIR)

from utils.preprocessing import parse_path, get_names
from hover_net.extract_patches import save_npy, to4dim, to5dim, create_dir

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
    match shape:
        case '518':
            check_shape = (518,518,5)
        case '270':
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
    match shape:
        case '518':
            check_shape = (518,518,4)
        case '270':
            check_shape = (270,270,4)
    npy_list = get_names(OUT_DIR, '.npy')
    for file in npy_list:
        arr = np.load(OUT_DIR + file + '.npy')
        if arr.shape != check_shape:
            assert False
    assert True
    shutil.rmtree(OUT_DIR)