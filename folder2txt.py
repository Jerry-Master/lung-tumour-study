"""

Script to generate txt files with the names of the files in the 
train, validation, test split folders.

Input
-----
The path to the folders.

Output
------
Txt files with one name per line, the names don't contain the 
(x,y).npy at the end.

"""
import argparse
import os
import re
from utils.preprocessing import parse_path, create_dir

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str,
                    help='Path to train folder as created by my_extract_patches.py')
parser.add_argument('--validation', type=str,
                    help='Path to validation folder as created by my_extract_patches.py')
parser.add_argument('--test', type=str,
                    help='Path to test folder as created by my_extract_patches.py')
parser.add_argument('--out_dir', type=str,
                    help='Path to save the txt files.')

def remove_coordinates(name):
    """
    Removes the pattern (number, number).npy from the name.
    """
    regex = '\([0-9]{1,3},[0-9]{1,3}\)'
    parenthesis = re.findall(regex, name)
    par_len = len(parenthesis[0])
    npy_len = len('.npy')
    return name[:-(par_len + npy_len)]

def read_files(path):
    """
    Returns a list of names at path without the (x,y).npy at the end.
    """
    files = os.listdir(path)
    file_list = []
    for file_name in files:
        file_list.append(remove_coordinates(file_name))
    file_list = list(set(file_list))
    return file_list

def save_txt(file_list, path, name):
    """
    Saves list of files in a txt, one name per line.
    """
    with open(path + name, 'a') as f:
        for file_name in file_list:
            print(file_name, file=f)

if __name__=='__main__':
    args = parser.parse_args()
    train_files = read_files(args.train)
    val_files = read_files(args.validation)
    test_files = read_files(args.test)

    OUT_DIR = parse_path(args.out_dir)
    create_dir(OUT_DIR)

    save_txt(train_files, OUT_DIR, 'train.txt')
    save_txt(val_files, OUT_DIR, 'validation.txt')
    save_txt(test_files, OUT_DIR, 'test.txt')