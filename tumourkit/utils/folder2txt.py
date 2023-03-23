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
from typing import List
import argparse
import os
import re
from .preprocessing import parse_path, create_dir

parser = argparse.ArgumentParser()
parser.add_argument('--train', type=str,
                    help='Path to train folder as created by my_extract_patches.py')
parser.add_argument('--validation', type=str,
                    help='Path to validation folder as created by my_extract_patches.py')
parser.add_argument('--test', type=str,
                    help='Path to test folder as created by my_extract_patches.py')
parser.add_argument('--out-dir', type=str,
                    help='Path to save the txt files.')

def remove_coordinates(name: str) -> str:
    """
    Removes the (number, number) pattern and '.npy' extension from the input string.

    :param name: The input string with a (number, number) pattern and '.npy' extension.
    :type name: str
    :return: The input string with the (number, number) pattern and '.npy' extension removed.
    :rtype: str
    """
    regex = '\([0-9]{1,3},[0-9]{1,3}\)'
    parenthesis = re.findall(regex, name)
    par_len = len(parenthesis[0])
    npy_len = len('.npy')
    return name[:-(par_len + npy_len)]

def read_files(path: str) -> List[str]:
    """
    Returns a list of file names without the (number, number).npy pattern at the end.

    :param path: The path to the directory containing the files.
    :type path: str
    :return: A list of file names without the (number, number).npy pattern at the end.
    :rtype: List[str]
    """
    files = os.listdir(path)
    file_list = []
    for file_name in files:
        file_list.append(remove_coordinates(file_name))
    file_list = list(set(file_list))
    return file_list

def save_txt(file_list: List[str], path: str, name: str) -> None:
    """
    Saves a list of file names to a text file with one name per line.

    :param file_list: The list of file names to save.
    :type file_list: list[str]
    :param path: The path to the directory where the text file will be saved.
    :type path: str
    :param name: The name of the text file to be saved.
    :type name: str
    :return: None
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