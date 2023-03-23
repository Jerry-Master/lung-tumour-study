"""
Converts QuPath geojson format into our png <-> csv format.

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
from typing import Dict, Any, Tuple, List, Optional
import argparse
import geojson
from skimage.draw import polygon
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utils.preprocessing import parse_path, create_dir, get_names, save_pngcsv
    

def read_gson(name: str, path: str) -> List[Dict[str,Any]]:
    """
    Reads the GeoJSON file at the specified path with the given name.

    :param name: The base name of the file (without extension).
    :type name: str
    :param path: The path to the directory containing the file.
    :type path: str
    :return: The contents of the GeoJSON file as a list of dictionaries.
    :rtype: List[Dict[str, Any]]
    """
    with open(path + name + '.geojson', 'r') as f:
        gson = geojson.load(f)
    return gson


def geojson2pngcsv(gson: List[Dict[str, Any]], num_classes: Optional[int] = 2) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes PNG and CSV labels from GeoJSON.

    :param gson: A list of GeoJSON features.
    :type gson: List[Dict[str, Any]]
    :param num_classes: The number of classes to use in the output CSV. If not provided, it defaults to 2 (tumour and non-tumour).
    :type num_classes: Optional[int]
    :return: A tuple containing the PNG image array and the Pandas DataFrame of the CSV file.
    :rtype: Tuple[np.ndarray, pd.DataFrame]
    
    This function takes a list of GeoJSON features and generates PNG and CSV labels from them.
    The width and height of the PNG image are assumed to be 1024.
    The function expects the GeoJSON to have specific format, with the geometry stored as a list of coordinates.
    If a feature has a label that is not "tumour" or "non-tumour", the label will be replaced with "ClassN", where N is the class number.
    """
    png = np.zeros((1024,1024), dtype=np.uint16)
    csv = pd.DataFrame([], columns=['id', 'label'])
    label_parser = {'tumour': 2, 'non-tumour': 1}
    if num_classes != 2:
        label_parser = {"Class"+str(k): k for k in range(1, num_classes+1)}
    for k, feature in enumerate(gson):
        # Draw filled contour
        contour = feature['geometry']['coordinates'][0]
        if len(contour) <= 1:
            continue
        assert(contour[0] == contour[-1])
        poly = np.array(contour[:-1])
        rr, cc = polygon(poly[:,0], poly[:,1], png.shape)
        png[cc,rr] = k+1

        # Save row to csv
        label = feature['properties']['classification']['name']
        label = label_parser[label]
        csv = pd.concat([csv, pd.DataFrame([[k+1,label]], columns=['id', 'label'])])
        
    return png, csv


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gson-dir', type=str, required=True, 
                        help='Path to the geojson files.')
    parser.add_argument('--png-dir', type=str, required=True,
                        help='Path to save the png files.')  
    parser.add_argument('--csv-dir', type=str, required=True,
                        help='Path to save the csv files.')
    parser.add_argument('--num-classes', type=int, default=2)
    return parser 


def main_with_args(args):
    gson_dir = parse_path(args.gson_dir)
    png_dir = parse_path(args.png_dir)
    csv_dir = parse_path(args.csv_dir)

    create_dir(png_dir)
    create_dir(csv_dir)

    names = get_names(gson_dir, '.geojson')
    for name in tqdm(names):
        gson = read_gson(name, gson_dir)
        png, csv = geojson2pngcsv(gson, args.num_classes)
        save_pngcsv(png, csv, png_dir, csv_dir, name)

def main():
    parser = _create_parser()
    args = parser.parse_args()
    main_with_args(args)
