"""

Module with utility functions for preprocessing.

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
from geojson import Feature, Polygon
import cv2
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Tuple


def parse_path(path: str) -> str:
    """
    Checks for trailing slash and adds it in case it isn't there.
    """
    if path[-1] != '/':
        return path + '/'
    return path

def create_dir(path: str) -> None:
    """
    Checks if the folder exists, if it doesn't, it is created.
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def get_names(path: str, pattern: str) -> List[str]:
    """
    Returns a list with all the files in <path> containing <pattern>.
    It removes the <pattern> substring from the names.
    """
    names_raw = os.listdir(path)
    names = []
    for name in names_raw:
        if pattern in name:
            names.append(name[:-len(pattern)])
    return names

def read_names(file_path: str) -> List[str]:
    """
    Given txt with one name at each line,
    returns a list with all the names.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        files = [line.strip() for line in lines]
    return files

def read_labels(name: str, png_dir: str, csv_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Input: name of file and paths to their location in png and csv format.
           Files should end in .GT_cells.png and .class.csv respectively.
    Output: png and csv of that file.
    """
    try:
        img = cv2.imread(png_dir + name + '.GT_cells.png', -1)
        csv = pd.read_csv(csv_dir + name + '.class.csv', header=None)
        csv.columns = ['id', 'label']
        return img, csv
    except:
        return None, None


def read_csv(name: str, csv_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Input: name of file and paths to their location in csv format.
           Files should end in .class.csv.
    Output: csv of that file.
    """
    try:
        csv = pd.read_csv(csv_dir + name + '.class.csv', header=None)
        csv.columns = ['id', 'label']
        return csv
    except:
        return None


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Input: Hovernet json path
    Output: Dictionary with nuclei information
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data['nuc']
    return nuc_info

def read_centroids(name: str, path: str) -> np.ndarray:
    """
    Format of the csv should be columns: X, Y, class
    """
    centroid_csv = pd.read_csv(path + name + '.centroids.csv')
    centroid_csv = centroid_csv.drop(centroid_csv[centroid_csv['class']==-1].index)
    return centroid_csv.to_numpy(dtype=int)

Point = Tuple[float,float]
Contour = List[Point]
def format_contour(contour: Contour) -> Contour:
    """
    Auxiliary function to pass from the cv2.findContours format to
    an array of shape (N,2). Additionally, the first point is added
    to the end to close the contour.
    """
    new_contour = np.reshape(contour, (-1,2)).tolist()
    new_contour.append(new_contour[0])
    return new_contour

def create_geojson(contours: List[Tuple[int,int]]) -> List[Dict[str, Any]]:
    """
    Input: List of pairs (contour, label).
        Contour is a list of points starting and finishing in the same point.
        label is an integer representing the class of the cell (1: non-tumour, 2: tumour)
    Returns: A list of dictionaries with the geojson format of QuPath
    """
    label_dict = ["background", "non-tumour", "tumour", "segmented"]
    colour_dict = [-9408287, -9408287, -9408287, -9408287]
    features = []
    for contour, label in contours:
        assert(label > 0)
        points = Polygon([contour])
        properties = {
                    "object_type": "annotation",
                    "classification": {
                        "name": label_dict[label],
                        "colorRGB": colour_dict[label]
                    },
                    "isLocked": False
                    }
        feat = Feature(geometry=points, properties=properties)
        features.append(feat)
    return features

def save_pngcsv(png: np.ndarray, csv: pd.DataFrame, png_path: str, csv_path: str, name: str) -> None:
    """
    Save png, csv pair in the folders png_path and csv_path with the given name.
    """
    png = np.array(png, dtype=np.uint16)
    cv2.imwrite(png_path + name + '.GT_cells.png', png)
    csv.to_csv(csv_path + name + '.class.csv', index=False, header=False)

def save_graph(graph: pd.DataFrame, path: str) -> None:
    """
    Saves dataframe in given path.
    """
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)
    graph.to_csv(path)