import os
from geojson import Feature, Polygon
import cv2
import pandas as pd
import numpy as np
import json

def parse_path(path):
    """
    Checks for trailing slash and adds it in case it isn't there.
    """
    if path[-1] != '/':
        return path + '/'
    return path

def create_dir(path):
    """
    Checks if the folder exists, if it doesn't, it is created
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def get_names(path, pattern):
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

def read_names(file_path):
    """
    Given txt with one name at each line,
    returns a list with all the names.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        files = [line.strip() for line in lines]
    return files

def read_labels(name, png_path, csv_path):
    """
    Input: name of file and paths to their location in png and csv format.
           Files should end in .GT_cells.png and .class.csv respectively.
    Output: png and csv of that file.
    """
    img = cv2.imread(png_path + name + '.GT_cells.png', -1)
    csv = pd.read_csv(csv_path + name + '.class.csv')
    csv.columns = ['id', 'label']
    return img, csv

def read_json(json_path):
    """
    Input: Hovernet json path
    Output: Dictionary with nuclei information
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data['nuc']
    return nuc_info

def create_geojson(contours):
    """
    Input: List of pairs (contour, label).
        Contour is a list of points starting and finishing in the same point.
        label is an integer representing the class of the cell (1: non-tumour, 2: tumour)
    Returns: A dictionary with the geojson format of QuPath
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
