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
import numpy.ma as ma
import json
from typing import Dict, Any, List, Tuple, Optional


def parse_path(path: str) -> str:
    """
    Parses a file path and checks for a trailing slash.

    :param path: The file path to parse.
    :type path: str
    :return: The parsed file path with a trailing slash.
    :rtype: str
    """
    if path[-1] != '/':
        return path + '/'
    return path

def create_dir(path: str) -> None:
    """
    Creates a directory at the specified path if it does not already exist.

    :param path: The path of the directory to create.
    :type path: str
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def get_names(path: str, pattern: str) -> List[str]:
    """
    This function returns a list of all files in a directory located at <path> that contain the substring <pattern> in their name. The <pattern> substring is then removed from the names and the resulting list is returned.

    :param path: The path of the directory to search in.
    :type path: str
    :param pattern: The substring to search for in file names.
    :type pattern: str
    :return: A list of file names without the <pattern> substring.
    :rtype: List[str]
    """
    names_raw = os.listdir(path)
    names = []
    for name in names_raw:
        if pattern in name:
            names.append(name[:-len(pattern)])
    return names

def read_names(file_path: str) -> List[str]:
    """
    Returns a list of file names in the specified path that contain the given pattern.

    :param path: The path to search for files.
    :type path: str
    :param pattern: The pattern to search for in file names.
    :type pattern: str
    :return: A list of file names in the specified path that contain the given pattern.
    :rtype: List[str]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        files = [line.strip() for line in lines]
    return files


def read_png(name: str, png_dir: str) -> np.ndarray:
    """
    Reads a PNG file from the specified directory and returns its image data as a NumPy array.

    :param name: The name of the PNG file (without extension).
    :type name: str
    :param png_dir: The directory containing the PNG file.
    :type png_dir: str
    :return: A NumPy array containing the image data.
    :rtype: np.ndarray or None

    This function reads a PNG file from the specified directory and returns its image data as a NumPy array.
    The function expects the PNG file to have a specific filename format: '<name>.GT_cells.png'.
    If the function is unable to read the PNG file or encounters an error, it returns `None`.
    """
    try:
        img = cv2.imread(os.path.join(png_dir, name + '.GT_cells.png'), -1)
        return img
    except:
        return None


def read_csv(name: str, csv_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Reads a CSV file from the specified directory and returns its contents as a Pandas DataFrame.

    :param name: The name of the CSV file (without extension).
    :type name: str
    :param csv_dir: The directory containing the CSV file.
    :type csv_dir: str
    :return: A tuple containing the ID and label columns of the CSV file as NumPy arrays, and the entire DataFrame.
    :rtype: Tuple[np.ndarray, np.ndarray, pd.DataFrame] or None

    This function reads a CSV file from the specified directory and returns its contents as a Pandas DataFrame.
    The function expects the CSV file to have a specific filename format: '<name>.class.csv'.
    If the function is unable to read the CSV file or encounters an error, it returns `None`.
    """
    try:
        csv = pd.read_csv(os.path.join(csv_dir, name + '.class.csv'), header=None)
        csv.columns = ['id', 'label']
        return csv
    except:
        return None
    

def read_labels(name: str, png_dir: str, csv_dir: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Reads a PNG and CSV file from the specified directories and returns their contents as a tuple.

    :param name: The base name of the files (without extensions).
    :type name: str
    :param png_dir: The directory containing the PNG file.
    :type png_dir: str
    :param csv_dir: The directory containing the CSV file.
    :type csv_dir: str
    :return: A tuple containing the NumPy array of the PNG file and the Pandas DataFrame of the CSV file.
    :rtype: Tuple[np.ndarray, pd.DataFrame] or None

    This function reads a PNG and CSV file from the specified directories and returns their contents as a tuple.
    The function expects the PNG and CSV files to have specific filename formats: '<name>.GT_cells.png' and '<name>.class.csv'.
    If the function is unable to read either file or encounters an error, it returns (`None`,`None`).
    """
    try:
        img = read_png(name, png_dir)
        csv = read_csv(name, csv_dir)
        return img, csv
    except:
        return None, None


def read_json(json_path: str) -> Dict[str, Any]:
    """
    Reads a Hovernet JSON file from the specified path and returns the nuclei information as a dictionary.

    :param json_path: The path to the Hovernet JSON file.
    :type json_path: str
    :return: A dictionary containing the nuclei information.
    :rtype: Dict[str, Any] or None

    This function reads a Hovernet JSON file from the specified path and returns the nuclei information as a dictionary.
    If the function is unable to read the file or encounters an error, it returns `None`.
    The function expects the JSON file to contain a 'nuc' key with the nuclei information.
    """
    with open(json_path) as json_file:
        data = json.load(json_file)
        nuc_info = data['nuc']
    return nuc_info

def read_centroids(name: str, path: str) -> np.ndarray:
    """
    Reads a CSV file from the specified directory containing centroids and returns their contents as a NumPy array.

    :param name: The base name of the CSV file (without extension).
    :type name: str
    :param path: The directory containing the CSV file.
    :type path: str
    :return: A NumPy array containing the centroids with class -1 removed and coordinates converted to integers.
    :rtype: np.ndarray

    This function reads a CSV file from the specified directory containing centroids and returns their contents as a NumPy array.
    The function expects the CSV file to have specific column names: 'X', 'Y', and 'class'.
    Centroids with class -1 are dropped and the coordinates are converted to integers.
    """
    centroid_csv = pd.read_csv(path + name + '.centroids.csv')
    centroid_csv = centroid_csv.drop(centroid_csv[centroid_csv['class']==-1].index)
    return centroid_csv.to_numpy(dtype=int)


def read_graph(name: str, graph_dir: str) -> pd.DataFrame:
    """
    Reads a graph in CSV format from the specified directory and returns it as a Pandas DataFrame.

    :param name: The base name of the graph file (without extension).
    :type name: str
    :param graph_dir: The directory containing the graph file.
    :type graph_dir: str
    :return: The graph nodes as a Pandas DataFrame.
    :rtype: pd.DataFrame or None

    This function reads a graph in CSV format from the specified directory and returns it as a Pandas DataFrame.
    The function expects the graph file to have a specific filename format: '<name>.nodes.csv'.
    If the function is unable to read the file or encounters an error, it returns `None`.
    """
    df = pd.read_csv(os.path.join(graph_dir, name + '.nodes.csv'))
    return df


Point = Tuple[float,float]
Contour = List[Point]
def format_contour(contour: Contour) -> Contour:
    """
    Formats a contour in cv2.findContours format to an array of shape (N,2).
    Additionally, the first point is added to the end to close the contour.

    :param contour: The contour to be formatted.
    :type contour: Contour
    :return: The formatted contour.
    :rtype: Contour
    """
    new_contour = np.reshape(contour, (-1,2)).tolist()
    new_contour.append(new_contour[0])
    return new_contour

def create_geojson(contours: List[Tuple[int,int]], num_classes: Optional[int] = 2) -> List[Dict[str, Any]]:
    """
    Converts a list of contours and their labels to a list of dictionaries 
    containing GeoJSON-formatted data.

    :param contours: A list of pairs (contour, label) where contour is a list of points starting 
        and finishing in the same point and label is an integer representing the class of the 
        cell (1: non-tumour, 2: tumour).
    :type contours: List[Tuple[int, int]]
    :param num_classes: The number of classes in the data. Defaults to 2.
    :type num_classes: Optional[int]
    :return: A list of dictionaries with the GeoJSON format of QuPath.
    :rtype: List[Dict[str, Any]]

    This function converts a list of contours and their labels to a list of dictionaries 
    containing GeoJSON-formatted data. The function expects the contours to be a list of 
    pairs (contour, label), where contour is a list of points starting and finishing in the 
    same point, and label is an integer representing the class of the cell (1: non-tumour, 
    2: tumour). 

    The function returns a list of dictionaries with the GeoJSON format of QuPath. The 
    number of classes can be specified with the num_classes parameter. By default, it is set 
    to 2, but it can be set to any integer greater than or equal to 1. 

    The function uses a list of labels and colour codes to create the classification data for 
    the output dictionaries. The labels and colour codes are stored in the label_dict and 
    colour_dict lists, respectively. The label_dict list contains the names of the labels and 
    the colour_dict list contains the corresponding RGB colour codes for each label. By default, 
    the label_dict list contains the values ["background", "non-tumour", "tumour", "segmented"] 
    and the colour_dict list contains the value [-9408287, -9408287, -9408287, -9408287]. If 
    the num_classes parameter is set to a value other than 2, the label_dict list is updated 
    to contain the values ["background", "Class1", "Class2", ..., "ClassN"], where N is the 
    number of classes, and the colour_dict list is updated to contain the value 
    [-9408287, ..., -9408287] where the length of the list is N+1. Although this list is
    superfluous and not used by QuPath.

    Each dictionary in the output list represents a cell contour and contains a Polygon 
    object with the contour points, a classification object with the name and colour of the 
    label, and a boolean indicating whether the object is locked. The dictionary is in the 
    format expected by QuPath for reading annotations.
    """
    label_dict = ["background", "non-tumour", "tumour", "segmented"]
    colour_dict = [-9408287, -9408287, -9408287, -9408287]
    if num_classes != 2:
        label_dict = ["background"] + ["Class" + str(i) for i in range(1, num_classes+1)]
    colour_dict = [-9408287] * (num_classes + 1)
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


def save_png(png: np.ndarray, png_path: str, name: str) -> None:
    """
    Saves a PNG image to a file.

    :param png: The image to save.
    :type png: np.ndarray
    :param png_path: The path to the folder where the PNG file should be saved.
    :type png_path: str
    :param name: The name of the PNG file (without extension).
    :type name: str

    The file extension is set to '.GT_cells.png'.
    """
    png = np.array(png, dtype=np.uint16)
    cv2.imwrite(os.path.join(png_path, name + '.GT_cells.png'), png)


def save_csv(csv: pd.DataFrame, csv_path: str, name: str) -> None:
    """
    Saves the given CSV file to the specified directory with the specified name and a '.class.csv' extension.

    :param csv: The CSV file to save.
    :type csv: pd.DataFrame
    :param csv_path: The directory to save the CSV file in.
    :type csv_path: str
    :param name: The base name for the file (without extensions).
    :type name: str
    """
    csv.to_csv(os.path.join(csv_path, name + '.class.csv'), index=False, header=False)


def save_pngcsv(png: np.ndarray, csv: pd.DataFrame, png_path: str, csv_path: str, name: str) -> None:
    """
    Saves the given PNG and CSV files to the specified directories with the specified name.

    :param png: The PNG file to save.
    :type png: np.ndarray
    :param csv: The CSV file to save.
    :type csv: pd.DataFrame
    :param png_path: The directory to save the PNG file in.
    :type png_path: str
    :param csv_path: The directory to save the CSV file in.
    :type csv_path: str
    :param name: The base name for the files (without extensions).
    :type name: str

    The PNG file is saved with a '.GT_cells.png' extension, and the CSV file is saved with a '.class.csv' extension.
    """
    save_png(png, png_path, name)
    save_csv(csv, csv_path, name)


def save_centroids(centroids: np.ndarray, centroids_dir: str, name: str) -> None:
    """
    Saves the given centroids to the specified directory with the specified name and a '.centroids.csv' extension.

    :param centroids: The centroids to save.
    :type centroids: np.ndarray
    :param centroids_dir: The directory to save the centroids in.
    :type centroids_dir: str
    :param name: The base name for the file (without extensions).
    :type name: str
    """
    df = pd.DataFrame(centroids, columns=['X','Y','class'])
    df.to_csv(os.path.join(centroids_dir, name + '.centroids.csv'), index=False)


def save_graph(graph: pd.DataFrame, path: str) -> None:
    """
    Saves the given graph as a CSV file to the specified path.

    :param graph: The graph to save.
    :type graph: pd.DataFrame
    :param path: The path to save the graph file to.
    :type path: str
    """
    graph.set_index('id', inplace=True)
    graph.sort_index(inplace=True)
    graph.to_csv(path)

def get_centroid_by_id(img: np.ndarray, idx: int) -> Tuple[int, int]:
    """
    Given an image and an id representing a component, returns the centroid
    of the component as a tuple (x, y).
    
    :param img: A NumPy array representing the image.
    :type img: np.ndarray
    :param idx: An integer representing the id of the component.
    :type idx: int
    :return: A tuple representing the centroid of the component.
    :rtype: Tuple[int, int]

    If there are no pixels with the given id in the image, it returns (-1, -1).
    """
    X, Y = np.where(img == idx)
    if len(X) == 0 or len(Y) == 0:
        return -1, -1
    return X.mean(), Y.mean()


def get_mask(png: np.ndarray, idx: int) -> np.ndarray:
    """
    Given segmentation mask with indices as pixel values,
    returns the mask corresponding to the given index.

    :param png: Segmentation mask with indices as pixel values.
    :type png: np.ndarray
    :param idx: Index of the mask to retrieve.
    :type idx: int
    :return: The mask corresponding to the given index.
    :rtype: np.ndarray
    """
    png_aux = png.copy()
    png_aux[png_aux!=idx] = 0
    png_aux[png_aux!=0] = 1
    return png_aux


def apply_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Apply a binary mask to an RGB image.

    :param img: RGB image.
    :type img: np.ndarray
    :param mask: Binary mask.
    :type mask: np.ndarray
    :return: Tuple containing the masked image, the bounded box mask, the center of the mask in the masked image.
    :rtype: Tuple[np.ndarray, np.ndarray, int, int]

    For efficiency only the bounded box of the mask is returned, together with the center of the box.
    Coordinates are indices of array.
    """
    img_aux = img.copy()
    img_aux = img_aux * mask.reshape(*mask.shape, 1)
    x,y,w,h = cv2.boundingRect((mask * 255).astype(np.uint8))
    X, Y = np.where(mask == 1)
    if len(X) == 0 or len(Y) == 0:
        cx, cy = -1, -1
    else:
        cx, cy = X.mean(), Y.mean()
    return img_aux[y:y+h, x:x+w].copy(), mask[y:y+h, x:x+w].copy(), cx, cy


def read_image(name: str, path: str) -> np.array:
    """
    Given name image (without extension) and folder path,
    returns array with pixel values (RGB).

    :param name: Name of the image (without extension).
    :type name: str
    :param path: Path to the folder where the image is located.
    :type path: str
    :return: Array with pixel values (RGB).
    :rtype: np.array
    """
    aux = cv2.imread(os.path.join(path, name+'.png'))
    return cv2.cvtColor(aux, cv2.COLOR_BGR2RGB)


def compute_perimeter(c: Contour) -> float:
    """
    Compute the perimeter of a given contour.
    Prerequisites: First and last point must be the same.

    :param c: Contour represented as a list of points (x,y).
              The first and last point must be the same.
    :type c: List[Tuple[float, float]]
    :return: The perimeter of the contour.
    :rtype: float
    """
    diff = np.diff(c, axis=0)
    dists = np.hypot(diff[:,0], diff[:,1])
    return dists.sum()


def extract_features(msk_img: np.ndarray, bin_msk: np.ndarray, debug=False) -> Dict[str, np.ndarray]:
    """
    Extracts features from a given RGB bounding box of a cell and its mask.

    :param msk_img: RGB bounding box of a cell.
    :type msk_img: np.ndarray
    :param bin_msk: Binary mask of the cell.
    :type bin_msk: np.ndarray
    :param debug: Flag indicating if debug mode is enabled.
    :type debug: bool
    :return: A dictionary containing different extracted features.
    :rtype: Dict[str, np.ndarray]
    """
    if len(msk_img) == 0 or msk_img.max() == 0:
        return {}
    gray_msk = cv2.cvtColor(msk_img, cv2.COLOR_RGB2GRAY)
    # bin_msk = (gray_msk > 0) * 1
    feats = {}
    feats['area'] = bin_msk.sum()

    if debug: import pdb; pdb.set_trace()
    contours, _ = cv2.findContours(
        (bin_msk * 255).astype(np.uint8), 
        mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
    )
    contour = format_contour(contours[0])
    feats['perimeter'] = compute_perimeter(contour)
    
    gray_msk = ma.masked_array(
        gray_msk, mask=1-bin_msk
    )
    feats['std'] = gray_msk.std()

    msk_img = ma.masked_array(
        msk_img, mask=1-np.repeat(bin_msk.reshape((*bin_msk.shape,1)), 3, axis=2)
    )
    red = msk_img[:,:,0].compressed()
    red_bins, _ = np.histogram(red, bins=5, density=True)
    feats['red'] = red_bins
    green = msk_img[:,:,1].compressed()
    green_bins, _ = np.histogram(green, bins=5, density=True)
    feats['green'] = green_bins
    blue = msk_img[:,:,2].compressed()
    blue_bins, _ = np.histogram(blue, bins=5, density=True)
    feats['blue'] = blue_bins
    return feats


def add_node(graph: Dict[str, float], feats: Dict[str, np.ndarray]) -> None:
    """
    Given a dictionary with vectorial features,
    converts them into one column per dimension
    and add them into the global dictionary.

    :param graph: Dictionary with graph node data.
    :type graph: Dict[str, float]
    :param feats: Dictionary with vectorial features of a graph node.
    :type feats: Dict[str, np.ndarray]
    :return: None. The graph dictionary is modified in place.    
    """
    for k, v in feats.items():
        if hasattr(v, '__len__'):
            for i, coord in enumerate(v):
                if k + str(i) not in graph:
                    graph[k + str(i)] = []
                graph[k + str(i)].append(coord)
        else:
            if k not in graph:
                graph[k] = []
            graph[k].append(v)