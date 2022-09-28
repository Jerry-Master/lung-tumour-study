import os
from geojson import Feature, Polygon

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

def create_geojson(contours):
    """
    Input: List of pairs (contour, label).
        Contour is a list of points starting and finishing in the same point.
        label is an integer representing the class of the cell
    Returns: A dictionary with the geojson format of QuPath
    """
    label_dict = ["background", "non-tumour", "tumour"]
    colour_dict = [-9408287, -9408287, -9408287]
    features = []
    for contour, label in contours:
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
