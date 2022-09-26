import os
from geojson import Feature, Polygon

def parse_path(path):
    if path[-1] != '/':
        return path + '/'
    return path

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def get_names(path, pattern):
    """
    Assumes the names end in <pattern>
    """
    names_raw = os.listdir(path)
    names = []
    for name in names_raw:
        if pattern in name:
            names.append(name[:-len(pattern)])
    return names

def create_geojson(contours):
    label_dict = ["non-tumour", "tumour"]
    colour_dict = [-9408287, -9408287]
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