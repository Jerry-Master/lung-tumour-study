"""

Converts QuPath geojson format into our png <-> csv format.

"""
import argparse
from ..utils/preprocessing import *
import geojson
import cv2
from skimage.draw import polygon
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gson_dir', type=str, required=True, 
                    help='Path to the geojson files.')
parser.add_argument('--png_dir', type=str, required=True,
                    help='Path to save the png files.')  
parser.add_argument('--csv_dir', type=str, required=True,
                    help='Path to save the csv files.')      

def read_gson(name, path):
    with open(path + name + '.geojson', 'r') as f:
        gson = geojson.load(f)
    return gson

def geojson2pngcsv(gson, names):
    """
    Computes png <-> csv labels from geojson. 
    Width and height are assumed to be 1024.
    """
    png = np.zeros((1024,1024))
    csv = pd.DataFrame([], columns=['id', 'label'])
    label_parser = {'tumour': 1, 'non-tumour': 0}
    for k, feature in enumerate(gson):
        # Draw filled contour
        contour = feature['geometry']['coordinates'][0]
        if len(contour) <= 1:
            continue
        assert(contour[0] == contour[-1])
        poly = np.array(contour[:-1])
        rr, cc = polygon(poly[:,0], poly[:,1], png.shape)
        png[rr,cc] = k+1

        # Save row to csv
        label = feature['properties']['classification']['name']
        label = label_parser[label]
        csv = pd.concat([csv, pd.DataFrame([[k+1,label]], columns=['id', 'label'])])
        
    return png, csv

def save_pngcsv(png, csv, png_path, csv_path, name):
    png = np.array(png, dtype=np.uint16)
    cv2.imwrite(png_path + name + '.png', png)
    csv.to_csv(csv_path + name + '.csv', index=False, header=False)

if __name__ == '__main__':
    args = parser.parse_args()
    GSON_DIR = parse_path(args.gson_dir)
    PNG_DIR = parse_path(args.png_dir)
    CSV_DIR = parse_path(args.csv_dir)

    create_dir(PNG_DIR)
    create_dir(CSV_DIR)

    names = get_names(GSON_DIR, '.geojson')
    for k, name in enumerate(names):
        print('Progress: {:2d}/{}'.format(k+1, len(names)), end="\r")
        gson = read_gson(name, GSON_DIR)
        png, csv = geojson2pngcsv(gson, name)
        save_pngcsv(png, csv, PNG_DIR, CSV_DIR, name)
    print()
