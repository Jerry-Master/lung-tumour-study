# Data processing
## Introduction

Biopsies are saved as tiles of 1024x1024 and the labels come in different formats. This module facilitates conversion between them. The two main supported formats are the following:

* PNG <-> CSV: This is a standard format for instance segmentation where every cell has an identifier. In the image (PNG) every pixel has the value of that identifier and in a separate table (CSV) each identifier has associated one class. The classes right now are: tumoural and non-tumoural. And the ID 0 is reserved for the background.
* GeoJSON: This is the format used in the QuPath program that I use for labelling the images. It is a standard geojson format where the geometry describes the contours of the cells and in the properties attribute is included the class of the cell.

Apart from those two formats there are other two formats storing additional information:

* Centroids: For evaluation purposes the centroids of the cells are precomputed and stored as a table (CSV) with the columns X, Y and class.
* Hovernet JSON: The output of the hovernet model comes with a JSON that has extra information like the position of the centroids or the contours. 

There is yet another format for storing graph nodes:

* `.nodes.csv`: Stores extracted attributes of cells and saves them in a csv. Current attributes are: (X,Y) center of bounding box, area, perimeter, gray level variance and RGB histogram.

## Files description

The supported conversions and their corresponding implementations are: 
* [PNG <-> CSV >> GeoJSON], implemented in `pngcsv2geojson.py`.
* [GeoJSON >> PNG <-> CSV], implemented in `geojson2pngcsv.py`.
* [PNG <-> CSV >> Centroids], implemented in `pngcsv2centroids.py`.
* [Hovernet JSON >> Centroids], implemented in `hovernet2centroids.py`.
* [Hovernet JSON >> GeoJSON], implemented in `hovernet2geojson.py`.
* [Centroids >> PNG], implemented in `centroids2png.py`. The png format here is different from the one mentioned above, this one just contains the centroids with a pixel value of 255.
* [PNG <-> CSV >> Graph], implemented in `pngcsv2graph.py`. It is implemented with multiprocessing because is a quite slow process. You can specify the number of threads in the flag `--num-workers`.

## Script usage

All the scripts are used in the same way `python3 [SCRIPT_NAME].py [FLAGS]`. There are five different flags that correspond to the paths of the files to convert:

* `--png-dir`, `--csv-dir`: The paths to the images and their labels, used in any script that converts to or from PNG <-> CSV format. It indicates input or output depending on the conversion.
* `--gson-dir`: Path to the geojson files, either input or output depending on conversion.
* `--json-dir`: Path to hovernet jsons, only used as input.
* `--output-path`: Path to folder where to save output, used in several scripts.
* `--centroids-path`: Path used in the `centroids2png.py` file to retrieve the centroids.
* `--orig-dir`: Path to original images.

If any folder indicated as output doesn't exist, it is created at runtime.
