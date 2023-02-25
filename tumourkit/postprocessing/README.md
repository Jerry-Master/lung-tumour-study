# Result post-processing

After training Hovernet and visualizing the results I detected one problem. Some cells were split in pieces. The solution is to merge those pairs of cells that share a big frontier. A first approach was made by using the R-Swoosh algorithm made for detecting duplicates in databases. Then, a morphological algorithm that yields equivalent results was found.

This module also contains scripts to extract the probability from Hovernet output and merging it into the graphs files.

## R-Swoosh

The algorithm itself is quite simple. You have a buffer where you save cells you have already visited, and while you are traversing the initial set of cells you check if there is some other cell in the buffer that can be merge with it. If there is, you join both of them and add the result to the initial set of cells. Otherwise, add the new cell to the buffer. The algorithm as expressed this way is quadratic, however, it can be converted to a linear one by using a windowed version of it. That last part was not implemented because we found a simpler algorithm that used morphological operations.

## Morphological equivalent

To detect a frontier using morphology you can just apply a gradient, take the difference and remove the background. This way you are left with only those pixels that were in the frontier between two cells. Moreover, the value in those pixels is the difference of the identifiers of both cells. By applying a convenient function prior to applying the morphological operations you can retrieve both indices, so that you can apply the merge operation by just changing both indices in the image to the same index.

### Usage

The usage is quite similar to that of the preprocessing files: `python3 merge_cells.py [FLAGS]`. Where the flags are:

* `--png-dir`, `--csv-dir`: The paths to the images and their labels respectively.
* `--output-path`: Path to save the new images and new labels. Two subfolders are created inside: `png/` and `csv/`.

## Hovernet probability

* `join_graph_gt.py`: Updates the class labels in the graph files with the GT class labels from centroid files. It removes cells that don't have a 1-1 matching, that is, we only consider cells in the prediction who are the closest cell to its nearest cell in the GT.
* `join_hovprob_graph.py`: Extracts probabilities from my version of hovernet json files and appends them in a new column (`prob1`) of the graph files. Should be executed after `join_graph_gt.py` for efficienfy reasons.

### Usage

Same as before: `python3 [script].py [FLAGS]`. Where the flags are:

* `--graph-dir`, `--centroids-dir`, `--json-dir`: The paths to the input folders.
* `--output-dir`: Path to folder where to save the result. If it coincides with any input directory, the input directory data will get overwritten.
