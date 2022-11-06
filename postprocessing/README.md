# Result post-processing

After training Hovernet and visualizing the results I detected one problem. Some cells were split in pieces. The solution is to merge those pairs of cells that share a big frontier. A first approach was made by using the R-Swoosh algorithm made for detecting duplicates in databases. Then, a morphological algorithm that yields equivalent results was found.

## R-Swoosh

The algorithm itself is quite simple. You have a buffer where you save cells you have already visited, and while you are traversing the initial set of cells you check if there is some other cell in the buffer that can be merge with it. If there is, you join both of them and add the result to the initial set of cells. Otherwise, add the new cell to the buffer. The algorithm as expressed this way is quadratic, however, it can be converted to a linear one by using a windowed version of it. That last part was not implemented because we found a simpler algorithm that used morphological operations.

## Morphological equivalent

To detect a frontier using morphology you can just apply a gradient, take the difference and remove the background. This way you are left with only those pixels that were in the frontier between two cells. Moreover, the value in those pixels is the difference of the identifiers of both cells. By applying a convenient function prior to applying the morphological operations you can retrieve both indices, so that you can apply the merge operation by just changing both indices in the image to the same index.

### Usage

The usage is quite similar to that of the preprocessing files: `python3 merge_cells.py [FLAGS]`. Where the flags are:

* `--png_dir`, `--csv_dir`: The paths to the images and their labels respectively.
* `--output_path`: Path to save the new images and new labels. Two subfolders are created inside: `png/` and `csv/`.
