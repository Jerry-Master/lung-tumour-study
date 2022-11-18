# Utils

## Preprocessing

Functions that appear more than once across the preprocessing scripts. There are functions for dealing with user input: `parse_path`, `create_dir`. For dealing with file terminations: `read_names`, `get_names`. And for reading and saving the different formats: `read_labels`, `read_json`, `create_geojson`, `save_pngcsv`, `read_centroids`. For more information call `help` on them.

## Postprocessing

Utility functions for the postprocessing module. They are mainly for the rswoosh algorithm. The two main functions used in rswoosh algorithm are: `create_comparator` and `merge_cells`. They provide the comparison and merge operation for the algorithm. The rest of the functions are auxiliary functions for those two.

## Nearest

Helper functions to compute the nearest point to a given set of points stored in a KDtree. Current functionality includes `find_nearest` and `find_nearest_dist_idx` that return index or index and distance respectively. And `generate_tree` to create the KDtree.
