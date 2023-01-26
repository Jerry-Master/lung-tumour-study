# Tests
## PNG <-> CSV to GeoJSON to Graph conversion

In order to check if this conversion works the following test was designed. If we call $f$ the PNG <-> CSV to GeoJSON, then the GeoJSON to PNG <-> CSV conversion should be $f^{-1}$. Therefore, we just have to check that $f^{-1} (f(\text{png, csv})) = (\text{png, csv})$. Since the algorithm of contour detection makes some approximations the result may not be exactly equal, therefore the result has a relative tolerance of $10$\%.

To test the creation of graph files, the positions of the centroids are compared to assure that at least the nodes have correct coordinates (`graph_centroid_test.py`). The identifier is also checked (`graph_idx_test.py`) since it is a key component when adding hovernet probabilities.

 ### Tests format
 
 In the folder `pngcsv` there should be png, csv pairs. The name of the png should end in `.GT_cells.png` and the name of the csv should end in `.class.csv`, both with the same name prefix. The format of the data is explained in the [preprocessing](../preprocessing) folder.
 
 ## Evaluation test
 
 The metric for this dataset is the weighted F1 score over the classes of the centroids. However, the centroids may not coincide and so a 1-1 correspondence of centroids is needed previous to creating the confusion matrix. The test `metrics_test` checks that this association is correctly done.

 The data of this tests is also used to test that the pairs of 1-1 correspondences are well made (`metrics_pairs_test.py`) and also to check that the confusion matrix containing the background class is correctly computed (`conf_matrix_sum_test.py`).
 
 ### Tests format
 
 Each test should have three separate files, ended in `.A.csv`, `.B.csv` and `.result.csv` respectively, and with the same name prefix. File A should contain the ground truth centroids as a table with columns X, Y and class. File B should contain predicted centroids and the file result must be the corresponding confusion matrix that should result from those predictions.
 
 ### Tests generation
 
 The file `generate_centroids.py` automatically generates tests. There are different ways of creating tests in there:
 
 * __Random points__: A and B both contain the same random points with some predefined classes. 
 * __Circle__: A and B both contain points in a circle, but B has the points slightly rotated and / or scaled.
 * __Random labels__: Same approaches as before for point generation, but now labels are randomly generated for A and B. The real confusion matrix is computed with sklearn's `confusion_matrix` function.
 * __Extra points__: Same as random labels but there are extra predicted points far away from the others that should be ignored.

## R-Swoosh test

The R-Swoosh is a very generic algorithm that depends on two operations: merge and equality. Tests here check that the algorithm works under some circumtances and fixing the relevant operations. There are also tests for the auxiliary functions employed. In the main test (`rswoosh_test.py`), several cells are given as input and the test checks that the predicted output has the same number of cells as the expected output.

### Tests format
 
Each test should have two separate files ending in `.input.json` and `.output.json` being each one of them the input to the rswoosh function and the expected output. The format of the files must be a list of cells, being a cell a tuple of two integers (id and class) and contour, where a contour is a list of points (tuples of two floats).

### Test generation

All the tests are made of the same basic unit: a circle split in two. In the input file both semicircles are included while in the output there is only the whole circle. Given that unit, each test has multiple of them across the plane translated and rotated. And not all the circles are split in two, just a random selection of them. Several radii are used too. 

### Auxiliar tests

Tests for the functions `remove_idx`, `get_N_closest_pairs_dists` and `get_greatest_connected_component` are also provided (`remove_idx_test.py`, `cpairs_test.py`, `get_conn_comp_test.py`). They both consists on few cases introduced manually to check simple errors. It does not pretend to be an exhaustive list of tests, just a simple one to detect minor implementation issues.

## Hovernet test

Hovernet has its own input format, in `hovernet_patches_test.py` the shape of the numpy input files is checked.

### Tests format

Same data in `pngcsv` as in the conversion tests, additional examples images must be provided inside `tiles`.

## Nodes test

In `read_nodes_test.py` the split of nodes is being tested so that the 'by_img' split and the 'total' split coincide in shapes.

### Tests format

In the folder `graphs` must be some example `.nodes.csv` files.
