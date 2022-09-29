# Tests
## PNG <-> CSV to GeoJSON conversion

In order to check if this conversion works the following test was designed. If we call $f$ the PNG <-> CSV to GeoJSON, then the GeoJSON to PNG <-> CSV should be $f^{-1}$. Therefore, we just have to check that $f^{-1} (f(\text{png, csv})) = (\text{png, csv})$. Since the algorithm of contour detection makes some approximations the result may not be exactly equal, therefore the result has a relative tolerance of $10\%$.

 ### Tests format
 
 In the folder `pngcsv` there should be png, csv pairs. The name of the png should end in `.GT_cells.png` and the name of the csv should end in `.class.csv`, both with the same name prefix. The format of the data is explained in the [preprocessing](../preprocessing) folder.
 
 ## Evaluation test
 
 The metric for this dataset is the weighted F1 score over the classes of the centroids. However, the centroids may not coincide and so a 1-1 correspondence of centroids is needed previous to creating the confusion matrix. The test `metrics_test` checks that this association is correctly done.
 
 ### Tests format
 
 Each test should have three separate files, ended in `.A.csv`, `.B.csv` and `.result.csv` respectively, and with the same name prefix. File A should contain the ground truth centroids as a table with columns X, Y and class. File B should contain predicted centroids and the file result must be the corresponding confusion matrix that should result from those predictions.
 
 ### Tests generation
 
 The file `generate_centroids.py` automatically generates tests. There are different ways of creating tests in there:
 
 * __Random points__: A and B both contain the same random points with some predefined classes. 
 * __Circle__: A and B both contain points in a circle, but B has the points slightly rotated and / or scaled.
 * __Random labels__: Same approaches as before for point generation, but now labels are randomly generated for A and B. The real confusion matrix is computed with sklearn's `confusion_matrix` function.
