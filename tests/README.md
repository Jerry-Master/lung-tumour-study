# Tests
## PNG <-> CSV to GeoJSON conversion

In order to check if this conversion works the following test was designed. If we call $f$ the PNG <-> CSV to GeoJSON, then the GeoJSON to PNG <-> CSV should be $f^{-1}$. Therefore, we just have to check that $f^{-1} (f(\text{png, csv})) = (\text{png, csv})$. Since the algorithm of contour detection makes some approximations the result may not be exactly equal, therefore the result has a relative tolerance of $10\%$.

 ### Tests format
 
 In the folder `pngcsv` there should be png, csv pairs. The name of the png should end in `.GT_cells.png` and the name of the csv should end in `.class.csv`, both with the same name prefix. The format of the data is explained in the [preprocessing](../preprocessing) folder.
