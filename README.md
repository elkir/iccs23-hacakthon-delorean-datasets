# delorean-datasets


Tasks:
Streamline & Extract: Loop through the dataset, apply a "reducer" to create a single time-series (for instance, a spatial mean across ensembles) for each file, and assemble them into one comprehensive file. [Completed]

Mathematical Manipulation: Run a loop that applies a mathematical operation (such as deriving wind power from wind speed) to a variable and extract this data into a new file, resulting in a fresh file for every original file. [Easy]

Combine & Compress: Execute a loop that first carries out task 2), then task 1), but only stores the final result. [Easy]

Distributed Processing: Implement a loop similar to 1) that processes files in parallel on up to 76 nodes and combines them into a final dataset. [Medium]

Cross-Date Reducer: Develop a loop that applies a reducer across each date of the year, creating a single timeseries spanning a whole year (with 8 overlapping files for each date x 50 different ensembles in each file). [Medium]

Distributed Cross-Date Reducer: Similar to 5), but in a distributed parallel manner, processing each file separately and assembling everything in the end. [Conceptually challenging - linearity constraints on the reducers?]

Spatial Reducers: Utilize GDAL/Shapely vector-based spatial reducers to map lat×lon grid data to nodes of the grid using the corresponding regions or country shapes. [Medium]

Memory Management: Parallelize and track memory usage of a project to optimize the number of CPUs available. (The CPU cluster has 76 CPUs but only 3.4 or 6.7GB of RAM per CPU, so for larger memory tasks, the CPU usage needs to be divided by (RAM used)/3.4GB). [Hard?]

Data Batching: Create a dataloader for Pytorch that can efficiently batch this data (the ensembles in each file are independent so the batch size can be reduced substantially). [Medium]

Bonus Round:
We also have another version of the files sourced from the reforecasts, which use a set of initial conditions from a range of years in each (re)forecast/hindcast:
★ Each file (~6.4GB in a GRIB format) contains:
(20 years) × (10 ensemble members) × (6 variables) × (6h time-series for a 4-week duration) × (0.5 lat × 0.5 lon grid of data).
