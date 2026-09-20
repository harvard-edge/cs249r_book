# Spatial-memory storage illustration

`volumetric_memory_scaling.csv` lists grid resolutions only. The chapter plotting cell derives three constructed storage models for one fixed cubic scene: dense `4N^3` bytes, an illustrative surface octree with `N^2` allocated 32-byte cells, and 500,000 Gaussians at an assumed 56 bytes each (28 decimal MB). The fixed Gaussian count is not an equal-geometric-accuracy assumption. These are not measured update or collision-query benchmarks.
