sobfu
============
software for 3d reconstruction of non-rigidly deforming scenes using depth data, based on sobolevfusion

dependencies
-----------
* boost
* nvidia gpu with cuda >8.0
* opencv
* pcl 1.8.1
* vtk

installation
------------
to install, run `source setup.sh -a`

usage
------------
`./build/bin/app /path/to/data /path/to/params <--enable-viz> <--enable-log> <--verbose>`

* the data directory must include folders `color` and `depth`
* the folder `params` has files with recommended parameters for some of the scenes from volumedeform and killingfusion datasets

### options
* with `--enable-viz`, screenshots from the pcl viewer will be logged in .png format to `/path/to/data/screenshots`
* with `--enable-log`, meshes computed from the model tsdf's via marching cubes will be logged in .vtk format to `/path/to/data/meshes`
* `--verbose` and `--vverbose` control verbosity of the solver

example reconstructions
------------
reconstructions of some of the scenes from the volumedeform and killingfusion projects can be viewed on our [youtube playlist](http://bit.ly/sobfu-yt)

there is a large trade-off between frame rate and reconstruction quality--sample reconstructions ran at approx. 2fps

issues
------------

* drift in longer scenes due to less than perfect registration
* topological changes

references 
------------
```
@InProceedings{Slavcheva_2018_CVPR,
author = {Slavcheva, Miroslava and Baust, Maximilian and Ilic, Slobodan},
title = {SobolevFusion: 3D Reconstruction of Scenes Undergoing Free Non-Rigid Motion},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```
