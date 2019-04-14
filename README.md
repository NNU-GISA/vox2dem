## 20190302
CUDA-accelerated binvox to heightmap conversion. Also cropped ROI and rescale into 256x256 image.
Inputs:
- path to a `.binvox` file

Outputs:
- original heightmap. If empty, will write to "empty_log.txt". This file must be created side-by-side with the b2h application.
- binvox metadata. include translation and scaling factor
- cropped heightmap. will be stored in another folder
- cropped perspective transform matrix. will be stored in .xml format in another folder.

main.cu is the main program.

Usage:

~~`cd cmake-build-debug`~~
~~Then do `make` and will build target b2h with main.cu~~

`CMake` in the same directory as this README.md document
Dependency: OpenCV and CUDA
Caveats:
- some warnings in the building process as sm_arch is not properly set. but currently the program runs ok.
- output paths in `main.cu` is hard coded and needed to be changed in your local OS.
- need to reconfigure according to your local systems.