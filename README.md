# vox2dem

## Introduction
A CUDA accelerated C++ tool to extract digital elevation models (i.e. heightmaps) from building voxels.

Input voxels to this tool should be of the format `.binvox`. This is a format used by [**binvox**](http://www.patrickmin.com/binvox/), which is a tool to read 3D models, rasterizes it into binary 3D voxel grid, and writes to voxel files.

Outputs of this tool are `.png` images. Shapes of the images depend on shape of the input voxels. For example, if a voxel grid is 256x256x256, then the output image will be 256x256 pixels.


## Usage

Build
```
git clone https://github.com/yuqli/vox2dem.git
cd build
cmake ../src
make
```
This will build target `v2d` from `main.cu`.  
Dependencies: OpenCV and CUDA

To use `v2d`:

`./v2d <cuda device> <binvox filename> <output root folder> <crop and scale >`

Options:
- `<cuda device>`: int, indicating which GPU to use, starting from 0.
- `<binvox filename>`: string, the full path to `.binvox` file.
- `<output root folder>`: string, the full path to the folder that stores output `.png` images.
- `<crop and scale>`: binary int. `0` means no crop or scale, `1` means crop and scale.   

Notes on output images:
1. Some voxels could be empty, i.e. none of the grids are occupied. In this case, the program will write paths to the empty voxels to `empty_log.txt` located in the `<output root folder>`.  
2. If choose no crop or scale, the DEM will be the height map on the x-y plane. i.e. for each pixel, the height is the greatest `z` axis value that is occupied. For __BuildingNet__ data, the pixel values in the result image are always integers in the range of 0 to 255, making the resultant image grayscale.
3. If choose crop and scale, this application will 1) binarize the image and 2) find the contour of the building shape and 3) creating a bounding box along shape contour and 4) crop and rescale the bounding box into a 256x256 pixel images.
4. For point 2 and 3 above, note every output image will have accompanying metadata. If no cropping, metadata will include the translation and scaling factor produced by **binvox**. If crop and scale, additional metadata will include the perspective transform matrix that could revert the cropped height map into original images. This perspective transform matrix will be stored in .xml format in the [TODO].


## Example
After building, run the following command from the `build` directory:

`./v2d 0 ../sample/test.binvox ../sample/result 1`

Resultant original image :

![](https://github.com/yuqli/vox2dem/blob/master/sample/result/output_original/test.png)

Resultant cropped image :

![](https://github.com/yuqli/vox2dem/blob/master/sample/result/output_cropped/test.jpg)

## Contact
For any questions please contact Yuqiong Li at yl5090 at nyu dot edu.
