// Date: 20190301
// Author: yuqiong Li
// read binvox and output digital elevation model (DEM) based on the voxel
// the DEM can also be interprted as a heightmap in which the z-axis is the heigh
//
// This example program reads a .binvox file and writes
// an ASCII version of the same file called "voxels.txt"
//
// 0 = empty voxel
// 1 = filled voxel
// A newline is output after every "dim" voxels (depth = height = width = dim)
//
// Note that this ASCII version is not supported by "viewvox" and "thinvox"
//
// The x-axis is the most significant axis, then the z-axis, then the y-axis.
// i.e. in binvox, read starts from y and then proceeds to z and finally
// Reference: http://www.patrickmin.com/binvox/binvox.html
//

#include <string>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cuda.h>
#include <opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
using namespace std;


#define index2(x, y, W) (x)*(W) + (y)  // used to index 2D objects in C style arrays; x is row and y is column
#define index3(z, x, y, H, W)  ((z)*(H)*(W)) + (x)*(W) + (y)  // used to index 3D objects in C style arrays
#define binvoxIndex3(z, x, y, H, W)  ((x)*(H)*(W)) + (z)*(W) + (y)  // used to index 3D objects in binvox voxels


using namespace std;

typedef unsigned char byte;

static int version;
static int D = 256;  // depth, also Z dim
const int W = 256;  // width, also X dim
const int H = 256;  // height, also Y dim
static int size;
static int imgsize;
static byte *voxels = 0;
static byte *img = 0;
static float tx, ty, tz;
static float scale;

__global__ void heightsKernel(byte * voxels, byte * heights);


int read_binvox(const string & filespec)
{

    ifstream *input = new ifstream(filespec.c_str(), ios::in | ios::binary);
    if ((*input).fail()){
        cout << "Error: file does not exist at " << filespec << "!" << endl;
    }
    //
    // read header
    //
    string line;
    *input >> line;  // #binvox
    if (line.compare("#binvox") != 0) {
        cout << "Error: first line reads [" << line << "] instead of [#binvox]" << endl;
        delete input;
        return 0;
    }
    *input >> version;
    cout << "reading binvox version " << version << endl;

    int depth, height, width;  // values from file, compare if the same
    depth = -1;
    int done = 0;
    while(input->good() && !done) {
        *input >> line;
        if (line.compare("data") == 0) done = 1;
        else if (line.compare("dim") == 0) {
            *input >> depth >> height >> width;
        }
        else if (line.compare("translate") == 0) {
            *input >> tx >> ty >> tz;
        }
        else if (line.compare("scale") == 0) {
            *input >> scale;
        }
        else {
            cout << "  unrecognized keyword [" << line << "], skipping" << endl;
            char c;
            do {  // skip until end of line
                c = input->get();
            } while(input->good() && (c != '\n'));

        }
    }
    if (!done) {
        cout << "  error reading header" << endl;
        return 0;
    }
    if (D == -1) {
        cout << "  missing dimensions in header" << endl;
        return 0;
    }

    size = W * H * D;
    voxels = new byte[size];  // danger! not initialized!
    if (!voxels) {
        cout << "  error allocating memory" << endl;
        return 0;
    }

    //
    // read voxel data
    //
    byte value;
    byte count;
    int index = 0;
    int end_index = 0;
    int nr_voxels = 0;

    input->unsetf(ios::skipws);  // need to read every byte now (!)
    *input >> value;  // read the linefeed char

    while((end_index < size) && input->good()) {
        *input >> value >> count;

        if (input->good()) {
            end_index = index + count;
            if (end_index > size) return 0;
            for(int i=index; i < end_index; i++) voxels[i] = value;

            if (value) nr_voxels += count;
            index = end_index;
        }  // if file still ok

    }  // while

    input->close();
    cout << "  read " << nr_voxels << " voxels" << endl;

    return 1;

}


int get_index(int x, int y, int z) {
    // used to get correct index from the voxel data
    // http://www.patrickmin.com/binvox/binvox.html
    int index = x * H*W + z * W + y;
    return index;
}


vector<string> split(const char *phrase, string delimiter){
    // https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c
    // used to parse file name
    vector<string> list;
    string s = string(phrase);
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}


void crop_min_box( Mat src_gray, string bid)
{
    // A function to crop and output minimum bounding box ...
    // @ src_gray : input gray scale image
    // @ bid : input file name to store results in folder
    // Also tested a version where return the results to see if reversion is successful

    // binarize grayscale image to binary image as to get simpler contour
    Mat binary_output;
    threshold(src_gray, binary_output, 5, 255, THRESH_BINARY);

    vector<vector<Point> > contours;
    findContours( binary_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE );

    vector<vector<Point> > contours_poly(contours.size());  // store polygons for contours
    // vector<Rect> rectangles(contours.size());  // stores bounding boxes
    vector<RotatedRect> rectangles(contours.size());  // stores bounding boxes
    // Rect minBoundRect;  // minimum bounding box
    RotatedRect minBoundRect;  // minimum bounding box

    // loop through and find smallest bounding boxes
    float min_width = 0;
    float min_height = 0;
    int min_square_idx = 0;

    for (size_t i = 0; i < contours.size(); i++)
    {
        // https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
        approxPolyDP(contours[i], contours_poly[i], 3, true );
        // Rect rectangle = boundingRect(contours_poly[i]);
        RotatedRect rectangle = minAreaRect(contours_poly[i]);
        rectangles[i] = rectangle;
        // Store the index position of the minimum square found
        if ((rectangle.size.width <= min_width) && (rectangle.size.height <= min_height))
        {
            min_width = rectangle.size.width;
            min_height = rectangle.size.height;
            min_square_idx = i;
        }
    }


    minBoundRect = rectangles[min_square_idx];

    Point2f rect_points[4];    // get rotated rectangle points
    minBoundRect.points( rect_points );

    Point2f dst_points[4];   // destination points anchors
    Point2f tl(0, 255);
    dst_points[0] = tl;
    Point2f bl(0, 0);
    dst_points[1] = bl;
    Point2f br(255, 0);
    dst_points[2] = br;
    Point2f tr(255, 255);
    dst_points[3] = tr;

    Mat M = getPerspectiveTransform(rect_points, dst_points);  // translation matrix
    Mat wrapped;
    Size wrapped_size(256, 256);
    warpPerspective(src_gray, wrapped, M, wrapped_size);
    // imshow("Wrapped binary output", wrapped);
    // string cropped_name = "/media/yuqiong/DATA/3dCityGan/playground/nyc_hmaps_cropped/" + bid + ".jpg";
    string cropped_name = "/data/gmldata/nyc_hmaps_cropped/" + bid + ".jpg";
    cv::imwrite(cropped_name, wrapped);

    // Write revert matrix to file!
    // string revert_name = "/media/yuqiong/DATA/3dCityGan/playground/nyc_hmaps_revert/" + bid + ".xml";
    string revert_name = "/data/gmldata/nyc_hmaps_revert/" + bid + ".xml";
    FileStorage out_file(revert_name, FileStorage::WRITE);
    out_file << "trans" << M;
    out_file.release();
}


Mat revert(Mat dst, string tls_file){
    // revert the original non-cropped file from cropped min-bounding box file
    // @ Mat dst : destination file. In correspondence to the dst file in the function crop_min_box
    // @ tls_file: filename stored for the ORIGINAL perspective transform. i.e. how to get from original picture to cropped ROI picture
    // example tls_file: "test.xml"
    // Read it back
    FileStorage in_file(tls_file, FileStorage::READ);
    Mat file_matrix;
    in_file["trans"] >> file_matrix;
    // cout << file_matrix << endl;
    in_file.release();

    // now test invert
    Mat invert;
    Size invert_size(256, 256);
    warpPerspective(dst, invert, file_matrix, invert_size, WARP_INVERSE_MAP);
    // TO-DO: add path to store revert images
    return invert;
}


int main(int argc, char **argv)
{
    if (argc != 3) {
        cout << "Usage: read_binvox <cuda device> <binvox filename>" << endl << endl;
        exit(1);
    }

    // check if exceeding number of GPUs
    int num_devices, device;
    device = *argv[1] - '0';
    cudaGetDeviceCount(&num_devices);

    if (device >= num_devices) {
        cout << "Error specifying GPU " << argv[1] << " that does not exist" << endl << endl;
        exit(1);
    }
    else
        cudaSetDevice(device);

    if (!read_binvox(argv[2])) {
        cout << "Error reading [" << argv[2] << "]" << endl << endl;
        exit(1);
    }

    vector<string> all_inputs = split(argv[2], "/");
    string bname = all_inputs.back();  // last element is file name "input.binvox"
    string bid = split(bname.c_str(), ".").front();    // first element is building id
    cout << "building file name " << bname << endl;
    cout << "building name " << bid << endl;


    imgsize = H * W;
    // https://stackoverflow.com/questions/2204176/how-to-initialise-memory-with-new-operator-in-c
    img = new byte[imgsize]();  // special syntax to initialize things to zero

    //------------------------------------ CPU version of code --------------------------------------//


    //------------------------------------- CUDA code starts ----------------------------------------//
    byte * d_v, * d_i;  // voxels and images on device

    cudaMalloc(& d_v, size * sizeof(byte));
    cudaMalloc(& d_i, imgsize * sizeof(byte));

    cudaMemcpy(d_v, voxels, size * sizeof(byte), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i, img, imgsize * sizeof(byte), cudaMemcpyHostToDevice);

    // assigning CUDA blocks and dimensions
    dim3 blocksPerGrid(8, 8, 1);
    dim3 threadsPerBlock(32, 32, 1); // 1024 threads per block and square, single output channel

    // start kernel
    heightsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_v, d_i);

    // check errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        cout << "Error: " << cudaGetErrorString(err) << endl;
        exit(-1);
    }

    // copy results from device to host
    cudaMemcpy(img, d_i, imgsize * sizeof(byte), cudaMemcpyDeviceToHost);

    //------------------------------------- CUDA code finish ----------------------------------------//

    cv::Mat image = cv::Mat(H, W, CV_8UC1, img);
    // string imageName = "/media/yuqiong/DATA/3dCityGan/playground/nyc_hmaps/" + bid + ".jpg";
    string imageName = "/data/gmldata/nyc_hmaps/" + bid + ".jpg";
    // zero check
    if (countNonZero(image) < 1){
        cout << "Error binvox outputs zero matrix" << endl;
        std::ofstream log;
        log.open("empty_log.txt", std::ios_base::app);
        log << imageName << endl;
        log.close();
        exit(-1);
    }

    cv::imwrite(imageName, image);
    // cv:namedWindow(imageName, cv::WINDOW_AUTOSIZE);
    // cv::imshow(imageName, image);
    // cv::waitKey(0);

    //
    // now write the meta data to as ASCII
    //
    // string metaName = "/media/yuqiong/DATA/3dCityGan/playground/nyc_hmaps_meta/" + bid + ".txt";
    string metaName = "/data/gmldata/nyc_hmaps_meta/" + bid + ".txt";
    ofstream *out = new ofstream(metaName.c_str());
    if(!out->good()) {
        cout << "Error opening [" << metaName << "]" << endl << endl;
        exit(1);
    }

    // cout << "Writing meta data to ASCII file..." << endl;

    *out << "#binvox ASCII data" << endl;
    *out << bid << endl;
    *out << "dim " << D << " " << H << " " << W << endl;
    *out << "translate " << tx << " " << ty << " " << tz << endl;
    *out << "scale " << scale << endl;
    out->close();

    // cout << "done" << endl << endl;

    //------------------------------------- Min bounding box code start ----------------------------------------//
    blur( image, image, Size(3,3) );
    crop_min_box(image, bid);
    /*
    // Just some code to check if the revert is successful
    Mat res = crop_min_box(image, bid);
    cv:namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::imshow("test", res);
    cv::waitKey(0);
    string revertName = "/media/yuqiong/DATA/3dCityGan/playground/nyc_hmaps_revert/" + bid + ".xml";
    Mat rev = revert(res, revertName);
    namedWindow("Revert", WINDOW_AUTOSIZE);
    imshow("Revert", rev);
    waitKey(0);
     */
    return 0;
}


// GPU version of finding heights
__global__ void heightsKernel(byte * voxels, byte * heights){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;  // row
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;  // column

    // every thread is in charge of looping through 255 values and find the last non-zero entry
    for (int z=255; z>=0; z--) {
        if (voxels[binvoxIndex3(z, x, y, H, W)] == 1){
            heights[index2(x, y, W)] = 255 - z;
            break;
        }
    }
}
