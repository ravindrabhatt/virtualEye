#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

using namespace cv;
using namespace std;

int threshold_value = 100;
int threshold_type = 0;
int const max_BINARY_value = 255;
Mat src, frame; 
Mat src_gray;
Mat des;
int thresh1 = 66, thresh2 = 46;
int max_thresh = 255;
RNG rng(12345);
//storage for path area
typedef struct  kernelBlock{
    int startx;
    int starty;
    int length;
    int width;
} kernelBlock;

/// Function header
void thresh_callback(int, void* );
void findPath(Mat, int);

/** @function main */
int main( int argc, char** argv )
{
    int deviceId = atoi(argv[1]);

    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }

    for(;;) {
        cap >> src;
        // Clone the current frame:
        Mat frame = src.clone();
        /// Convert image to gray and blur it
        cvtColor( frame, src_gray, CV_BGR2GRAY );
        blur( src_gray, src_gray, Size(3,3) );

        thresh_callback( 0, 0 );
        findPath(des, 3);
        imshow("path", src);
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return(0);
}

/** @function thresh_callback */
void thresh_callback(int, void* )
{
  Mat canny_output, dst;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  /// Detect edges using canny 118 127// 161 18
  Canny( src_gray, canny_output, 161, 18, 3 );
  /// Find contours
  findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

  /// Draw contours
  Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    }

    //convert to gray
    cvtColor(drawing, dst, CV_RGB2GRAY);

    //threshold to get only W/B
    threshold(dst, drawing, threshold_value, max_BINARY_value,threshold_type);

    //store final result
    des = drawing.clone();
    imshow("edges", des);
  
}
 
//add pixel values in given range
int sumRange(Mat dst, kernelBlock *block){
    int sum = 0;
    for(int i = block->startx; i < block->startx + block->length; i++){
        for(int j = block->starty; j < dst.rows; j++){
            sum += dst.at<uchar>(j, i);
        }
    }
    return sum; 
}

//insert block properties
void addDataKernel(kernelBlock *block, int sx, int sy, int len, int wid){
    block->startx = sx;
    block->starty = sy;
    block->length = len;
    block->width = wid;
}

/* @function findPath */

void findPath(Mat dst, int passes){ 
    int length = (4 * dst.cols) / 13;
    int width = (1 * dst.rows) / 3;
    int sumLowest = INT_MAX, sum;

    kernelBlock *current = new kernelBlock();
    addDataKernel(current, 0, dst.rows - width - 1, length, width);
    kernelBlock *min = new kernelBlock();
    addDataKernel(min, 0, dst.rows - width - 1, length, width);
    kernelBlock *rem = NULL, *add = NULL;

    //find sum of starting kernelBlock
    sum = sumRange(dst, current);
    
    while(1){

        if(rem == NULL)
            rem = new kernelBlock();

        addDataKernel(rem, current->startx , current->starty, 2, width);
        
        if(add == NULL)
            add = new kernelBlock();

        addDataKernel(add, current->startx + length , current->starty , 2, width);

        //breaking condition
        if( ( add->startx + 2 ) >= (dst.cols))
            break;

        sum = sum - sumRange(dst, rem) + sumRange(dst, add) ;

        if(sum < sumLowest){
            sumLowest = sum;
            min->startx = current->startx + 2;
        }    
        
        current->startx += 2;
    
    }         

    rectangle(src, Point(min->startx, min->starty), Point(min->startx + min->length, src.rows - 2), Scalar(0,255,0), 5, 8, 0);
    
    delete min, current, add, rem;
}