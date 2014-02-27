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
    int sum;
} kernelBlock;

/// Function header
void threshBlock(int, void* );
void findPath(Mat);
kernelBlock* findPath(Mat, int, int);
double getPSNR(Mat, Mat);

/** @function main */
int main( int argc, char** argv )
{
    int deviceId = atoi(argv[1]);
    Mat prevFrame , newFrame;//store prev and new frame

    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    
    cap >> prevFrame;
    for(;;) {
        cap >> src;
        // Clone the current frame:
        Mat frame = src.clone();
        newFrame = src.clone();

        double diffInFrames = getPSNR(prevFrame, newFrame);

        if(diffInFrames < 18.64){
            prevFrame = src.clone();
            /// Convert image to gray and blur it
            cvtColor( frame, src_gray, CV_BGR2GRAY );
            blur( src_gray, src_gray, Size(3,3) );

            threshBlock(0, 0);
            findPath(des);
        }

        imshow("path", src);
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return(0);
}

/** @function threshBlock */
void threshBlock(int, void* )
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

/* @function findPath - wrapper*/
/* finds path in three portions of image */
void findPath(Mat dst){
    kernelBlock* minLeft;
    kernelBlock* minCentre;
    kernelBlock* minRight;

    int partLength = dst.cols / 3;
    int blockLength = (4 * dst.cols) / 13;
    int start = 0;
    int end = partLength + (blockLength / 2);

    minLeft = findPath(dst, start, end);
    
    start = end - blockLength;
    end = end + partLength;
    minCentre = findPath(dst, start, end );

    start = end - blockLength;
    end = dst.cols;
    minRight= findPath(dst, start, end );

    double diff12 = ( minLeft->startx + minLeft->length ) - minCentre->startx;
    double diff23 = ( minCentre->startx + minCentre->length ) - minRight->startx ;

    int x, y, len;
    y = minCentre->starty;

    if(diff23  > diff12){
        x = ( ( minCentre->startx + minCentre->length ) + minRight->startx ) / 2 ;
    }
    else{
        x = ( ( minLeft->startx + minLeft->length ) +  minCentre->startx ) / 2;
    }

    x = x - ( dst.cols / 12 );
    rectangle(src, Point(x, y), Point( x + ( dst.cols / 6 ), src.rows - 2), Scalar(255, 0, 0), 5, 8, 0);
}

/* @function findPath */

kernelBlock* findPath(Mat dst, int xstart, int xend){ 
    int length = (4 * dst.cols) / 13;
    int width = (1 * dst.rows) / 3;
    int sumLowest = INT_MAX, sum;

    kernelBlock *current = new kernelBlock();
    addDataKernel(current, xstart, dst.rows - width - 1, length, width);
    kernelBlock *min = new kernelBlock();
    addDataKernel(min, xstart, dst.rows - width - 1, length, width);
    kernelBlock *rem = NULL, *add = NULL;

    //find sum of starting kernelBlock
    sum = sumRange(dst, current);
    min->sum = sum;
    current->sum = sum;

    while(1){

        if(rem == NULL)
            rem = new kernelBlock();

        addDataKernel(rem, current->startx , current->starty, 2, width);
        
        if(add == NULL)
            add = new kernelBlock();

        addDataKernel(add, current->startx + length , current->starty , 2, width);

        //breaking condition
        if( ( add->startx + 2 ) >= (xend))
            break;

        sum = sum - sumRange(dst, rem) + sumRange(dst, add) ;

        if(sum < sumLowest){
            sumLowest = sum;
            min->startx = current->startx + 2;
            min->sum = sumLowest;
        }    
        
        current->startx += 2;
    
    }         

    rectangle(src, Point(min->startx, min->starty), Point(min->startx + min->length, src.rows - 2), Scalar(0,255,0), 5, 8, 0);
    
    delete current, add, rem;
    return min;
}

double getPSNR(Mat I1, Mat I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else{
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}