/*
 * test_slic.cpp.
 *
 * Written by: Pascal Mettes.
 *
 * This file creates an over-segmentation of a provided image based on the SLIC
 * superpixel algorithm, as implemented in slic.h and slic.cpp.
 */
 
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
using namespace std;

#include <opencv2/opencv.hpp>
#include "slic.h"
using namespace cv;


int main(int argc, char *argv[]) {
    /* Load the image and convert to Lab colour space. */
    Mat image = imread("dog.png", 1);
    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);
    
    /* Yield the number of superpixels and weight-factors from the user. */
    int w = image.cols, h = image.rows;
    int nr_superpixels = 400;
    int nc = 40;

    double step = sqrt((w * h) / (double) nr_superpixels);

    /* Perform the SLIC superpixel algorithm. */
    Slic slic;
    slic.generate_superpixels(lab_image, int(step), nc);
    slic.create_connectivity(lab_image);
    slic.display_contours(image, Vec3b(0,0,255));
    
    /* Display the contours and show the result. */
    imshow("result", image);
    waitKey(0);
}
