#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>
#include "slic.h"

using namespace std;
using namespace cv;

const char* keys =
{
    "{help h usage ? |      | print this message   }"
    "{@input         |      | input image path     }"
    "{@superpixels   |      | number of superpixels}"
    "{@nc            |      | compactness factor   }"
    "{@output        |      | output image path    }"
};

int main(int argc, char* argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about("SLIC Superpixel Segmentation");

    if (parser.has("help") || argc == 1) {
        parser.printMessage();
        return 0;
    }

    String inputPath = parser.get<String>("@input");
    int nr_superpixels = parser.get<int>("@superpixels");
    int nc = parser.get<int>("@nc");
    String outputPath = parser.get<String>("@output");

    if (!parser.check()) {
        parser.printErrors();
        return -1;
    }

    // Load the image and convert to Lab colour space.
    Mat image = imread(inputPath, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Error: Could not read the image." << endl;
        return -1;
    }

    Mat lab_image;
    cvtColor(image, lab_image, COLOR_BGR2Lab);

    // Calculate step size
    int w = image.cols, h = image.rows;
    double step = sqrt((w * h) / (double)nr_superpixels);

    // Perform the SLIC superpixel algorithm.
    Slic slic;
    slic.generate_superpixels(lab_image, step, nc);
    slic.create_connectivity(lab_image);

    // Display the contours and show the result.
    slic.display_contours(image, Scalar(255, 0, 0));

    imshow("result", image);
    waitKey(0);

    // Save the result
    if (!imwrite(outputPath, image)) {
        cerr << "Error: Could not save the output image." << endl;
        return -1;
    }

    cout << "Segmentation complete. Output saved to " << outputPath << endl;

    return 0;
}