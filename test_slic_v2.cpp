
#include <iostream>
#include <vector>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "slic.h"

int main( int argc, char ** argv )
{
    cout << "====\n\t" << argv[0] << "\n====\n";

    // Load Image
    cv::Mat image = cv::imread( "../dog.png", CV_LOAD_IMAGE_COLOR );
    cout << "Image Dimensions: " << image.rows << " x " << image.cols << " x " << image.channels() << endl;
    cv::imshow( "input image", image );


    cv::Mat lab_image;
    cv::cvtColor( image, lab_image, CV_BGR2Lab );

    // Params
    int w = image.cols, h = image.rows;
    int nr_superpixels = 400;
    int nc = 40;
    double step = sqrt((w * h) / (double) nr_superpixels); ///< step size per cluster
    cout << "Params:\n";
    cout << "step size per cluster: " << step << endl;
    cout << "Weight: " << nc << endl;
    cout << "Number of superpixel: "<< nr_superpixels << endl;



    Slic slic;
    IplImage __ipl__lab_image=lab_image;
    IplImage __ipl__image = image;
    slic.generate_superpixels(&__ipl__lab_image, step, nc);
    slic.create_connectivity(&__ipl__lab_image);

    slic.display_contours(&__ipl__image, CV_RGB(255,0,0));
    cv::imshow( "result", image );
    // cv::imwrite( "dog_segmentation_v2.png", image );

    cv::waitKey(0);
    return 0;
}
