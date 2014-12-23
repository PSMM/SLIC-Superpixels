#include "slic.h"


/*
 * Constructor. Nothing is done here.
 */
Slic::Slic() {

}

/*
 * Destructor. Clear any present data.
 */
Slic::~Slic() {
    clear_data();
}

/*
 * Clear the data as saved by the algorithm.
 *
 * Input : -
 * Output: -
 */
void Slic::clear_data() {
    clusters.release();
    distances.release();
    centers.release();
    center_counts.clear();
}

/*
 * Initialize the cluster centers and initial values of the pixel-wise cluster
 * assignment and distance values.
 *
 * Input : The image (cv::Mat).
 * Output: -
 */
void Slic::init_data(const cv::Mat &image) {
    /* Initialize the cluster and distance matrices. */

    clusters  = cv::Mat_<int>(image.cols,image.rows,-1);
    distances = cv::Mat_<double>(image.cols,image.rows,DBL_MAX);

    /* Initialize the centers and counters. */
    for (int i = step; i < image.cols - step/2; i += step) {
        for (int j = step; j < image.rows - step/2; j += step) {
            /* Find the local minimum (gradient-wise). */
            cv::Point nc = find_local_minimum(image, cv::Point(i,j));
            cv::Vec3b colour = image.at<cv::Vec3b>(nc.y, nc.x);
            
            /* Generate the center vector. */
            Vec5d center(colour[0], colour[1], colour[2], nc.x, nc.y);
            
            /* Append to vector of centers. */
            centers.push_back(center);
            center_counts.push_back(0);
        }
    }
}

/*
 * Compute the distance between a cluster center and an individual pixel.
 *
 * Input : The cluster index (int), the pixel (cv::Point), and the Lab values of
 *         the pixel (cv::Scalar).
 * Output: The distance (double).
 */
double Slic::compute_dist(int ci, cv::Point pixel, cv::Vec3b colour) {
    Vec5d cen(centers(ci));
    double dc = sqrt(pow(cen[0] - colour[0], 2) + pow(cen[1]
            - colour[1], 2) + pow(cen[2] - colour[2], 2));
    double ds = sqrt(pow(cen[3] - pixel.x, 2) + pow(cen[4] - pixel.y, 2));
    
    return sqrt(pow(dc / nc, 2) + pow(ds / ns, 2));
    
    //double w = 1.0 / (pow(ns / nc, 2));
    //return sqrt(dc) + sqrt(ds * w);
}

/*
 * Find a local gradient minimum of a pixel in a 3x3 neighbourhood. This
 * method is called upon initialization of the cluster centers.
 *
 * Input : The image (cv::Mat &) and the pixel center (cv::Point).
 * Output: The local gradient minimum (cv::Point).
 */
cv::Point Slic::find_local_minimum(const cv::Mat_<cv::Vec3b> &image, cv::Point center) {
    double min_grad = DBL_MAX;
    cv::Point loc_min(center.x, center.y);
    
    for (int i = center.x-1; i < center.x+2; i++) {
        for (int j = center.y-1; j < center.y+2; j++) {
            cv::Vec3b c1 = image(j+1, i);
            cv::Vec3b c2 = image(j, i+1);
            cv::Vec3b c3 = image(j, i);
            /* Convert colour values to grayscale values. */
            double i1 = c1[0];
            double i2 = c2[0];
            double i3 = c3[0];
            /*double i1 = c1.val[0] * 0.11 + c1.val[1] * 0.59 + c1.val[2] * 0.3;
            double i2 = c2.val[0] * 0.11 + c2.val[1] * 0.59 + c2.val[2] * 0.3;
            double i3 = c3.val[0] * 0.11 + c3.val[1] * 0.59 + c3.val[2] * 0.3;*/
            
            /* Compute horizontal and vertical gradients and keep track of the
               minimum. */
            if (sqrt(pow(i1 - i3, 2)) + sqrt(pow(i2 - i3,2)) < min_grad) {
                min_grad = fabs(i1 - i3) + fabs(i2 - i3);
                loc_min.x = i;
                loc_min.y = j;
            }
        }
    }
    
    return loc_min;
}

/*
 * Compute the over-segmentation based on the step-size and relative weighting
 * of the pixel and colour values.
 *
 * Input : The Lab image (cv::Mat), the stepsize (int), and the weight (int).
 * Output: -
 */
void Slic::generate_superpixels(const cv::Mat &img, int step, int nc) {
    this->step = step;
    this->nc = nc;
    this->ns = step;

    /* make a new Mat header, that allows us to iterate the image more efficiently. */
    cv::Mat_<cv::Vec3b> image(img);

    /* Clear previous data (if any), and re-initialize it. */
    clear_data();
    init_data(image);
    
    /* Run EM for 10 iterations (as prescribed by the algorithm). */
    for (int i = 0; i < NR_ITERATIONS; i++) {
        /* Reset distance values. */
        distances = FLT_MAX;

        for (int j = 0; j < centers.rows; j++) {
            Vec5d cen(centers(j));
            /* Only compare to pixels in a 2 x step by 2 x step region. */
            for (int k = int(cen[3]) - step; k < int(cen[3]) + step; k++) {
                for (int l = int(cen[4]) - step; l < int(cen[4]) + step; l++) {
                
                    if (k >= 0 && k < image.cols && l >= 0 && l < image.rows) {
                        cv::Vec3b colour = image(l, k);
                        double d = compute_dist(j, cv::Point(k,l), colour);
                        
                        /* Update cluster allocation if the cluster minimizes the
                           distance. */
                        if (d < distances(k,l)) {
                            distances(k,l) = d;
                            clusters(k,l) = j;
                        }
                    }
                }
            }
        }
        
        /* Clear the center values. */
        for (int j = 0; j < centers.rows; j++) {
            centers(j) = 0;
            center_counts[j] = 0;
        }
        
        /* Compute the new cluster centers. */
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < image.rows; k++) {
                int c_id = clusters(j,k);
                
                if (c_id != -1) {
                    cv::Vec3b colour = image(k, j);                    
                    centers(c_id) += Vec5d(colour[0], colour[1], colour[2], j, k);                    
                    center_counts[c_id] += 1;
                }
            }
        }

        /* Normalize the clusters. */
        for (int j = 0; j < centers.rows; j++) {
            centers(j) /= center_counts[j];
        }
    }
}

/*
 * Enforce connectivity of the superpixels. This part is not actively discussed
 * in the paper, but forms an active part of the implementation of the authors
 * of the paper.
 *
 * Input : The image (cv::Mat).
 * Output: -
 */
void Slic::create_connectivity(const cv::Mat &image) {
    int label = 0, adjlabel = 0;
    const int lims = (image.cols * image.rows) / (centers.rows);
    
    const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};
    
    /* Initialize the new cluster matrix. */
    cv::Mat_<int> new_clusters(image.cols,image.rows,-1);

    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            if (new_clusters(i,j) == -1) {
                vector<cv::Point> elements;
                elements.push_back(cv::Point(i, j));
            
                /* Find an adjacent label, for possible use later. */
                for (int k = 0; k < 4; k++) {
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    
                    if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                        if (new_clusters(x,y) >= 0) {
                            adjlabel = new_clusters(x,y);
                        }
                    }
                }
                
                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        
                        if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                            if (new_clusters(x,y) == -1 && clusters(i,j) == clusters(x,y)) {
                                elements.push_back(cv::Point(x, y));
                                new_clusters(x,y) = label;
                                count += 1;
                            }
                        }
                    }
                }
                
                /* Use the earlier found adjacent label if a segment size is
                   smaller than a limit. */
                if (count <= lims >> 2) {
                    for (int c = 0; c < count; c++) {
                        new_clusters(elements[c].x, elements[c].y) = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
    clusters = new_clusters;
}

/*
 * Display the cluster centers.
 *
 * Input : The image to display upon (cv::Mat) and the colour (cv::Vec3b).
 * Output: -
 */
void Slic::display_center_grid(cv::Mat &image, cv::Scalar colour) {
    for (int i = 0; i < centers.rows; i++) {
        cv::circle(image, cv::Point2d(centers(i)[3], centers(i)[4]), 2, colour, 2);
    }
}

/*
 * Display a single pixel wide contour around the clusters.
 *
 * Input : The target image (cv::Mat) and contour colour (cv::Vec3b).
 * Output: -
 */
void Slic::display_contours(cv::Mat &image, cv::Vec3b colour) {
    const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	/* Initialize the contour vector and the matrix detailing whether a pixel
	 * is already taken to be a contour. */
	vector<cv::Point> contours;
    cv::Mat_<uchar> istaken(image.cols, image.rows, uchar(0));
    
    /* Go through all the pixels. */
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            int nr_p = 0;
            
            /* Compare the pixel to its 8 neighbours. */
            for (int k = 0; k < 8; k++) {
                int x = i + dx8[k], y = j + dy8[k];
                
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    if (istaken(x,y) == false && clusters(i,j) != clusters(x,y)) {
                        nr_p += 1;
                    }
                }
            }
            
            /* Add the pixel to the contour list if desired. */
            if (nr_p >= 2) {
                contours.push_back(cv::Point(i,j));
                istaken(i,j) = true;
            }
        }
    }
    
    /* Draw the contour pixels. */
    for (int i = 0; i < (int)contours.size(); i++) {
        image.at<cv::Vec3b>(contours[i].y, contours[i].x) = colour;
    }
}

/*
 * Give the pixels of each cluster the same colour values. The specified colour
 * is the mean RGB colour per cluster.
 *
 * Input : The target image (cv::Mat).
 * Output: -
 */
void Slic::colour_with_cluster_means(cv::Mat &image) {
    vector<cv::Vec3b> colours(centers.rows);
    
    /* Gather the colour values per cluster. */
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            int index = clusters(i,j);
            colours[index] += image.at<cv::Vec3b>(j, i);
        }
    }
    
    /* Divide by the number of pixels per cluster to get the mean colour. */
    for (int i = 0; i < (int)colours.size(); i++) {
        colours[i] /= center_counts[i];
    }
    
    /* Fill in. */
    for (int i = 0; i < image.cols; i++) {
        for (int j = 0; j < image.rows; j++) {
            image.at<cv::Vec3b>(j, i) = colours[clusters(i,j)];;
        }
    }
}
