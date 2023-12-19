// https://learnopencv.com/blob-detection-using-opencv-python-c/
// https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gaf259efaad93098103d6c27b9e4900ffa
// https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-bmt.2017.0161
// https://en.wikipedia.org/wiki/Random_sample_consensus
// http://iust-projects.ir/post/dip06/
// https://arxiv.org/pdf/1503.07460.pdf
// https://towardsdatascience.com/a-bonsai-and-an-ellipse-f68c78dbacb8

/* Ideona: smanettare con i parametri di blob detector finché non trovo
 * abbastanza punti sul contorno dell'orologio. A quel punto fare RANSAC con
 * ellisse.
 */

#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <thread>

cv::String window_name = "keypoints";
cv::Mat im, im_with_keypoints;
std::vector<cv::KeyPoint> keypoints;
cv::Ptr<cv::SimpleBlobDetector> detector;
cv::SimpleBlobDetector::Params params;

std::atomic<bool> should_update_img, should_update_window;

static std::ostream&
operator <<(std::ostream& os, const cv::SimpleBlobDetector::Params& params) {
#define print(name) "\t." << #name << " = " << params.name << ",\n"
	return os
		<< std::boolalpha
		<< "{\n"
		<< "\t." << "blobColor" << " = " << (int) params.blobColor << ",\n"
		<< print(collectContours)
		<< print(filterByArea)
		<< print(filterByCircularity)
		<< print(filterByColor)
		<< print(filterByConvexity)
		<< print(filterByInertia)
		<< print(maxArea)
		<< print(maxCircularity)
		<< print(maxConvexity)
		<< print(maxInertiaRatio)
		<< print(maxThreshold)
		<< print(minArea)
		<< print(minCircularity)
		<< print(minConvexity)
		<< print(minDistBetweenBlobs)
		<< print(minInertiaRatio)
		<< print(minRepeatability)
		<< print(minThreshold)
		<< print(thresholdStep)
		<< '}'
		<< std::noboolalpha;
#undef print
}

template<typename T>
void callback(int pos, void *userdata) {
	// TODO: input validation (of pose) based on T.
	// std::is_same<T,bool>::value
	// std::is_same<T,float>::value
	// std::is_same<T,size_t>::value
	T *field_to_change = (T *) userdata;
	*field_to_change = pos;
	try {
		detector->setParams(params);
	} catch (const cv::Exception& ex) {
		if (ex.code == cv::Error::Code::StsBadArg) {
			std::cerr << "bad arg detected!\n";
		}
	}
	should_update_img.store(true);
}

void update_img() {
	while (true) {
		if (!should_update_img.load()) {
			std::this_thread::yield();
			continue;
		}
		// Should I clear the vector to avoid leaks?
		detector->detect(im, keypoints);
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the
		// circle corresponds to the size of blob.
		// The keypoint that I'm interested in should be roughly cente
		drawKeypoints(im, keypoints, im_with_keypoints, cv::Scalar(0,0,255),
					  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		should_update_img.store(false);
		should_update_window.store(true);
		std::cout << "update done\n";
	}
}

int main() {
	im = cv::imread("itsabouttime/data/coco/000000000064.jpg", cv::IMREAD_GRAYSCALE);
	//               x    y   width height
	im = im(cv::Rect(100, 30, 200,  200)).clone();
	if (im.empty()) {
		std::cerr << "could not open the image\n";
		return 1;
	}

	std::thread updater(update_img);
	updater.detach();

	// To create the window.
	detector = cv::SimpleBlobDetector::create();
	callback<bool>(0, &params.collectContours);
	while (!should_update_window.load()) {
		std::this_thread::yield();
	}
	cv::imshow(window_name, im_with_keypoints);
	(void) cv::pollKey();
	should_update_window.store(false);

	std::cout << params << '\n';

	// bool callback should just be buttons but it is a QT only thing...
#define CREATE_TRACKBAR(name, count, T) \
cv::createTrackbar(#name, window_name, nullptr, count, callback<T>, &params.name);
// cv::setTrackbarPos(#name, window_name, std::min((int) params.name, count));

	CREATE_TRACKBAR(collectContours, 1, bool);
	CREATE_TRACKBAR(filterByColor,   1, bool);

	CREATE_TRACKBAR(filterByArea, 1, bool);
	CREATE_TRACKBAR(minArea,     im.rows*im.cols, float); cv::setTrackbarMin("minArea", window_name, 1);
	CREATE_TRACKBAR(maxArea,     im.rows*im.cols, float);

	CREATE_TRACKBAR(filterByCircularity, 1,  bool);
	CREATE_TRACKBAR(minCircularity,     10,  float); cv::setTrackbarMin("minCircularity", window_name, 1);
	CREATE_TRACKBAR(maxCircularity,     10, float);

	CREATE_TRACKBAR(filterByConvexity, 1, bool);
	CREATE_TRACKBAR(minConvexity,     10, float); cv::setTrackbarMin("minCircularity", window_name, 1);
	CREATE_TRACKBAR(maxConvexity,     10, float);

	CREATE_TRACKBAR(filterByInertia,  1, bool);
	CREATE_TRACKBAR(minInertiaRatio, 10, float); cv::setTrackbarMin("minInertiaRatio", window_name, 1);
	CREATE_TRACKBAR(maxInertiaRatio, 10, float);

	CREATE_TRACKBAR(thresholdStep, 10, float);
	CREATE_TRACKBAR(minThreshold,  10, float);
	CREATE_TRACKBAR(maxThreshold,  10, float);

	CREATE_TRACKBAR(minDistBetweenBlobs, 10, float); cv::setTrackbarMin("minDistBetweenBlobs", window_name, 1);
	CREATE_TRACKBAR(minRepeatability,    10, size_t); cv::setTrackbarMin("minRepeatability", window_name, 1);
#undef CREATE_TRACKBAR

	// TODO: add key to dump the parameters.

	while (cv::pollKey() != 'q') {
		if (should_update_window.load()) {
			cv::imshow(window_name, im_with_keypoints);
			should_update_window.store(false);
		}
		std::this_thread::yield();
	}

	return 0;
}
