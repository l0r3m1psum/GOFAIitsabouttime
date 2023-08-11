#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/softfloat.hpp"

// To enable se the following environment variable OPENCV_TRACE=1
#include "opencv2/core/utils/trace.hpp"

#include <iostream>

#define BREAKPOINT __builtin_debugtrap()

static void
show(const cv::Mat& img) {
	cv::imshow("visualization", img);
	cv::waitKey(0);
}

static std::ostream&
print_mat(const cv::Mat& mat) {
	return std::cout
		<< cv::typeToString(mat.type()) << '\n'
		<< mat << '\n';
}

static cv::softfloat
line_length(cv::Point2f a, cv::Point2f b) {
	// NOTE: should we use cv::normL2Sqr?
	cv::Point2f diff = a - b;
	cv::softfloat res(cv::sqrt(diff.x*diff.x + diff.y*diff.y));
	return res;
}

int
main(int argc, char **argv) {
	cv::Mat img = cv::imread("example_clock.jpg");

	if (img.empty()) {
		std::cerr << "unable to read image";
		return 1;
	}

	cv::Mat grayed(img.rows, img.cols, CV_8UC1);
	cv::cvtColor(img, grayed, cv::COLOR_BGR2GRAY);
	CV_Assert(grayed.size() == cv::Size(img.cols, img.rows));
	CV_Assert(grayed.type() == CV_8UC1);

	cv::Mat thresholded(img.rows, img.cols, CV_8UC1);
	{
		int lowerb = 0, upperb = 100;
		cv::inRange(grayed, lowerb, upperb, thresholded);
		CV_Assert(thresholded.size() == cv::Size(img.cols, img.rows));
		CV_Assert(thresholded.type() == CV_8UC1);
	}

	cv::Mat edged(img.rows, img.cols, CV_8UC1);
	{
		double threshold1 = 100, threshold2 = 200;
		int apertureSize = 3;
		bool L2gradient = false;
		cv::Canny(thresholded, edged, threshold1, threshold2, apertureSize, L2gradient);
		CV_Assert(edged.size() == cv::Size(img.cols, img.rows));
		CV_Assert(edged.type() == CV_8UC1);
	}

	cv::Mat lines;
	{
		CV_TRACE_REGION("test");
		double rho = 1, theta = CV_PI / 180;
		int threshold = 45;
		double minLineLenght = 45, maxLineGap = 100;
		cv::HoughLinesP(edged, lines, rho, theta, threshold, minLineLenght, maxLineGap);
		CV_Assert(lines.cols == 1);
		CV_Assert(lines.type() == CV_32SC4);
	}

	constexpr int K = 2;
	static unsigned char centers_data[CV_ELEM_SIZE(CV_32FC1)*K*4] = {};
	// NOTE: centers can be a MatX
	cv::Mat centers(K, 4, CV_32FC1, centers_data);
	{
		cv::Mat lines_float;
		lines.convertTo(lines_float, CV_32F);
		cv::Mat bestLabels(lines_float.rows, 1, CV_32S);
		cv::TermCriteria criteria(cv::TermCriteria::EPS
			+ cv::TermCriteria::MAX_ITER, 10, 1.0);
		int attempts = 10;
		double compactness = cv::kmeans(
			lines_float,
			K,
			bestLabels,
			criteria,
			attempts,
			cv::KMEANS_RANDOM_CENTERS,
			centers
		);
		CV_Assert(centers.size() == cv::Size(4, K));
		CV_Assert(centers.type() == CV_32FC1);
		CV_Assert(bestLabels.size() == cv::Size(1, lines_float.rows));
		CV_Assert(bestLabels.type() == CV_32S);
	}

	cv::Point2f min_p1 = centers.at<cv::Point2f>(0,0);
	cv::Point2f min_p2 = centers.at<cv::Point2f>(0,1);
	cv::Point2f hour_p1 = centers.at<cv::Point2f>(1,0);
	cv::Point2f hour_p2 = centers.at<cv::Point2f>(1,1);
	if (line_length(min_p1, min_p2) < line_length(hour_p1, hour_p2)) {
		cv::swap(min_p1, hour_p1);
		cv::swap(min_p2, hour_p2);
	}

	// Given that y = mx + q is the equation of the line, where m is the slope
	// and q is the intercept with the y-axis. If you want to find the slope of
	// a line passing between two point you can create a system of equation and
	// solving that will give you:
	//         y_2 - y_1
	//     m = ---------
	//         x_2 - x_1
	// Using the atan() function on m you find the slope of said line in
	// radiants. In our case we need atan2() to determine the quadrant of the
	// angle. We also have to fliy the Y axis of the points to transport them
	// from the "image space" (where y goes down) to the euclidean space (where
	// y goes up). We then have to find the tip ot the clock's hand i.e the
	// furthest from the intersection of the lines passing between the two
	// segments. Finally we have to the origin of the degrees from east to north
	// i.e. 90 degree counterclockwise.
	cv::Point2f intersec;
	{
		float x1 = min_p1.x, x2 = min_p2.x, x3 = hour_p1.x, x4 = hour_p2.x,
		      y1 = min_p1.y, y2 = min_p2.y, y3 = hour_p1.y, y4 = hour_p2.y;
		float denumerator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);
		float det_12 = x1*y2 - y1*x2, det_34 = x3*y4 - y3*x4;
		intersec.x = (det_12*(x3-x4) - (x1-x2)*det_34)/denumerator;
		intersec.y = (det_12*(y3-y4) - (y1-y2)*det_34)/denumerator;
	}
	if (line_length(intersec, min_p1) > line_length(intersec, min_p2)) {
		cv::swap(min_p1, min_p2);
	}
	if (line_length(intersec, hour_p1) > line_length(intersec, hour_p2)) {
		cv::swap(hour_p1, hour_p2);
	}
	// cv::fastAtan2 returns degrees not radiants.
	float min_angle = cv::fastAtan2((-min_p2.y) - (-min_p1.y), min_p2.x - min_p1.x);
	float hour_angle = cv::fastAtan2((-hour_p2.y) - (-hour_p1.y), hour_p2.x - hour_p1.x);
	// We rotate the the angle to be 0 at 12 o'clock i.e. by 90 degrees
	// counterclockwise.
	min_angle = fmod(90 - min_angle + 360, 360);
	hour_angle = fmod(90 - hour_angle + 360, 360);

	float minute_frac = min_angle/6, hour_frac = hour_angle/30;

	int hour = minute_frac >= 30 ? cvFloor(hour_frac) : cvRound(hour_frac);
	int minute = cvFloor(minute_frac);

	{
		cv::Mat viz = img.clone();
		cv::Scalar red(0, 0, 0xff), green(0, 0xff, 0), blue(0xff, 0, 0);
		for (int rowi = 0; rowi < lines.rows; ++rowi) {
			cv::Mat row = lines.row(rowi);
			cv::Point2f p1(row.at<int32_t>(0,0), row.at<int32_t>(0,1));
			cv::Point2f p2(row.at<int32_t>(0,2), row.at<int32_t>(0,3));
			cv::line(viz, p1, p2, red);
		}
		cv::arrowedLine(viz, min_p1, min_p2, green);
		cv::arrowedLine(viz, hour_p1, hour_p2, green);
		char text_buf[16];
		snprintf(text_buf, sizeof text_buf, "%d:%d", hour, minute);
		cv::putText(viz,
			text_buf,
			cv::Point(10,40),
			cv::FONT_HERSHEY_PLAIN,
			3, // scale
			green,
			3, // thickness
			cv::FILLED,
			false);
		cv::drawMarker(viz, intersec, blue);
		cv::imshow("visualization", viz);
		(void) cv::pollKey(); // Forces the window to show up.
	}

	// TODO: handle eventual second hand (doing k-means with k=3 and discarding
	// the longest hand.)

	BREAKPOINT;

	return 0;
}
