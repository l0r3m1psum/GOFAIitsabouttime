#include "opencv2/core/utils/logger.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define BREAKPOINT __builtin_debugtrap()
// An dereferentiable array litteral.
#define A(T, ...) (std::initializer_list<T>({__VA_ARGS__}).begin())
// template<typename T> using alias = T; // TODO: use this instead.

// $ sysctl hw.cachelinesize
// hw.cachelinesize: 64
constexpr int cache_line_size = 64;
constexpr int seed = 42;

static double
line_length(cv::Point2f a, cv::Point2f b) {
	cv::Point2f diff = a - b;
	return cv::norm(diff);
}

static cv::Mat
copyTo(cv::Mat src, cv::Mat dst) {
	src.copyTo(dst);
	return dst;
}

static void
putText(cv::Mat img, const cv::String &text, cv::Point org) {
	cv::HersheyFonts fontFace = cv::FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	cv::Scalar color(0x00, 0xff, 0x00); // green
	int thickness = 1;
	cv::LineTypes lineType = cv::FILLED;
	bool bottomLeftOrigin = false;
	cv::putText(img, text, org, fontFace, fontScale, color, thickness, lineType,
		bottomLeftOrigin);
}

// TODO: this function should be able to visualize a list of list of images
// to switch between them we can use up and dowon arrow. We have to remember the
// posizion of the trackbar and change the dimension according to de dataset.
static bool
explore_dataset(
	const cv::String &window_name,
	const std::vector<cv::Mat>& imgs) {
	CV_Assert(imgs.size() > 0);

	cv::namedWindow(window_name, cv::WINDOW_NORMAL);

	int index_max = imgs.size() - 1; // NOTE: this is a conversion from size_t to int
	struct name_and_imgs {
		const char *window_name;
		const std::vector<cv::Mat> &imgs;
	} data = {window_name.c_str(), imgs};
	auto update_img = [](int pos, void *userdata) {
		name_and_imgs *data = (name_and_imgs *) userdata;
		cv::imshow(data->window_name, data->imgs[pos]);
	};
	cv::createTrackbar("index", window_name, nullptr, index_max, update_img, &data);

	cv::imshow(window_name, imgs[0]);
	for (;;) {
		int res;
		switch (res = cv::pollKey()) {
		case -1: break; // If no key was pressed.
		case 'q': return false; // Quit.
		case 'r': cv::destroyWindow(window_name); return true; // Reload.
		case 63235: {
			/* right arrow */
			int pos = cv::getTrackbarPos("index", window_name);
			if (pos < index_max) {
				cv::setTrackbarPos("index", window_name, pos + 1);
			}
		}; break;
		case 63234: {
			/* left arrow */
			int pos = cv::getTrackbarPos("index", window_name);
			if (pos > 0) {
				cv::setTrackbarPos("index", window_name, pos - 1);
			}
		}; break;
		}
	}
	
	// TODO: destroy window.
}

// Debugging utilities. ////////////////////////////////////////////////////////

// TODO: it would be better to use a hashmap and iterate ofer it to have a
// consistend order. The best thing would be to hash the ids.
struct DebugVec {
	cv::Mutex mutex;
	std::vector<cv::Mat> mats;

	void append(cv::Mat mat) {
		mutex.lock();
		mats.push_back(mat);
		mutex.unlock();
	}
	void append(std::vector<cv::Mat> &stages) {
		cv::Mat tmp;
		for (cv::Mat &stage : stages) {
			if (stage.type() == CV_8UC1) {
				cvtColor(copyTo(stage, tmp), stage, cv::COLOR_GRAY2BGR);
			}
			CV_Assert(stage.type() == CV_8UC3);
		}
		cv::hconcat(stages, tmp);
		append(tmp);
	}
};

static DebugVec skipped_imgs;
static DebugVec wrong_imgs;

struct ParallelClassification CV_FINAL : public cv::ParallelLoopBody {

	struct alignas(cache_line_size) Size {
		size_t size;
		Size& operator++() {
			size++;
			return *this;
		};
	};

	const std::vector<cv::Mat>& imgs;
	const std::vector<int16_t>& labels;
	std::vector<Size> &correct;
	std::vector<Size> &skipped;
	// parameters
	int mask_radius_scale;
	double canny_threshold1;
	double canny_threshold2;
	int connected_component_area;
	int hough_lines_threshold;
	double hough_lines_minLineLength;
	double hough_lines_maxLineGap;

	ParallelClassification(
		const std::vector<cv::Mat>& imgs,
		const std::vector<int16_t>& labels,
		std::vector<Size> &correct,
		std::vector<Size> &skipped,
		int mask_radius_scale,
		double canny_threshold1,
		double canny_threshold2,
		int connected_component_area,
		int hough_lines_threshold,
		double hough_lines_minLineLength,
		double hough_lines_maxLineGap
	) : imgs(imgs), labels(labels), correct(correct), skipped(skipped),
		mask_radius_scale(mask_radius_scale),
		canny_threshold1(canny_threshold1),
		canny_threshold2(canny_threshold2),
		connected_component_area(connected_component_area),
		hough_lines_threshold(hough_lines_threshold),
		hough_lines_minLineLength(hough_lines_minLineLength),
		hough_lines_maxLineGap(hough_lines_maxLineGap) {
		CV_Assert(imgs.size() == labels.size());
		CV_Assert(correct.size() == skipped.size());
		CV_Assert(skipped.size() == cv::getNumThreads());
	}

	virtual void operator() (const cv::Range& range) const CV_OVERRIDE {
		cv::theRNG().state = seed;
		cv::setRNGSeed(seed);
		cv::Mat copied; // For temporary copies needed by the various filters.
		cv::Size standardized_size(128, 128);
		cv::Mat annulus_mask(standardized_size, CV_8UC1, cv::Scalar(0));
		{
			// If you torture it enough...
			cv::Mat img; cv::Point center; cv::Scalar color; cv::LineTypes lineType;
			int radius, thickness, shift;
			
			cv::Mat outer_circle = annulus_mask.clone();
			outer_circle = 255;
			cv::Mat inner_circle = annulus_mask.clone();
			// TODO: this radius has to be changed.
			cv::circle(img = inner_circle, center = standardized_size/2,
					   radius = center.x/2, color = cv::Scalar(255), thickness = -1,
					   lineType = cv::LINE_8, shift = 0);
			
			cv::subtract(outer_circle, inner_circle, annulus_mask);
		}
		for (int r = range.start; r < range.end; ++r) {
			std::vector<cv::Mat> stages; // Records the transformation stages for debugging.
			int thread_num = cv::getThreadNum();
			cv::Mat img = imgs[r];

			// We standardize the size of the image to a resolution that it is
			// good enough to allow us to distinguish features and small enough
			// to discard useless features/noise. This also allows us to not
			// calculate parameters to the other algorithms as a function of the
			// image resolution (e.g. we can use a fixed bluring kernel instead
			// of using one that grows with the size of the image). Also if an
			// image is too small this gives us more pizels to work with.
			{
				double fx = 0, fy = 0;
				cv::InterpolationFlags interpolation =
					img.size().area() < standardized_size.area()
					? cv::INTER_CUBIC : cv::INTER_AREA;
				cv::resize(copyTo(img, copied), img, standardized_size, fx, fy,
					interpolation);
			}

			cv::Mat work_img(img.rows, img.cols, CV_8UC1);

			// We try multiple times to find enough lines (2) in the image with
			// a sequence of different parameters. Usually the clock with
			// thicker hands can tollerate a stronger denoising and a less
			// sensitive edge detection. The opposite is true for the clocks
			// with thinner hands.
			cv::Mat lines;
			bool enough_lines = false;
			for (int attempt = 0; attempt < 3; ++attempt) {
				stages.clear();
				stages.push_back(img);
				{
					cv::cvtColor(img, work_img, cv::COLOR_BGR2GRAY);
					CV_Assert(work_img.size() == img.size());
					CV_Assert(work_img.type() == CV_8UC1);
				}

				// High pass filter for sharpening the image and removing noise.
				{
					int ksize = A(int, 7, 5, 3)[attempt];
					cv::medianBlur(copyTo(work_img, copied), work_img, ksize);
					CV_Assert(work_img.size() == img.size());
					CV_Assert(work_img.type() == CV_8UC1);
					stages.push_back(work_img.clone());
				}

				// https://cvexplained.wordpress.com/tag/canny-edge-detector/
				{
					double threshold1 = A(double, canny_threshold1, canny_threshold1/2, canny_threshold1/4)[attempt],
						threshold2 = A(double, canny_threshold2, canny_threshold2/2, canny_threshold2/4)[attempt];
					// NOTE: should I optimize this two too?
					int apertureSize = 3;
					bool L2gradient = true;
					cv::Canny(copyTo(work_img, copied), work_img, threshold1,
						threshold2, apertureSize, L2gradient);
					CV_Assert(work_img.size() == img.size());
					CV_Assert(work_img.type() == CV_8UC1);
				}

				{
					cv::Mat labels, stats, centroids;
					int connectivity = 8, ltype = CV_16U, ccltype = cv::CCL_DEFAULT;
					int n_labels = cv::connectedComponentsWithStats(
						work_img, labels, stats, centroids,
						connectivity, ltype, ccltype
					);
					CV_Assert(stats.rows == n_labels);
					CV_Assert(stats.type() == CV_32S);
					CV_Assert(centroids.rows == n_labels);
					CV_Assert(centroids.type() == CV_64F);
					CV_Assert(labels.type() == ltype);

					// The first connected component should be the black
					// background of the image, that has to be discarded.
					int width = stats.at<int>(0, cv::CC_STAT_WIDTH);
					int height = stats.at<int>(0, cv::CC_STAT_HEIGHT);
					if (cv::Size(width, height) != standardized_size) {
						CV_LOG_ERROR(NULL, "bad background???");
						break;
					}
					
					// If the number of connected components is greater than the
					// possible hue values we treat this as an error. Even if
					// this is not technically an error the number of connected
					// components is still way too high for this kind of image.
					// NOTE: this chould be a feedback point.
					if (n_labels > 255) {
						CV_LOG_ERROR(NULL, "too many connected components");
						break;
					}
					
					// We create the visualization for debugging.
					{
						cv::Mat tmp;
						labels.convertTo(tmp, CV_8UC1);
						double min_generic = 0, max_generic = 0;
						cv::minMaxLoc(tmp, &min_generic, &max_generic);
						int max = (int) max_generic;
						cv::Mat hue = tmp * 255/max;
						cv::Mat saturation = (tmp > 0) * 255;
						cv::Mat value = saturation;
						cv::Mat color_tmp_chans[] = {hue, saturation, value};
						cv::Mat color_tmp;
						cv::merge(color_tmp_chans, 3, color_tmp);
						cvtColor(color_tmp, color_tmp, cv::COLOR_HSV2BGR);
						stages.push_back(color_tmp);
					}

					cv::Mat mask = cv::Mat::zeros(standardized_size, CV_8UC1);
#if 0
					// We remove all the "small" connected components.
					for (int i = 1; i < n_labels; ++i) {
						if (stats.at<int>(i, cv::CC_STAT_AREA) > connected_component_area)
							continue;
						cv::bitwise_or(labels == i, mask, mask);
					}
					cv::bitwise_not(mask, mask);
					cv::bitwise_and(work_img, mask, work_img);
#endif
					
					stages.push_back(annulus_mask);
#if 1
					mask = cv::Mat::zeros(standardized_size, CV_8UC1);
					for (int i = 1; i < n_labels; ++i) {
						cv::Mat intersection(standardized_size, CV_8UC1, cv::Scalar(0));
						cv::Mat filtered = labels == i;
						cv::bitwise_and(filtered, annulus_mask, intersection);
						float percentage_covered = (float) cv::countNonZero(intersection)
							/ cv::countNonZero(filtered);
						if (percentage_covered > .5) {
							cv::bitwise_or(filtered, mask, mask);
						}
					}
					stages.push_back(mask);
					{
						std::vector<cv::Mat> contours;
						cv::Mat hierarchy;
						cv::RetrievalModes mode = cv::RETR_LIST;
						cv::Point offset = cv::Point();
						cv::ContourApproximationModes method = cv::CHAIN_APPROX_SIMPLE;
						cv::findContours(mask, contours, hierarchy, mode, method, offset);

						std::vector<cv::RotatedRect> ellipses(contours.size());
						for (cv::Mat contour : contours) {
#if 0
							CV_Assert(contour.depth() == CV_32S);
							CV_Assert(contour.cols == 1);
							CV_Assert(contour.rows % 2 == 0);
							contour = contour.reshape(0, contour.rows/2);
							CV_Assert(contour.cols == 2);
							// Now we have a point per row.
#endif
							if (contour.rows < 5*2) {
								continue;
							}
							cv::RotatedRect ellipse = cv::fitEllipse(contour);
							// FIXME: this heuristic is shit. The ellipse found
							// should also be inside the image for the most part.
							if (ellipse.size.area() < standardized_size.area() * .8) {
								continue;
							}
							// TODO: here if too many ellipses are detected or
							// zero this is a good feed back point. The ideal
							// number should be one ellipse. If the number is
							// more than one the ellipses shuld all be near each
							// other (almost overlappint). If they are arranged
							// like this they can be fused with some sort of
							// averaging!
							ellipses.push_back(ellipse);
						}
						cv::Mat testerino = cv::Mat::zeros(standardized_size, CV_8UC3);
						for (cv::RotatedRect ellipse : ellipses) {
							cv::ellipse(testerino, ellipse, 255);
						}
						stages.push_back(testerino);
					}
					cv::bitwise_not(mask, mask);
					cv::bitwise_and(work_img, mask, work_img);
#endif

					stages.push_back(work_img.clone());
				}

				{
					double rho = 1, theta = 1 * CV_PI / 180;
					// Since most images are small and noisy this parameters need
					// to be low.
					int threshold = hough_lines_threshold;
					double minLineLenght = hough_lines_minLineLength,
						maxLineGap = hough_lines_maxLineGap;
					cv::HoughLinesP(work_img, lines, rho, theta, threshold,
						minLineLenght, maxLineGap);
					if (lines.rows >= 2) {
						CV_Assert(lines.cols == 1);
						CV_Assert(lines.type() == CV_32SC4);
						enough_lines = true;
						break;
					}
				}
			}
			if (!enough_lines) {
				skipped_imgs.append(stages);
				++skipped[thread_num];
				continue;
			}

			constexpr int K = 2;
			unsigned char centers_data[CV_ELEM_SIZE(CV_32FC1)*K*4] = {};
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
			// segments.
			cv::Point2f intersec;
			{
				float x1 = min_p1.x, x2 = min_p2.x, x3 = hour_p1.x, x4 = hour_p2.x,
				      y1 = min_p1.y, y2 = min_p2.y, y3 = hour_p1.y, y4 = hour_p2.y;
				float denumerator = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4);
				float det_12 = x1*y2 - y1*x2, det_34 = x3*y4 - y3*x4;
				intersec.x = (det_12*(x3-x4) - (x1-x2)*det_34)/denumerator;
				intersec.y = (det_12*(y3-y4) - (y1-y2)*det_34)/denumerator;
			}
			// Since most images have the clock perfectly entered this performs
			// way better than it should.
			intersec = cv::Point2f(img.size()/2);
			// We want that p2 is the furthest from the center of the clock.
			if (line_length(intersec, min_p1) > line_length(intersec, min_p2)) {
				cv::swap(min_p1, min_p2);
			}
			if (line_length(intersec, hour_p1) > line_length(intersec, hour_p2)) {
				cv::swap(hour_p1, hour_p2);
			}
			// min_p1 = hour_p1 = intersec;
			// cv::fastAtan2 returns degrees not radiants.
			float min_angle = cv::fastAtan2((-min_p2.y) - (-min_p1.y), min_p2.x - min_p1.x);
			float hour_angle = cv::fastAtan2((-hour_p2.y) - (-hour_p1.y), hour_p2.x - hour_p1.x);
			// We rotate the the angle to be 0 at 12 o'clock i.e. by 90 degrees
			// counterclockwise.
			min_angle = fmod(90 - min_angle + 360, 360);
			hour_angle = fmod(90 - hour_angle + 360, 360);

			// The graph of valid configurations of hour and minute hand is
			// given by the following function (in the hour and minute space,
			// not the one of degrees):
			//     y = 60x % 60 where 0 <= x < 12
			// NOTE: We could see how distant our pair (min_angle, hour_angle)
			// is from an ideal solution and act upon that. This could be an
			// hint for straightening the image.

			static_assert((360/60 == 6) & (360/12 == 30), "bad math");
			float minute_frac = min_angle/6, hour_frac = hour_angle/30;

			int hour = cvFloor(hour_frac);
			int minute = cvRound(minute_frac);

			// The valid hours in the dataset are 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.
			if (hour == 0) {
				hour = 12;
			}
			int predicted = hour*60 + minute;

			CV_Assert(predicted >= 60);
			CV_Assert(predicted <= 780);
			CV_Assert(labels[r] >= 60);
			CV_Assert(labels[r] <= 780);

			static_assert(13*60 == 780, "bad math");
			static_assert(12*60 == 720, "bad math");
			// Here we use modulus 720 and not 780 because the difference
			// between predicted and label removes the quirks in the strange way
			// that the dataset uses.
			if (cv::cv_abs((predicted - labels[r]) % 720) <= 1) {
				++correct[thread_num];
			} else {
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
				cv::String predicted_text = cv::format("pred:  %d:%d", hour, minute);
				putText(viz, predicted_text, cv::Point(5,20));
				int label_hour = labels[r]/60;
				int label_min = labels[r]%60;
				cv::String label_text = cv::format("label: %d:%d", label_hour, label_min);
				putText(viz, label_text, cv::Point(5, 110));
				cv::drawMarker(viz, intersec, blue);
				stages.push_back(viz);
				wrong_imgs.append(stages);
			}
		}
	}
	
	float get_accuracy() const {
		size_t tot_correct = 0;
		for (auto n : correct) {
			tot_correct += n.size;
		}
		return (float)tot_correct / imgs.size();
	}
	
	float get_skipped() const {
		size_t tot_skipped = 0;
		for (auto n : skipped) {
			tot_skipped += n.size;
		}
		return (float)tot_skipped / imgs.size();
	}
};

#pragma mark Optimizer

// Cool but unused templates.
template<unsigned N, typename T, typename... Ts>
struct type_switch {
	using type = typename type_switch<N - 1, Ts...>::type;
};
template<typename T, typename... Ts>
struct type_switch<0, T, Ts...> {
	using type = T;
};

// We increment start by step until it is less then stop.
struct Param {
	enum struct Tag {INT, LONG, FLOAT, DOUBLE,};
	union Val {
		int i; long l; float f; double d;
		Val(int value)    : i(value) {}
		Val(long value)   : l(value) {}
		Val(float value)  : f(value) {}
		Val(double value) : d(value) {}
	};

	Tag tag;
	Val start;
	Val stop;
	Val step;
	Val current;

	Param(int start, int stop, int step)
		: tag(Param::Tag::INT),    start(start), stop(stop), step(step), current(start) {CV_Assert(invariant());}
	Param(long start, long stop, long step)
		: tag(Param::Tag::LONG),   start(start), stop(stop), step(step), current(start) {CV_Assert(invariant());}
	Param(float start, float stop, float step)
		: tag(Param::Tag::FLOAT),  start(start), stop(stop), step(step), current(start) {CV_Assert(invariant());}
	Param(double start, double stop, double step)
		: tag(Param::Tag::DOUBLE), start(start), stop(stop), step(step), current(start) {CV_Assert(invariant());}

	template<typename T>
	bool check_invariant(T start, T stop, T step, T current) {
		if (step == 0) {
			return false;
		} else if (step > 0) {
			return start < stop && current < stop;
		} else /* step < 0 */ {
			return start > stop && current > stop;
		}
	}

	bool invariant() {
		switch (tag) {
		case Tag::INT:    return check_invariant(start.i, stop.i, step.i, current.i);
		case Tag::LONG:   return check_invariant(start.l, stop.l, step.l, current.l);
		case Tag::FLOAT:  return check_invariant(start.f, stop.f, step.f, current.f);
		case Tag::DOUBLE: return check_invariant(start.d, stop.d, step.d, current.d);
		}
		CV_Error(cv::Error::Code::StsBadArg, "bad tag");
	}

	void reset() {
		CV_Assert(invariant());
		switch (tag) {
		case Tag::INT:    current.i = start.i; return;
		case Tag::LONG:   current.l = start.l; return;
		case Tag::FLOAT:  current.f = start.f; return;
		case Tag::DOUBLE: current.d = start.d; return;
		}
		CV_Error(cv::Error::Code::StsBadArg, "bad tag");
	}

	void next() {
		CV_Assert(invariant());
		CV_Assert(has_next());
		switch (tag) {
		case Tag::INT:    current.i += step.i; return;
		case Tag::LONG:   current.l += step.l; return;
		case Tag::FLOAT:  current.f += step.f; return;
		case Tag::DOUBLE: current.d += step.d; return;
		}
		CV_Error(cv::Error::Code::StsBadArg, "bad tag");
	}

	template<typename T>
	bool check_has_next(T a, T b, T c) {
		using limits = std::numeric_limits<T>;
		// a + b > limits::max()
		if (b > 0 && a > limits::max() - b) {
			return false;
		}
		// a + b < limits::min()
		if (b < 0 && a < limits::min() - b) {
			return false;
		}
		// Now we can do the test safely.
		return b > 0 ? a + b < c : a + b > c;
	}

	bool has_next() {
		CV_Assert(invariant());
		// https://stackoverflow.com/a/1514309
		// if b == 0 overflow/underflow can't happen.
		switch (tag) {
		case Tag::INT:    return check_has_next(current.i, step.i, stop.i);
		case Tag::LONG:   return check_has_next(current.l, step.l, stop.l);
		case Tag::FLOAT:  return check_has_next(current.f, step.f, stop.f);
		case Tag::DOUBLE: return check_has_next(current.d, step.d, stop.d);
		}
		CV_Error(cv::Error::Code::StsBadArg, "bad tag");
	}

#if 0
	unsigned size() {
		switch (tag) {
		case Tag::INT:    return (stop.i - start.i)/step.i; // division may cause UB for -1
		case Tag::LONG:   return (stop.l - start.l)/step.l;
		// option 1: saturate for out of range numbers https://stackoverflow.com/a/2545218
		// option 2: float ULP distance (or even better distance not based on step)
		if step < ulp
			return ulp_dist(stop-start)
		else
			// questo scaling è mantiene la correttezza che mi serve???
			// no perche se step = 0.1 il la distanza si ingrandisce e potrebbe
			// andare a inf
			// what about absorption and catastrophic cancellation???
			return ulp_dist((stop-start)/step);
		case Tag::FLOAT:  return (stop.f - start.f)/step.f;
		case Tag::DOUBLE: return (stop.d - start.d)/step.d;
		}
	}
#endif
};

std::ostream& operator <<(std::ostream& os, Param p) {
	switch (p.tag) {
	case Param::Tag::INT:    return os << p.current.i;
	case Param::Tag::LONG:   return os << p.current.l;
	case Param::Tag::FLOAT:  return os << p.current.f;
	case Param::Tag::DOUBLE: return os << p.current.d;
	}
	CV_Error(cv::Error::Code::StsBadArg, "bad tag");
}

struct ParamIter {
	Param *params;
	unsigned length;
	bool finished;

	void next() {
		unsigned j = 0;
		while (j < length && !params[j].has_next()) {
			params[j].reset();
			j++;
		}
		if (j == length) {
			finished = true;
			return;
		}
		params[j].next();
	}
};

// TODO: make the order of DebugVec deterministic using an hashtable with ids

extern "C" {
// Return true if this function should be called again.
bool
run_classifier(void *ids_, void *imgs_, void *labels_) {
	// cv::setNumThreads(1);
	int num_threads = cv::getNumThreads();

	const std::vector<int64>& ids = *reinterpret_cast<std::vector<int64>*>(ids_);
	const std::vector<cv::Mat>& imgs = *reinterpret_cast<std::vector<cv::Mat>*>(imgs_);
	const std::vector<int16_t>& labels = *reinterpret_cast<std::vector<int16_t>*>(labels_);

	std::vector<ParallelClassification::Size> correct(num_threads),
		skipped(num_threads);

#if 1
	ParallelClassification classifier(imgs, labels, correct, skipped,
									  34,  // mask_radius_scale THIS NOT USED ANYMORE!
									  79,  // canny_threshold1
									  200, // canny_threshold2
									  63,  // connected_component_area THIS IS NOT USED ANYMORE!
									  15,  // hough_lines_threshold
									  9,  // hough_lines_minLineLength
									  99   // hough_lines_maxLineGap
	);
	cv::parallel_for_(cv::Range(0, imgs.size()), classifier);
	
	CV_LOG_INFO(NULL, "correct: " << classifier.get_accuracy());
	CV_LOG_INFO(NULL, "skipped: " << classifier.get_skipped());

	cv::String window_name = "misclassified";
	return explore_dataset(window_name, wrong_imgs.mats);
#else
	Param params[] {
		// mask_radius_scale
		{34, 37, 1}, //35
		// canny_threshold1
		{79., 82., 1.}, // 80
		// canny_threshold2
		{199., 202., 1.}, // 200
		// connected_component_area
		{8*8-1, 8*8+2, 1}, // 8*8
		// hough_lines_threshold
		{14, 17, 1}, // 15
		// hough_lines_minLineLength
		{9., 12., 1.}, // 10
		// hough_lines_maxLineGap
		{99., 102., 2.}, // 100
	};
	constexpr size_t params_length = sizeof params / sizeof *params;

	float best_score = 0;
	// We initialize it with dummy parameters.
	Param best_params[params_length] = {
		{0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1},
	};
	for (ParamIter it = {params, params_length, false}; !it.finished; it.next()) {
		size_t tot_correct = 0, tot_skipped = 0;
		ParallelClassification classifier(imgs, labels, correct, skipped,
			it.params[0].current.i, // mask_radius_scale THIS NOT USED ANYMORE!
			it.params[1].current.d, // canny_threshold1
			it.params[2].current.d, // canny_threshold2
			it.params[3].current.i, // connected_component_area THIS IS NOT USED ANYMORE!
			it.params[4].current.i, // hough_lines_threshold
			it.params[5].current.d, // hough_lines_minLineLength
			it.params[6].current.d  // hough_lines_maxLineGap
		);
		cv::parallel_for_(cv::Range(0, imgs.size()), classifier);
		float score = classifier.get_accuracy();
		CV_LOG_INFO(NULL, "correct: " << score);
		CV_LOG_INFO(NULL, "skipped: " << classifier.get_skipped());
		if (score > best_score) {
			best_score = score;
			std::copy(it.params, it.params + it.length, best_params);
		}
		CV_LOG_INFO(NULL, "best accuracy: " << best_score);
		std::fill(correct.begin(), correct.end(), ParallelClassification::Size{0});
		std::fill(skipped.begin(), skipped.end(), ParallelClassification::Size{0});
		skipped_imgs.mats.clear(); wrong_imgs.mats.clear();
	}

	CV_LOG_INFO(NULL, "best score: " << best_score);
	std::stringstream ss;
	for (unsigned i = 0; i < params_length; ++i) {
		ss << best_params[i] << ' ';
	}
	CV_LOG_INFO(NULL, "best params: " << ss.str());

	return false;
#endif
}
}
