/* https://forum.opencv.org/t/detect-an-ellipse-and-get-its-dimensions/6949
 * https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569894217.pdf
 * https://link.springer.com/article/10.1007/s10851-019-00928-6
 * https://math.stackexchange.com/questions/654275/homography-between-ellipses
 * https://openaccess.thecvf.com/content_cvpr_2016/papers/Huang_Homography_Estimation_From_CVPR_2016_paper.pdf
 * https://cmp.felk.cvut.cz/~chum/papers/chum-icpr12.pdf
 * https://www.cs.ubc.ca/labs/lci/thesis/ankgupta/gupta11crv.pdf
 */

// TODO: lldb python script to print cv::Mat to sixel.

// 1. Read from the CSV all the bounding boxes and identifiers and the file in
//    full color, then crop it and keep just the ROI in memory and then compress
//    it. The ideal thing would be to use a custom allocator to keep the dataset
//    in a continous chunk of memory.
// 2. Find an ellipse in the image (maybe looking a data in a specific color
//    channel since the border should always have the same color) and than
//    "straighten" it some way. Maybe just turning the bounding rectangle into a
//    square. Since all images are taken in the right orientation (i.e. are not
//    flipped or rotated) the only straightning we have to do is a projective
//    correction.
// 3. Use mine and Giorgio's algorithm to determine the hour. We have to then
//    tollerate the presence of a third hand the (second one). This can be done
//    by looking at the compatness value returned by the k-means algorithm when
//    k=3 and k=2. We chose the clustering of the instance of k-means with the
//    best compactness. If we have a third hand we simply discard it.
//
// Some algorithms like blurring should be applied (or not) as a function of the
// resolution of the image. Some image are too small to any kind of
// preprocessing on.

// https://www.digiater.nl/openvms/decus/vax90b1/krypton-nasa/all-about-sixels.text
// https://en.wikipedia.org/wiki/Sixel
// https://vt100.net/docs/vt3xx-gp/chapter14.html
// DCS \x90 Device Control String
// ST \x9c String Terminator
// std::cout << "\x90??????\x9c\n";
// no sixel support in Terminal.app :(

#include "opencv2/core/softfloat.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <errno.h>
#include <fstream>
#include <iostream>

#define BREAKPOINT __builtin_debugtrap()
// An dereferentiable array litteral.
#define A(T, ...) (std::initializer_list<T>({__VA_ARGS__}).begin())

// $ sysctl hw.cachelinesize
// hw.cachelinesize: 64
constexpr int cache_line_size = 64;
constexpr int seed = 42;

enum struct retCode {
	OK,
	BAD_GLOB,
	BAD_IMG,
	BAD_READ,
	BAD_PATH,
	BAD_IMG_ID,
	BAD_CSV,
	BAD_ALLOC
};

// Utilities. //////////////////////////////////////////////////////////////////

// TODO: I should use tracing macros instead.
struct Timed_block {
	cv::TickMeter tm;
	const char *name;
	Timed_block(const char *name) : tm(), name(name) {
		tm.start();
	}
	~Timed_block() {
		tm.stop();
		std::cout << '\'' << name << "' took "
		<< static_cast<long long>(tm.getTimeMilli()) << "ms\n";
	}
};

static cv::softfloat
line_length(cv::Point2f a, cv::Point2f b) {
	// NOTE: should we use cv::normL2Sqr? or cv::norm? or cv::magnitude?
	cv::Point2f diff = a - b;
	cv::softfloat res(cv::sqrt(diff.x*diff.x + diff.y*diff.y));
	return res;
}

static cv::Mat
copyTo(cv::Mat src, cv::Mat dst) {
	src.copyTo(dst);
	return dst;
}

void putText(cv::Mat img, const cv::String &text, cv::Point org) {
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
static void
explore_dataset(
	const cv::String &window_name,
	const std::vector<cv::Mat>& imgs) {
	CV_Assert(imgs.size() > 0);

	cv::namedWindow(window_name, cv::WINDOW_NORMAL);

	int index = 0;
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
		case -1: break; // If no key was pressed;
		case 'q': return;
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

	// Saddly windows are not destroyed.
	cv::destroyWindow(window_name);
	cv::destroyAllWindows();
	(void) cv::waitKey(1);
}

// Debugging utilities. ////////////////////////////////////////////////////////

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

// Implementation functions. ///////////////////////////////////////////////////

static retCode
read_csv(
		const char *path,
		const std::vector<int64>& ids,
		const std::vector<cv::Mat>& imgs,
		std::vector<int16_t>& labels,
		std::vector<cv::Rect>& bboxes,
		int base
	) {

	CV_Assert(path);
	CV_Assert(ids.size() == labels.size());
	CV_Assert(labels.size() == bboxes.size());
	CV_Assert(2 <= base & base <= 16);

	std::fstream csv_stream(path, std::ios_base::in);
	if (csv_stream.fail()) {
		std::cerr << "unable to open '" << path << "'\n";
		return retCode::BAD_PATH;
	}

	const char *expected_header = "img_id,file_name,bbox_gt,hour,minute,bbox_det";
	std::string csv_line;
	std::getline(csv_stream, csv_line);
	if (csv_line != expected_header) {
		std::cerr << path << ":1 the header is not '" << expected_header << "'\n";
		return retCode::BAD_CSV;
	}
	size_t lineno = 1;
	while (std::getline(csv_stream, csv_line)) {
		lineno++;
		char *end_char = nullptr;
		int64 id = std::strtoull(csv_line.c_str(), &end_char, base);
		if (*end_char != ',') {
			std::cerr << path << ':' << lineno
				<< " the first column does not contain a numeric identifier\n";
			return retCode::BAD_CSV;
		}
		if (errno == ERANGE) {
			std::cerr << path << ':' << lineno
				<< "the identifier number is too big\n";
			return retCode::BAD_CSV;
		}

		size_t file_name_start_pos = end_char - csv_line.c_str() + 1;
		size_t bbox_gt_start_pos = csv_line.find(',', file_name_start_pos);
		if (bbox_gt_start_pos == csv_line.npos) {
			std::cerr << path << ':' << lineno
				<< "does not have literal in the bbox_gt column\n";
			return retCode::BAD_CSV;
		}
		bbox_gt_start_pos++;

		if (csv_line[bbox_gt_start_pos] != '"') {
			std::cerr << path << ':' << lineno << " \n";
			return retCode::BAD_CSV;
		}

		// bbox_gt contains multiple bounding boxes if there is more than one
		// clock in a given picture. We skip it since in the last column there
		// is bbox_det which is always a single bounding box.
		// NOTE: if we want to go back and read them setting a limit of 8 or 16
		// bounding boxes might be good.

		const char *bbox_gt_end = csv_line.c_str() + bbox_gt_start_pos + 1;

		{
			bool escaped = false;
			while (true) {
				if (*bbox_gt_end == 0)
					break;
				if ((*bbox_gt_end == '"') & !escaped)
					break;
				if (escaped) {
					escaped = false;
				} else if (*bbox_gt_end == '\\') {
					escaped = true;
				}
				bbox_gt_end++;
			}
			if (*bbox_gt_end == 0) {
				std::cerr << path << ':' << lineno << " the Python's literal is not"
					" correct\n";
				return retCode::BAD_CSV;
			}
			CV_Assert(*bbox_gt_end == '"');
		}

		int hour = 0, minute = 0;
		int parsed_numbers = std::sscanf(bbox_gt_end+1, ",%2d,%2d", &hour, &minute);
		if (parsed_numbers != 2) {
			std::cerr << path << ':' << lineno << " could not find hours and minutes\n";
			return retCode::BAD_CSV;
		}
		if (hour <= 0 | hour > 12) {
			std::cerr << path << ':' << lineno << " bad hour\n";
			return retCode::BAD_CSV;
		}
		if (minute < 0 | minute > 59) {
			std::cerr << path << ':' << lineno << " bad minute\n";
			return retCode::BAD_CSV;
		}
		int16_t label = hour*60 + minute;

		// Looking for the start of the bbox_det literal.
		size_t bbox_det_start_pos = csv_line.find('"', bbox_gt_end + 1 - csv_line.c_str());
		if (bbox_det_start_pos == csv_line.npos) {
			std::cerr << path << ':' << lineno
				<< "does not have a literal in bbox_det column\n";
			return retCode::BAD_CSV;
		}

		union {
			float a[4];
			struct {
				float x1, y1, x2, y2;
			};
		} bbox;
		parsed_numbers = std::sscanf(csv_line.c_str() + bbox_det_start_pos,
			"\"[%7f, %7f, %7f, %7f", // NOTE: this number 7 here is a bit arbitrary.
			&bbox.a[0], &bbox.a[1], &bbox.a[2], &bbox.a[3]);
		if (parsed_numbers != 4) {
			std::cerr << path << ':' << lineno << " the Python's literal is not"
				" correct\n";
			return retCode::BAD_CSV;
		}

#if 1
		size_t position = ids.size();
		for (size_t i = 0; i < ids.size(); ++i) {
			if (ids[i] == id) {
				position = i;
				break;
			}
		}
		if (position == ids.size()) {
			std::cerr << path << ':' << lineno << " unknown identifier\n";
			return retCode::BAD_IMG_ID;
		}
		size_t location = position;
#else
		auto position = std::find(ids.cbegin(), ids.cend(), id);
		if (position == ids.cend()) {
			std::cerr << path << ':' << lineno << " unknown identifier\n";
			return retCode::BAD_IMG_ID;
		}
		size_t location = position - ids.cbegin();
#endif
		labels[location] = label;

		// I don't know why this voodoo here is needed. Taken from data.py
		bbox.x2 += bbox.x1; bbox.y2 += bbox.y1;
		cv::Rect rect(bbox.x1, bbox.y1, bbox.x2-bbox.x1, bbox.y2-bbox.y1);
		const cv::Mat& img = imgs[location];
		cv::Rect img_rect = cv::Rect(0, 0, img.size().width, img.size().height);
		if ((rect | img_rect) != img_rect) {
			std::cerr << path << ':' << lineno << " the bounding box is not "
				"contained in the image\n";
			return retCode::BAD_CSV;
		}
		bboxes[location] = rect;
	}

	if (csv_stream.bad()) {
		std::cerr << path << ':' << lineno
			<< " unable to read next line\n";
		return retCode::BAD_READ;
	}

	return retCode::OK;
}

// To be used only with paths returned by glob.
struct ParallelRead CV_FINAL : public cv::ParallelLoopBody {

	// Posso usare OpenCV parallel_for_ lèggendo dai CSV se divido il file CSV
	// in proc_count (e.g. 12) parti e ogni thread, tranne il primo, cercano
	// un "punto di sincronizzazione" ovvero una \n e cominciano a leggere da
	// là fino alla \n che è ila punto di sincronizzazione del thread
	// successivo. Se accade un errore devo contare tutte le righe da iniziò
	// file per comunicare dove è accaduto.

	// TODO: this should take as a range the number of bytes in a CSV file

	struct RetCode { alignas(cache_line_size) retCode res; };

	const std::vector<cv::String>& paths;
	std::vector<int64>& ids;
	std::vector<cv::Mat>& imgs;
	int base;
	cv::Mutex &mutex;
	std::vector<RetCode> &results;

	ParallelRead(
		const std::vector<cv::String>& paths,
		std::vector<int64>& ids,
		std::vector<cv::Mat>& imgs,
		int base,
		cv::Mutex &mutex,
		std::vector<RetCode> &results
	) : paths(paths), ids(ids), imgs(imgs), base(base), mutex(mutex),
		results(results) {
		CV_Assert(paths.size() == ids.size());
		CV_Assert(ids.size() == imgs.size());
		CV_Assert(2 <= base & base <= 16);
		CV_Assert(results.size() == cv::getNumThreads());
	}

	virtual void operator() (const cv::Range& range) const CV_OVERRIDE {
		for (int r = range.start; r < range.end; ++r) {
			int thread_num = cv::getThreadNum();
			const cv::String &path = paths[r];
			imgs[r] = cv::imread(path);
			size_t id_start = path.find_last_of('/');

			// Due to the glob pattern we know that:
			CV_Assert(path.find(".jpg", path.size() - 4) != cv::String::npos);
			CV_Assert(id_start != cv::String::npos);
			CV_Assert(id_start + 1 < path.size());

			char *end_char = nullptr;
			ids[r] = std::strtoull(path.c_str() + id_start + 1, &end_char, base);

			if (imgs[r].data == nullptr) {
				mutex.lock();
				std::cerr << "unable to read image '" << path << "'\n";
				mutex.unlock();
				results[thread_num] = RetCode{retCode::BAD_IMG};
				return;
			}
			if (*end_char != '.') {
				mutex.lock();
				std::cerr << "the name of the image is not only a base " << base
					<< " identifier number\n";
				mutex.unlock();
				results[thread_num] = RetCode{retCode::BAD_IMG_ID};
				return;
			}
			if (errno == ERANGE) {
				mutex.lock();
				std::cerr << "the name of the image (parsed as a number base "
					<< base << ") is too big\n";
				mutex.unlock();
				results[thread_num] = RetCode{retCode::BAD_IMG_ID};
				return;
			}
		}
	}

};

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
		// FIXME: this does not make the algorithm deterministic.
		cv::theRNG().state = seed;
		cv::setRNGSeed(seed);
		cv::Mat copied; // For temporary copies needed by the various filters.
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
			cv::Size standardized_size(128, 128);
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

				// We mask the image before the edge detection to minimize the
				// number of edges that cound confuse the subsequent pahse.
				if (true) {
					cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
					cv::Point center = img.size()/2;
					// int radius = center.x/1.8; // 128/1.8 = 71
					int radius = mask_radius_scale;
					cv::Scalar white(0xff, 0xff, 0xff);
					int thickness = cv::FILLED, lineType = cv::LINE_8, shift = 0;
					cv::circle(mask, center, radius, white, thickness, lineType, shift);
					cv::bitwise_and(copyTo(work_img, copied), mask, work_img);
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
					stages.push_back(work_img.clone());
				}

				// At this point the edge detector always detects a circle
				// arround the previous mask. We remove it by using a slightly
				// smaller circular mask.
				if (true) {
					cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
					cv::Point center = img.size()/2;
					// int radius = center.x/1.8-1;
					int radius = mask_radius_scale-1;
					cv::Scalar white(0xff, 0xff, 0xff);
					int thickness = cv::FILLED, lineType = cv::LINE_8, shift = 0;
					cv::circle(mask, center, radius, white, thickness, lineType, shift);
					cv::bitwise_and(copyTo(work_img, copied), mask, work_img);
					CV_Assert(work_img.size() == img.size());
					CV_Assert(work_img.type() == CV_8UC1);
					stages.push_back(work_img.clone());
				}

				// TODO: maybe dilate the image a bit.

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

					// TODO: it would be cool to highlight just the "small"
					// connected components.
					cv::Mat mask = cv::Mat::zeros(standardized_size, CV_8UC1);
					for (int i = 1; i < n_labels; ++i) {
						int area = connected_component_area;
						if (stats.at<int>(i, cv::CC_STAT_AREA) > connected_component_area)
							continue;
						cv::bitwise_or(labels == i, mask, mask);
					}
					cv::bitwise_not(mask, mask);
					cv::bitwise_and(work_img, mask, work_img);

					if (n_labels > 255) {
						CV_LOG_ERROR(NULL, "too many connected components");
						break;
					}
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
					stages.push_back(work_img.clone());
				}

				{
					// NOTE: should I optimize this two too?
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
			static unsigned char centers_data[CV_ELEM_SIZE(CV_32FC1)*K*4] = {};
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
			// We want that p2 is the furthest from the center of the clock.
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
				cv::String predicted_text = cv::format("%d:%d", hour, minute);
				putText(viz, predicted_text, cv::Point(10,40));
				int label_hour = labels[r]/60;
				int label_min = labels[r]%60;
				cv::String label_text = cv::format("%d:%d", label_hour, label_min);
				putText(viz, label_text, cv::Point(10, 80));
				cv::drawMarker(viz, intersec, blue);
				stages.push_back(viz);
				wrong_imgs.append(stages);
			}
		}
	}
};


#pragma mark Optimizer

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

#pragma mark Main

// br set -X main -p return
int
main(int argc, char **argv) {
	Timed_block b("all");
	CV_LOG_INFO(NULL, "starting");

	std::terminate_handler terminate_ = std::set_terminate([]() -> void {
		std::exception_ptr ex = std::current_exception();
		try {
			std::rethrow_exception(ex);
		} catch (const std::bad_alloc& ex) {
			std::cerr << "unable to allocate memory: " << ex.what() << '\n';
			std::exit(static_cast<int>(retCode::BAD_ALLOC));
		} catch (const std::exception& ex) {
			std::cerr << "termination due to uncaught exception: " << ex.what() << '\n';
			std::abort();
		} catch (...) {
			std::cerr << "something else was thrown\n";
			std::abort();
		}
	});

	// (void) cv::setBreakOnError(true);
	int num_threads = cv::getNumThreads();
	cv::theRNG().state = seed;
	cv::setRNGSeed(seed);

	std::vector<cv::String> coco_paths, openimg_paths;

	try {
		Timed_block b("globing");
		bool recursive = false;
		cv::glob("itsabouttime/data/coco/*.jpg", coco_paths, recursive);
		cv::glob("itsabouttime/data/openimg/*.jpg", openimg_paths, recursive);
	} catch (cv::Exception& ex) {
		std::cerr << ex.msg << '\n';
		return static_cast<int>(retCode::BAD_GLOB);
	}

	size_t coco_size = coco_paths.size(), openimg_size = openimg_paths.size();
	std::vector<int64>    coco_ids(coco_size),    openimg_ids(openimg_size);
	std::vector<cv::Mat>  coco_imgs(coco_size),   openimg_imgs(openimg_size);
	std::vector<int16_t>  coco_labels(coco_size), openimg_labels(openimg_size);
	std::vector<cv::Rect> coco_bboxes(coco_size), openimg_bboxes(openimg_size);

	retCode res = retCode::OK;

	{
		Timed_block b("reading images");

		cv::Mutex mutex;
		std::vector<ParallelRead::RetCode> results(num_threads);

		int base = 10;
		ParallelRead coco_reader(coco_paths, coco_ids, coco_imgs, base, mutex, results);
		cv::parallel_for_(cv::Range(0, coco_size), coco_reader);
		for (const auto& res : results) {
			if (res.res != retCode::OK) {
				return static_cast<int>(res.res);
			}
		}

		base = 16;
		ParallelRead openimg_reader(openimg_paths, openimg_ids, openimg_imgs, base, mutex, results);
		cv::parallel_for_(cv::Range(0, openimg_size), openimg_reader);
		for (const auto& res : results) {
			if (res.res != retCode::OK) {
				return static_cast<int>(res.res);
			}
		}
	}

	{
		Timed_block b("reading CSVs");
		if ((res = read_csv("itsabouttime/data/coco_final.csv", coco_ids, coco_imgs, coco_labels, coco_bboxes, 10)) != retCode::OK) {
			return static_cast<int>(res);
		}
		if ((res = read_csv("itsabouttime/data/openimg_final.csv", openimg_ids, openimg_imgs, openimg_labels, openimg_bboxes, 16)) != retCode::OK) {
			return static_cast<int>(res);
		}
	}

	size_t dataset_size = coco_size + openimg_size;
	std::vector<cv::Mat> imgs(dataset_size);
	std::vector<int16_t> labels(dataset_size);
	{
		Timed_block b("cropping");
		for (size_t i = 0; i < coco_size; ++i) {
			coco_imgs[i] = coco_imgs[i](coco_bboxes[i]).clone();
			CV_Assert(coco_imgs[i].isContinuous());
			CV_Assert(!coco_imgs[i].isSubmatrix());
			imgs[i] = coco_imgs[i];
			labels[i] = coco_labels[i];
		}
		for (size_t i = 0; i < openimg_size; ++i) {
			openimg_imgs[i] = openimg_imgs[i](openimg_bboxes[i]).clone();
			CV_Assert(openimg_imgs[i].isContinuous());
			CV_Assert(!openimg_imgs[i].isSubmatrix());
			imgs[coco_size + i] = openimg_imgs[i];
			labels[coco_size + i] = openimg_labels[i];
		}
	}
	coco_imgs.clear();
	coco_bboxes.clear();
	coco_labels.clear();
	openimg_imgs.clear();
	openimg_bboxes.clear();
	openimg_labels.clear();

	// cv::setNumThreads(1);

	size_t tot_correct = 0, tot_skipped = 0;
	std::vector<ParallelClassification::Size> correct(num_threads),
		skipped(num_threads);
	{
		Timed_block b("classification");
		ParallelClassification classifier(imgs, labels, correct, skipped,
			35, 79, 201, 63, 14, 10, 99
		);
		cv::parallel_for_(cv::Range(0, dataset_size), classifier);
	}
	for (auto n : correct) {
		tot_correct += n.size;
	}
	for (auto n : skipped) {
		tot_skipped += n.size;
	}

	float score = (float) tot_correct / dataset_size;
	std::cout
		<< "correct: " << score << '\n'
		<< "skipped: " << (float) tot_skipped / dataset_size << '\n';

	// (void) cv::startWindowThread();
	cv::String window_name = "skipped";
	// explore_dataset(window_name, skipped_imgs.mats);
	window_name = "misclassified";
	explore_dataset(window_name, wrong_imgs.mats);

	return static_cast<int>(retCode::OK);

	Param params[] {
		// mask_radius_scale
		// {70, 73, 1}, // 71
		{34, 37, 1}, //35
		// canny_threshold1
		{79., 82., 1.}, // 80
		// canny_threshold2
		{199., 202., 1.}, // 200
		// connected_component_area
		{8*8-1, 8*8+2, 1}, // 8*8
		// hough_lines_rho
		/* ignored for now */
		// hough_lines_theta
		/* ignored for now */
		// hough_lines_threshold
		{14, 17, 1}, // 15
		// hough_lines_minLineLength
		{9., 12., 1.}, // 10
		// hough_lines_maxLineGap
		{99., 102., 2.}, // 100
	};
	constexpr size_t params_length = sizeof params / sizeof *params;

	float best_score = 0;
	// 35 79 201 63 14 10 99
	Param best_params[params_length] = {
		{0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1}, {0,2,1},
	};
	for (ParamIter it = {params, params_length, false};
			!it.finished;
			it.next()) {
		// we clear the stuff from the previous execution.
		skipped_imgs.mats.clear();
		wrong_imgs.mats.clear();
		size_t tot_correct = 0, tot_skipped = 0;
		std::vector<ParallelClassification::Size> correct(num_threads),
			skipped(num_threads);
		{
			Timed_block b("classification");
			ParallelClassification classifier(imgs, labels, correct, skipped,
				it.params[0].current.i,
				it.params[1].current.d,
				it.params[2].current.d,
				it.params[3].current.i,
				it.params[4].current.i,
				it.params[5].current.d,
				it.params[6].current.d
			);
			cv::parallel_for_(cv::Range(0, dataset_size), classifier);
		}
		for (auto n : correct) {
			tot_correct += n.size;
		}
		for (auto n : skipped) {
			tot_skipped += n.size;
		}

		float score = (float) tot_correct / dataset_size;
		std::cout
			<< "correct: " << score << '\n'
			<< "skipped: " << (float) tot_skipped / dataset_size << '\n';
		if (score > best_score) {
			std::cout << "**** best: " << score << '\n';
			best_score = score;
			for (unsigned i = 0; i < it.length; ++i) {
				best_params[i] = it.params[i];
			}
		}
	}

	std::cout << "best score: " << best_score << '\n';
	for (unsigned i = 0; i < params_length; ++i) {
		switch (best_params[i].tag) {
		case Param::Tag::INT: std::cout << best_params[i].current.i << ' '; break;
		case Param::Tag::LONG: std::cout << best_params[i].current.l << ' '; break;
		case Param::Tag::FLOAT: std::cout << best_params[i].current.f << ' '; break;
		case Param::Tag::DOUBLE: std::cout << best_params[i].current.d << ' '; break;
		}
	}
	std::cout << '\n';

	// (void) cv::startWindowThread();
	// explore_dataset(window_name, skipped_imgs.mats);
	window_name = "misclassified";
	explore_dataset(window_name, wrong_imgs.mats);

	return static_cast<int>(retCode::OK);
}
