// https://www.digiater.nl/openvms/decus/vax90b1/krypton-nasa/all-about-sixels.text
// https://en.wikipedia.org/wiki/Sixel
// https://vt100.net/docs/vt3xx-gp/chapter14.html
// DCS \x90 Device Control String
// ST \x9c String Terminator
// std::cout << "\x90??????\x9c\n";
// no sixel support in Terminal.app :(

#include "opencv2/core/utils/logger.hpp"
#include "opencv2/imgcodecs.hpp"

#include <algorithm>
#include <cerrno>
#include <fstream>
#include <iostream>

#include <dlfcn.h>

#define BREAKPOINT __builtin_debugtrap()

struct ParallelRead CV_FINAL : public cv::ParallelLoopBody {

	std::string csv;
	const char *base_dir;
	int base;
	cv::Mutex& mutex;
	std::vector<int64>& ids;
	std::vector<cv::Mat>& imgs;
	std::vector<int16_t>& labels;

	ParallelRead(
		std::ifstream&& csv_stream,
		const char *base_dir,
		int base,
		cv::Mutex& mutex,
		std::vector<int64>& ids,
		std::vector<cv::Mat>& imgs,
		std::vector<int16_t>& labels
	) : base_dir(base_dir), base(base), mutex(mutex), ids(ids),
		imgs(imgs), labels(labels) {
			CV_Assert(2 <= base && base <= 16);
			if (ids.size() != imgs.size() | imgs.size() != labels.size()) {
				CV_LOG_WARNING(NULL, "the vectors have different sizes.");
			}
			// If anything goes wrong we throw.
			csv_stream.exceptions(std::ifstream::failbit | std::ifstream::badbit);
			// https://insanecoding.blogspot.com/2011/11/how-to-read-in-file-in-c.html
			csv_stream.seekg(0, std::ios::end);
			csv.resize(csv_stream.tellg());
			csv_stream.seekg(0, std::ios::beg);
			csv_stream.read(&csv[0], csv.size());
			csv_stream.close();
		}

	using const_iterator = std::string::const_iterator;
	
	std::iterator_traits<const_iterator>::difference_type get_line_number(const_iterator end) const {
		return std::count(csv.cbegin(), end, '\n');
	}

	unsigned long long read_number_ending_in_coma(const_iterator begin, char **end_char, int base) const {
		errno = 0;
		unsigned long long number = std::strtoull(&*begin, end_char, base);
		if ((*end_char != &*begin) & (**end_char != ',')) {
			CV_Error(cv::Error::Code::StsParseError, "unable to convert number");
		}
		if (errno == ERANGE) {
			CV_Error(cv::Error::Code::StsParseError, "number too big");
		}
		
		return number;
	}
	
	float read_float_ending_in(const_iterator begin, char **end_char, char end) const {
		float res = std::strtof(&*begin, end_char);
		if ((*end_char != &*begin) & (**end_char != end)) {
			CV_Error(cv::Error::Code::StsParseError, "unable to convert float");
		}
		if (res == HUGE_VALF) {
			CV_Error(cv::Error::Code::StsParseError, "float too big");
		}
		return res;
	}

	// TODO: fuzz this.
	virtual void operator() (const cv::Range& range) const CV_OVERRIDE {
		CV_Assert(range.end <= csv.size());

		// We start by adjusting our range in the file to be a sequence starting
		// and ending at "line boundaries".
		const_iterator begin = csv.cbegin() + range.start;
		const_iterator end = csv.cbegin() + range.end;
		CV_Assert(end <= csv.cend());
		if (begin == csv.cbegin()) {
			const char *expected_header = "img_id,file_name,bbox_gt,hour,minute,bbox_det";
			if (csv.rfind(expected_header, 0) != 0) {
				CV_Error(cv::Error::Code::StsParseError, "unexpected header");
			}
		}
		begin = std::find(begin, csv.cend(), '\n');
		if (begin == csv.cend()) {
			// There is nothing left to read for this thread since the one
			// before it will read the last line.
			return;
		}
		++begin;
		// Now begin points at the start of a line.
		if (begin == csv.cend()) {
			// If the last line is empty we are in the same situation as before.
			return;
		}
		if (begin >= end) {
			// The range was too small, again the previous thread will read this
			// line.
			return;
		}
		end = std::find(end, csv.cend(), '\n');
		CV_Assert(begin < end);

		const_iterator line_begin = begin;
		const_iterator line_end = std::find(line_begin, end, '\n');

		std::string file_name, file_path;

		// TODO: before throwing an exception call get_line_number and log it as error.
		do {
			CV_Assert(line_begin < line_end);

			// FIXME: this should be a uint64 not a int64.
			// TODO: do the stupid check for int64 < unsigned long long.
			char *end_char = nullptr;
			int64 id = read_number_ending_in_coma(line_begin, &end_char, base);

			const_iterator file_name_begin = line_begin + (end_char - &*line_begin + 1);
			const_iterator file_name_end = std::find(file_name_begin, line_end, ',');
			file_name.assign(file_name_begin, file_name_end);

			file_path.assign(base_dir);
			file_path.append(file_name);
			cv::Mat img = cv::imread(file_path);
			if (img.empty()) {
				CV_Error(cv::Error::Code::StsBadArg, "unable to read image");
			}

			// bbox_gt contains multiple bounding boxes if there is more than one
			// clock in a given picture. We skip it since in the last column there
			// is bbox_det which is always a single bounding box.
			const_iterator bbox_gt_begin = file_name_end + 1;
			if (*bbox_gt_begin != '"') {
				CV_Error(cv::Error::Code::StsParseError, "bbox_gt is not a string");
			}
			const_iterator bbox_gt_end = std::find(bbox_gt_begin + 1, line_end, '"');
			if (*bbox_gt_end != '"') {
				CV_Error(cv::Error::Code::StsParseError, "bbox_gt does not have"
						 " a closing \" (escaping is not supported)");
			}

			if (*(bbox_gt_end + 1) != ',') {
				CV_Error(cv::Error::Code::StsParseError, "missing the hour column");
			}
			const_iterator hour_start = bbox_gt_end + 2;
			unsigned long long hour = read_number_ending_in_coma(hour_start, &end_char, 10);
			const_iterator minute_start = line_begin + (end_char - &*line_begin + 1);
			unsigned long long minute = read_number_ending_in_coma(minute_start, &end_char, 10);
			if ((hour <= 0) | (hour > 12)) {
				CV_Error(cv::Error::Code::StsBadArg, "the hour is not a number in [0, 12)");
			}
			if ((minute < 0) | (minute > 59)) {
				CV_Error(cv::Error::Code::StsBadArg, "the minute is not a number in (0, 60)");
			}
			// NOTE: why the fuck is this an int and not an unsigned?
			int16_t label = hour*60 + minute;
			
			const_iterator bbox_det_start = line_begin + (end_char - &*line_begin + 1);
			if (*bbox_det_start != '"') {
				CV_Error(cv::Error::Code::StsParseError, "bbox_det is not a string");
			}
			if (*(bbox_det_start + 1) != '[') {
				CV_Error(cv::Error::Code::StsParseError, "bbox_det does not contain a list");
			}
			// We exploit the fact that strtof discard initial white spaces.
			float x1 = read_float_ending_in((bbox_det_start + 2), &end_char, ',');
			float y1 = read_float_ending_in(line_begin + (end_char - &*line_begin + 1), &end_char, ',');
			float x2 = read_float_ending_in(line_begin + (end_char - &*line_begin + 1), &end_char, ',');
			float y2 = read_float_ending_in(line_begin + (end_char - &*line_begin + 1), &end_char, ']');
			
			// I don't know why this voodoo here is needed. Taken from data.py
			x2 += x1; y2 += y1;
			cv::Rect rect(x1, y1, x2-x1, y2-y1);
			cv::Rect img_rect = cv::Rect(0, 0, img.size().width, img.size().height);
			if ((rect | img_rect) != img_rect) {
				CV_Error(cv::Error::Code::StsBadArg, "the bounding box is not "
						 "contained in the image");
			}
			img = img(rect).clone();
			CV_Assert(img.isContinuous());
			CV_Assert(!img.isSubmatrix());

			mutex.lock();
				ids.push_back(id);
				imgs.push_back(img);
				labels.push_back(label);
			mutex.unlock();

			// NOTE: is this reallly needed?
			if (line_end == end) {
				break;
			}
			line_begin = line_end + 1;
			line_end = std::find(line_begin, end, '\n');
		} while (line_end != end);
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

#pragma mark Entry Point

void terminate_handler() {
	// TODO: OpenCV exception cought should return their error code.
	std::exception_ptr ex = std::current_exception();
	try {
		std::rethrow_exception(ex);
	} catch (const std::bad_alloc& ex) {
		CV_LOG_FATAL(NULL, "unable to allocate memory: " << ex.what());
		std::exit(cv::Error::Code::StsNoMem);
	} catch (const std::exception& ex) {
		CV_LOG_FATAL(NULL, "termination due to uncaught exception: " << ex.what());
		std::abort();
	} catch (...) {
		CV_LOG_FATAL(NULL, "something else was thrown\n");
		std::abort();
	}
}

int main() {
	std::terminate_handler old_terminate_handler = std::set_terminate(terminate_handler);
	
	std::vector<int64> ids;
	std::vector<cv::Mat> imgs;
	std::vector<int16_t> labels;

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

	{
		cv::Mutex mutex; // TODO: use std::mutex instead of std::recursive_mutex.
		int num_threads = cv::getNumThreads();
		// TODO: why is it necessary to specify 12 as the number of nstripes?

		std::ifstream csv_stream("itsabouttime/data/coco_final.csv");
		const char *base_dir = "itsabouttime/data/coco/";
		int base = 10;
		ParallelRead coco_reader(std::move(csv_stream), base_dir, base, mutex, ids, imgs, labels);
		cv::parallel_for_(cv::Range(0, coco_reader.csv.size()), coco_reader, 12);

		csv_stream.open("itsabouttime/data/openimg_final.csv");
		base_dir = "itsabouttime/data/openimg/";
		base = 16;
		ParallelRead openimg_reader(std::move(csv_stream), base_dir, base, mutex, ids, imgs, labels);
		cv::parallel_for_(cv::Range(0, openimg_reader.csv.size()), openimg_reader, 12);
	}

	ids.shrink_to_fit();
	imgs.shrink_to_fit();
	labels.shrink_to_fit();
	
	bool debug = true; // TODO: this should be passed from the command line.
	while (true) {
		void* module = dlopen("build/Debug/libalgorithm.dylib", RTLD_NOW);
		if (module == NULL) {
			CV_LOG_FATAL(NULL, dlerror());
			return cv::Error::StsBadArg;
		}
		bool (* run_classifier)(bool, void *, void *) = (bool (*)(bool, void *, void *))dlsym(module, "run_classifier");
		if (run_classifier == NULL) {
			CV_LOG_FATAL(NULL, dlerror());
			return cv::Error::StsBadArg;
		}
		if (!run_classifier(debug, (void *)&imgs, (void *)&labels)) {
			break;
		}
		if (dlclose(module) == -1) {
			CV_LOG_FATAL(NULL, dlerror());
			return cv::Error::StsError;
		}
	}
	
	if (false) {
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
			size_t tot_correct = 0, tot_skipped = 0;
			// TODO: enrich the api to enable this usecase.
			{
//				ParallelClassification classifier(imgs, labels, correct, skipped,
//					it.params[0].current.i,
//					it.params[1].current.d,
//					it.params[2].current.d,
//					it.params[3].current.i,
//					it.params[4].current.i,
//					it.params[5].current.d,
//					it.params[6].current.d
//				);
//				cv::parallel_for_(cv::Range(0, dataset_size), classifier);
			}
			float score = .5;
			std::cout
				<< "correct: " << score << '\n'
				<< "skipped: " << (float) tot_skipped / imgs.size() << '\n';
			if (score > best_score) {
				std::cout << "**** best: " << score << '\n';
				best_score = score;
				std::copy(it.params, it.params + it.length, best_params);
			}
		}

		std::cout << "best score: " << best_score << '\n';
		for (unsigned i = 0; i < params_length; ++i) {
			std::cout << best_params[i] << ' ';
		}
		std::cout << '\n';

		return 0;
	}

	return 0;
}
