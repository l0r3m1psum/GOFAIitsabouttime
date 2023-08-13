static void
hconcat(
		const cv::Mat& left,
		const cv::Mat& right,
		cv::Mat& tmp,
		cv::Mat& res
	) {
	int left_height = left.size().height;
	int right_height = right.size().height;
	bool left_smaller = left_height < right_height;
	const cv::Mat& small_one = left_smaller ? left : right;
	int diff = left_smaller ? right_height - left_height : left_height - right_height;

	{
		int top = diff, bottom = 0, left = 0, right = 0;
		int borderType = cv::BORDER_CONSTANT | cv::BORDER_ISOLATED;
		cv::Scalar borderValue(0);
		cv::copyMakeBorder(
			small_one, tmp,
			top, bottom, left, right,
			borderType,
			borderValue
		);
	}

	const cv::Mat& padded_left = left_smaller ? tmp : left;
	const cv::Mat& padded_right = left_smaller ? right : tmp;
	cv::hconcat(padded_left, padded_right, res);
}

static void
explore_dataset(
		const char *window_name,
		const char *window_title,
		const std::vector<cv::Mat>& imgs
	) {
	// TODO: a lot of functions take String refs so there might be a lot of
	// unnecessary allocations

	CV_Assert(window_name);
	CV_Assert(window_title);

	// TODO: We should dynamically set the title with "dataset id" to easily identify images.
	cv::setWindowTitle(window_name, window_title);

	int index = 0;
	int index_max = imgs.size() - 1; // NOTE: this is a conversion from size_t to int
	cv::createTrackbar("index", window_name, nullptr, index_max, nullptr, &index);
	int delta = 0;
	cv::setMouseCallback(
		window_name,
		[](int event, int x, int y, int flags, void *userdata) -> void {
			int *res = (int *)userdata;
			if (event == cv::EVENT_MOUSEWHEEL) {
				// NOTE: this on macOS is broken.
				*res = cv::getMouseWheelDelta(flags);
			}
		},
		&delta);

	cv::Mat concatenated;
	// TODO: set this threshold as a function of the resolution of the image.
	// This function shall be learned from the data.
	constexpr int thresh = 200;
	std::vector<std::vector<cv::Point>> contours;

	for (;;) {
		const cv::Mat& img = imgs[index];
		cv::Mat work_img(img.clone());

		// https://forum.opencv.org/t/detect-an-ellipse-and-get-its-dimensions/6949
		cv::blur(work_img.clone(), work_img, cv::Size(2,2));
		cv::Canny(work_img.clone(), work_img, thresh, thresh*2);
		cv::findContours(work_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		// This is my current heuristic for determining the frame of the clock.
		// This has some obvious problems, mainly the fact that looks only for
		// circular frames (squared ones for example are ignored).
		// Also some images like openimg 557 seems to have an ellipse that is
		// not even in the image. This can be solved by selecting only ellipses
		// for which their area is for some percentage in the image.
		std::vector<cv::RotatedRect> minRect(contours.size()), minEllipse(contours.size());
		for (size_t i = 0; i < contours.size(); ++i) {
			minRect[i] = cv::minAreaRect(contours[i]);
			if (contours[i].size() > 5) {
				minEllipse[i] = cv::fitEllipse(contours[i]);
			}
		}
		auto max = max_element(minEllipse.begin(), minEllipse.end(),
			[](cv::RotatedRect lhs, cv::RotatedRect rhs) -> bool {
				return lhs.size.area() < rhs.size.area();
			});
		if (max != minEllipse.end()) {
			cv::Mat canvas(img.clone());
			cv::ellipse(canvas, *max, cv::Scalar(0xAA, 0xAA, 0x0), 1);
			cv::rectangle(canvas, max->boundingRect(), cv::Scalar(255,0,0), 1);

			// https://theailearner.com/2020/11/06/perspective-transformation/
			cv::Point2f points[4];
			max->points(points);

			// BGR
			cv::line(canvas, points[0], points[2], cv::Scalar(0,0xff,0));
			cv::line(canvas, points[1], points[3], cv::Scalar(0,0,0xff));

			cv::Point2f bottomLeft = points[0], topLeft = points[1],
				topRight = points[2], bottomRight = points[3];
			cv::Point2f height_sides = bottomLeft - topLeft;
			float height = cv::sqrt(height_sides.x*height_sides.x + height_sides.y*height_sides.y);
			cv::Point2f width_sides = bottomLeft - bottomRight;
			float width = cv::sqrt(width_sides.x*width_sides.x + width_sides.y*width_sides.y);
			float side = cv::min(height, width);
			cv::Mat transform = cv::getPerspectiveTransform(points,
				(cv::Point2f[4]){{0, 0},{0, side},{side, side},{side, 0}});
			cv::warpPerspective(canvas, work_img, transform, cv::Size(side, side));
			cv::Mat tmp = cv::Mat();
			hconcat(
				canvas,
				work_img,
				tmp,
				concatenated
			);
		} else {
			// TODO: make this an image with an error written on top.
			concatenated = cv::Mat();
		}

		cv::imshow(window_name, concatenated);
		switch (cv::pollKey()) {
		case 'q': return;
		}
	}
}

// This struct can go wrong in so many ways... Almost none of them are checked.
// So please be carefull when using it.
struct Param {
	enum struct Tag {
		BOOL,
		INT,
		FLOAT,
		DOUBLE,
	};
	union Val {
		bool b;
		int i;
		float f;
		double d;
		Val(bool b)   { this->b = b; }
		Val(int i)    { this->i = i; }
		// TODO: check for invalid numbers like nan and so on.
		Val(float f)  { this->f = f; }
		Val(double d) { this->d = d; }
		Val() { this->d = 0.; }
	};

	Tag tag;
	Val min;
	Val max;
	Val step;
	Val current;

	// TODO: check that step does not cause any weird overflow.
	bool can_grow() {
		switch (tag) {
		case Tag::BOOL:   return current.b == false;
		case Tag::INT:    return current.i + step.i < max.i;
		case Tag::FLOAT:  return current.f + step.f < max.f;
		case Tag::DOUBLE: return current.d + step.d < max.d;
		default: CV_Error(cv::Error::Code::StsBadArg, "bad tag");
		}
	}

	void grow() {
		switch (tag) {
		case Tag::BOOL:   current.b = true; break;
		case Tag::INT:    current.i += step.i; break;
		case Tag::FLOAT:  current.f += step.f; break;
		case Tag::DOUBLE: current.d += step.d; break;
		default: CV_Error(cv::Error::Code::StsBadArg, "bad tag");
		}
	}

	void init() {
		switch (tag) {
		case Tag::BOOL:   current.b = false; break;
		case Tag::INT:    current.i = min.i; break;
		case Tag::FLOAT:  current.f = min.f; break;
		case Tag::DOUBLE: current.d = min.d; break;
		default: CV_Error(cv::Error::Code::StsBadArg, "bad tag");
		}
	}

	// Returning an int from here would require checking in advance that the
	// operation performed does not overflow said integer.
	int iterations_required() {
		switch (tag) {
		case Tag::BOOL:   return 2;
		// There may be some -1 in this calculation
		case Tag::INT:    return (max.i - min.i)/step.i;
		case Tag::FLOAT:  return cvFloor((max.f - min.f)/step.f);
		case Tag::DOUBLE: return  cvFloor((max.d - min.d)/step.d);
		default: CV_Error(cv::Error::Code::StsBadArg, "bad tag");
		}
	}
};

std::ostream& operator <<(std::ostream& os, Param p) {
	switch (p.tag) {
	case Param::Tag::BOOL:   return os << p.current.b;
	case Param::Tag::INT:    return os << p.current.i;
	case Param::Tag::FLOAT:  return os << p.current.f;
	case Param::Tag::DOUBLE: return os << p.current.d;
	default: CV_Error(cv::Error::Code::StsBadArg, "bad tag");
	}
}

Param params[] = {
	{Param::Tag::BOOL,  Param::Val(true), Param::Val(false), Param::Val(true)},
	{Param::Tag::INT,   Param::Val(0),    Param::Val(10),    Param::Val(3)},
	{Param::Tag::FLOAT, Param::Val(1.f),  Param::Val(4.f),   Param::Val(1.f)}
};

// TODO: extract the parameters from my algorithm and try to optimize them.

Param optimize_parameters(Param *params, int length) {
	CV_Assert(params);
	CV_Assert(length > 0);

	long long tot = 1;
	for (int i = 0; i < length; ++i) {
		tot *= params[i].iterations_required()
		params[i].init();
	}
	CV_LOG_INFO(NULL, "Parameter space size: " << tot);
	Param best = params[0];

	while (true) {
		// Visit.
		std::cout << '[';
		for (int i = 0; i < length - 1; ++i) {
			std::cout << params[i] << ", ";
		}
		std::cout << params[length - 1] << "]\n";

		int i = 0;
		while (!params[i].can_grow() & (i < length)) {
			params[i].init();
			++i;
		}
		if (i == length) {
			break;
		}
		params[i].grow();
	}

	return best;
}

void highPassFileters() {
	// https://stackoverflow.com/questions/4993082/
	if (false) {
		cv::Matx33f kernel(
			-1, -1, -1,
			-1, 9, -1,
			-1, -1, -1
		);
		int ddepth = -1;
		cv::filter2D(copyTo(work_img, copied), work_img, ddepth, kernel);
		// cv::Mat tmp;
		// cv::GaussianBlur(work_img, tmp, cv::Size(0, 0), 3);
		// cv::addWeighted(tmp, 1.5, work_img, -0.5, 0, work_img);
	}
	if (false) {
		cv::Mat tmp;
		cv::GaussianBlur(work_img, tmp, cv::Size(0, 0), 2);
		cv::addWeighted(work_img, 2.2, tmp, -1, 0, work_img);
	}
}
