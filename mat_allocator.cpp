#include <opencv2/core/mat.hpp>
#include <opencv2/core/core_c.h>
#include <iostream>

// Copied from core/src/matrix.cpp the StdMatAllocator class
struct MyMatAllocator CV_FINAL : public cv::MatAllocator {
	cv::UMatData* allocate(
		int dims,
		const int* sizes,
		int type,
		void* data0,
		size_t* step,
		cv::AccessFlag /*flags*/,
		cv::UMatUsageFlags /*usageFlags*/
	) const CV_OVERRIDE {
		size_t total = CV_ELEM_SIZE(type);
		for (int i = dims-1; i >= 0; i--) {
			if (step) {
				if (data0 && step[i] != CV_AUTOSTEP) {
					CV_Assert(total <= step[i]);
					total = step[i];
				}
				else
					step[i] = total;
			}
			total *= sizes[i];
		}
		std::cout << "Allocating " << total << " bytes\n";
		uchar* data = data0 ? (uchar*)data0 : (uchar*)cv::fastMalloc(total);
		cv::UMatData* u = new cv::UMatData(this);
		u->data = u->origdata = data;
		u->size = total;
		if(data0)
			u->flags |= cv::UMatData::USER_ALLOCATED;

		return u;
	}

	bool allocate(
		cv::UMatData* u,
		cv::AccessFlag /*accessFlags*/,
		cv::UMatUsageFlags /*usageFlags*/
	) const CV_OVERRIDE {
		if(!u) return false;
		return true;
	}

	void deallocate(cv::UMatData* u) const CV_OVERRIDE {
		if(!u)
			return;

		CV_Assert(u->urefcount == 0);
		CV_Assert(u->refcount == 0);
		if (!(u->flags & cv::UMatData::USER_ALLOCATED)) {
			std::cout << "Deallocating " << u->size << " bytes\n";
			cv::fastFree(u->origdata);
			u->origdata = 0;
		}
		delete u;
	}
};

int main() {
	MyMatAllocator allocator;
	cv::Mat::setDefaultAllocator(&allocator);
	cv::Mat m = cv::Mat::ones(10, 10, CV_8UC1);
	std::cout << m << '\n';
	m = cv::Mat::zeros(10, 10, CV_8UC1);
	m = cv::Mat::zeros(10, 11, CV_8UC1);
	m.at<char>(0, 0) = 1;
	std::cout << m << '\n';
}
