#include "pch.h"
#include "shenchong.h"

uchar inline rgb2gray(uint r, uint g, uint b) {
	return static_cast<uchar>((r * 76 + g * 150 + b * 30) >> 8);
}

cv::Mat convertMVImage(MVImage &m_image) {
	const int w = m_image.GetWidth();
	const int h = m_image.GetHeight();
	cv::Mat result(h, w, CV_8UC1);
	auto mptr = static_cast<uchar*>(m_image.GetBits());
	auto cvptr = result.ptr<uchar>(0);
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			*(cvptr++) = rgb2gray(*(mptr + 2), *(mptr + 1), *mptr);
			mptr += 3;
		}
	}
	return result;
}

cv::Mat convertMVImageColor(MVImage &m_image) {
	const int w = m_image.GetWidth();
	const int h = m_image.GetHeight();
	cv::Mat result(h, w, CV_8UC3);
	auto mptr = static_cast<cv::Vec3b*>(m_image.GetBits());
	auto cvptr = result.ptr<cv::Vec3b>(0);
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			*(cvptr++) = *(mptr++);
		}
	}
	return result;
}

uint pixelCount(cv::Mat &image, uchar gray)
{
	uint count = 0;
	for (auto ptr = image.data; ptr != image.dataend; ptr++) {
		count += *ptr == gray;
	}
	return count;
}

LPCWSTR stringToLPCWSTR(std::string orig) {
	size_t origsize = orig.length() + 1;
	const size_t newsize = 100;
	size_t convertedChars = 0;
	wchar_t wcstring[101];
	MultiByteToWideChar(CP_ACP, 0, orig.c_str(), -1, wcstring, newsize);
	//mbstowcs_s(&convertedChars, wcstring, origsize, orig.c_str(), _TRUNCATE);
	return wcstring;
}

std::array<uint, 256> getHist(cv::Mat & image) {
	std::array<uint, 256> result = { 0 };
	for (auto ptr = image.data; ptr != image.dataend; ptr++)
		result[*ptr]++;
	return result;
}


void Threshold(cv::Mat& image, uint th, bool inv) {
	auto ptr = image.ptr<uchar>(0);
	for (auto ptr = image.data; ptr != image.dataend; ptr++)
		*ptr = ((*ptr > th) ^ inv) ? 255 : 0;
}


cv::Mat ThresholdNew(cv::Mat& image, uint th, bool inv) {
	auto result = image.clone();
	Threshold(result, th, inv);
	return result;
}


uint TwinPeakThreshold(cv::Mat& image) {
	const int divTh = 20;
	auto hist = getHist(image);
	int pxcnt = 0;
	int midpxcnt = (image.rows * image.cols) >> 1;

	uint average = 0;
	int median;

	for (int i = 0; i < 256; i++) {
		average += hist[i] * i;
	}
	average /= image.cols * image.rows;

	for (int i = 0; i < 256; i++) {
		if (pxcnt + hist[i] >= midpxcnt) {
			median = i;
			break;
		}
		pxcnt += hist[i];
	}
	uint lmax, lmaxp;
	uint rmax, rmaxp;

	do {
		lmax = rmax = 0;
		for (int i = 0; i < average; i++) {
			if (hist[i] > lmax) {
				lmax = hist[i];
				lmaxp = i;
			}
		}

		for (int i = average; i < 256; i++) {
			if (hist[i] > rmax) {
				rmax = hist[i];
				rmaxp = i;
			}
		}

		if (rmaxp - lmaxp < divTh) {
			if (median > average)
				average = rmaxp - divTh;
			else
				average = lmaxp + divTh;
		}
		else break;
	} while (true);

	//uint temp = lmax + rmax;
	return (lmaxp + rmaxp) >> 1;
	//return (lmaxp * lmax / temp + rmaxp * rmax / temp);
}


uint IterThreshold(cv::Mat& image) {
	auto hist = getHist(image);

	std::array<uint64, 256> sum = { 0 };
	for (int i = 1; i < 256; i++) {
		sum[i] = sum[i - 1] + hist[i] * i;
		hist[i] += hist[i - 1];
	}

	uint th = sum[255] / hist[255];
	uint lastth = 0;

	while (true) {
		uint averL = sum[th] / hist[th];
		uint averR = (sum[255] - sum[th]) / (hist[255] - hist[th]);
		th = (averL + averR) >> 1;
		if (th == lastth) {
			return th;
		}
		else {
			lastth = th;
		}
	}
}

void HuMoments(cv::Mat& image, double output[7]) {
	ULONG64 m00 = 0,
		m10 = 0,
		m01 = 0,
		m20 = 0,
		m11 = 0,
		m02 = 0,
		m30 = 0,
		m21 = 0,
		m12 = 0,
		m03 = 0;
	auto ptr = image.ptr<uchar>(0);
	for (uint y = 0; y < image.rows; y++) {
		for (uint x = 0; x < image.cols; x++) {
			ULONG64 val = *ptr;
			m00 += val;
			m10 += val * x;
			m01 += val * y;
			m20 += val * x * x;
			m11 += val * x * y;
			m02 += val * y * y;
			m30 += val * x * x * x;
			m21 += val * x * x * y;
			m12 += val * x * y * y;
			m03 += val * y * y * y;
			ptr++;
		}
	}

	double cx = m10 * 1.0 / m00,
		cy = m01 * 1.0 / m00;

	double mu00 = m00;
	double mu20 = m20 - m10 * cx,
		mu11 = m11 - m10 * cy,
		mu02 = m02 - m01 * cy;
	double mu30 = m30 - cx * (3 * mu20 + cx * m10),
		mu21 = m21 - cx * (2 * mu11 + cx * m01) - cy * mu20,
		mu12 = m12 - cy * (2 * mu11 + cy * m10) - cx * mu02,
		mu03 = m03 - cy * (3 * mu02 + cy * m01);

	double mu00_2 = mu00 * mu00;
	double mu00_3 = mu00_2 * sqrt(mu00);

	double nu20 = mu20 / mu00_2,
		nu02 = mu02 / mu00_2,
		nu11 = mu11 / mu00_2,
		nu30 = mu30 / mu00_3,
		nu12 = mu12 / mu00_3,
		nu21 = mu21 / mu00_3,
		nu03 = mu03 / mu00_3;

	double nu30_m_nu12_3 = nu30 - 3 * nu12,
		nu21_3_m_nu03 = 3 * nu21 - nu03,
		nu30_p_nu12 = nu30 + nu12,
		nu21_p_nu03 = nu21 + nu03,
		nu20_m_nu02 = nu20 - nu02;

	output[0] = nu20 + nu02;
	output[1] = (nu20 - nu02) * (nu20 - nu02) + 4.0 * nu11 * nu11;
	output[2] = nu30_m_nu12_3 * nu30_m_nu12_3 + nu21_3_m_nu03 * nu21_3_m_nu03;
	output[3] = nu30_p_nu12 * nu30_p_nu12 + nu21_p_nu03 * nu21_p_nu03;
	output[4] = nu30_m_nu12_3 * nu30_p_nu12 * (nu30_p_nu12 * nu30_p_nu12 - 3.0 * nu21_p_nu03 * nu21_p_nu03)
		+ nu21_3_m_nu03 * nu21_p_nu03 * (3.0 * nu30_p_nu12 * nu30_p_nu12 - nu21_p_nu03 * nu21_p_nu03);
	output[5] = nu20_m_nu02 * (nu30_p_nu12 * nu30_p_nu12 - nu21_p_nu03 * nu21_p_nu03) + 4.0 * nu11 * nu30_p_nu12 * nu21_p_nu03;
	output[6] = nu21_3_m_nu03 * nu30_p_nu12 * (nu30_p_nu12 * nu30_p_nu12 - 3.0 * nu21_p_nu03 * nu21_p_nu03)
		- nu30_m_nu12_3 * nu21_p_nu03 * (3.0 * nu30_p_nu12 * nu30_p_nu12 - nu21_p_nu03 * nu21_p_nu03);
}


SampleEntry * nearestClassifier(std::vector<double> sample, std::list<SampleEntry> &data, double(*distFunc)(std::vector<double>, std::vector<double>)) {
	double mindist = INFINITY;
	SampleEntry * min = nullptr;
	for (auto &i : data) {
		auto dist = (*distFunc)(i.featVector, sample);
		if (dist < mindist) {
			mindist = dist;
			min = &i;
		}
	}

	return min;
}
/*
double euclideanDist(std::vector<double> v1, std::vector<double> v2, bool noSqr)
{
	if (v1.size() != v2.size())
		return std::numeric_limits<double>::quiet_NaN();
	auto len = v1.size();
	double result = 0;
	for (int i = 0; i < len; i++) {
		result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	if (!noSqr)
		result = sqrt(result);
	return result;
}
*/
double pNorm(std::vector<double> v, uint p, bool noRoot) {
	if (p == 0)
		return std::numeric_limits<double>::quiet_NaN();
	auto len = v.size();
	double result = 0;
	for (int i = 0; i < len; i++) {
		double temp = 1;
		for (int j = 0; j < p; j++) {
			temp *= abs(v[i]);
		}
		result += temp;
	}
	if (!noRoot) {
		result = pow(result, 1.0 / p);
	}
	return result;
}

SampleEntry::SampleEntry(std::string str, std::vector<double> vect, uint c) : name(str), featVector(vect), code(c)
{
}

std::list<HarrisPointEntry> HarrisPoint(cv::Mat image, double k, double rTh)
{
	const uchar localSize = 3;

	cv::Mat_<double> lambdaP(image.rows, image.cols);
	cv::Mat_<double> lambdaN(image.rows, image.cols);
	cv::Mat_<cv::Vec2d> eigenvectorP(image.rows, image.cols);
	cv::Mat_<cv::Vec2d> eigenvectorN(image.rows, image.cols);
	cv::Mat gradx, grady;
	std::list<HarrisPointEntry> result;
	HarrisPointEntry p;

	//求梯度
	cv::Sobel(image, gradx, CV_64F, 1, 0, 3);
	cv::Sobel(image, grady, CV_64F, 0, 1, 3);
	//cv::copyMakeBorder(gradx, gradx, );
	for (int y = 2; y < image.rows - 2; y++) {
		for (int x = 2; x < image.cols - 2; x++) {
			double h11, h12, h21, h22;
			h11 = h12 = h21 = h22 = 0;
			for (int i = -1; i <= 1; i++) {
				for (int j = -1; j <= 1; j++) {
					auto ix = gradx.at<double>(y + i, x + j);
					auto iy = grady.at<double>(y + i, x + j);
					h11 += ix * ix;
					h12 += ix * iy;
					h21 += ix * iy;
					h22 += iy * iy;
				}
			}

			auto temp = sqrt(4 * h12 * h21 + (h11 - h22) * (h11 - h22));
			auto lp = (h11 + h22 + temp) / 2;
			auto ln = (h11 + h22 - temp) / 2;
			lambdaN.at<double>(y, x) = ln;
			lambdaP.at<double>(y, x) = lp;

			auto r = lp * ln - k * (lp + ln) * (lp + ln);
			if (r > rTh) {
				eigenvectorP.at<cv::Vec2d>(y, x) = cv::Vec2d(h12 * h12 / ((h11 - lp) * (h11 - lp) + h12 * h12),
					(h11 - lp) * (h11 - lp) / ((h11 - lp) * (h11 - lp) + h12 * h12));
				eigenvectorN.at<cv::Vec2d>(y, x) = cv::Vec2d(h12 * h12 / ((h11 - ln) * (h11 - ln) + h12 * h12),
					(h11 - ln) * (h11 - ln) / ((h11 - ln) * (h11 - ln) + h12 * h12));

				p.pos = cv::Point(x, y);
				p.ln = lambdaN.at<double>(y, x);
				p.lp = lambdaP.at<double>(y, x);
				p.vecn = eigenvectorN.at<cv::Vec2d>(y, x);
				p.vecp = eigenvectorP.at<cv::Vec2d>(y, x);

				result.push_back(p);
			}
			else {
				lambdaN.at<double>(y, x) = 0;
				lambdaP.at<double>(y, x) = 0;
			}
		}
	}

	//非极大值抑制
	result.remove_if(
		[&](HarrisPointEntry poi) {
		auto x = poi.pos.x;
		auto y = poi.pos.y;
		auto ptr = &lambdaN.at<double>(y, x);
		for (int i = -1; i <= 1; i++) {
			for (int j = -1; j <= 1; j++) {
				if ((i || j) && (*(ptr + i * lambdaN.cols + j) >= *ptr)) {
					return true;
				}
			}
		}
		return false;
	}
	);

	//cv::imshow("lambdaP", lambdaP);
	//cv::imshow("lambdaN", lambdaN);

	return result;
}

void multiSkeletonConcaveDivision(cv::Mat& image) {
	//for 100% image
	const int begini = 2;
	const int step = 2;
	const int endi = 10;
	std::list<cv::Point> result;

	cv::Mat contour;//边缘
	cv::Mat thinned;//骨架
	cv::Mat invDial;//反相
	auto elem = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	cv::dilate(image, contour, cv::Mat());
	cv::subtract(contour, image, contour);

	cv::subtract(cv::Mat::ones(image.size(), CV_8UC1) * 255, image, invDial);

	for (int i = begini; i < endi; i += step) {
		result.clear();

		cv::dilate(invDial, thinned, elem, cv::Point(-1, -1), i);

		thinned = skeletonize(thinned);
		//thinImage(thinned);

		auto conPtr = contour.ptr(0);
		auto invPtr = thinned.ptr(0);
		for (uint y = 0; y < contour.rows; y++) {
			for (uint x = 0; x < contour.cols; x++) {
				if (*conPtr == 255 && *invPtr == 255) {
					cv::Point cur(x, y);

					for (auto pt : result) {
						auto dist = cv::norm(pt - cur);
						if (dist < i * 1.9 && dist > i - step) {
							cv::line(image, pt, cur, cv::Scalar(0), 1, cv::LINE_4);
						}
					}

					result.push_back(cur);
				}

				conPtr++;
				invPtr++;
			}
		}
	}
}

void curvatureConcaveDivision(cv::Mat& image) {
	const int localRange = 5;
	const int curvStep = 4;
	const int th = 15;

	std::vector<std::vector<cv::Point> > contours;
	//std::vector<double> vecCurv;
	std::vector<int> interest;
	std::list<cv::Point> result;

	cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < contours.size(); i++) {
		interest.clear();

		auto& now = contours[i];

		std::vector<double> vecCurv = getCurvature(now, curvStep);

		for (int j = 0; j < now.size(); j++) {
			bool flag = true;
			for (int k = -localRange; k <= localRange; k++) {
				int cmpMark = j + k;
				cmpMark %= now.size();
				if (vecCurv[j] < vecCurv[cmpMark]) {
					flag = false;
					break;
				}
			}
			if (flag) {
				interest.push_back(j);
			}
		}

		for (int j = 0; j < interest.size(); j++) {
			bool flag = true;
			int nxtMark = (j + 1) % interest.size();
			int preMark = (j - 1) % interest.size();
			cv::LineIterator litr(image, now[interest[preMark]], now[interest[nxtMark]]);
			++litr;
			for (int k = 1; k < litr.count - 1; k++, ++litr) {
				if (image.at<uchar>(litr.pos()) == 255) {
					flag = false;
					break;
				}
			}
			if (flag) {
				result.push_back(now[interest[j]]);
			}
		}
	}

	for (auto pt = result.begin(); pt != result.end(); pt++) {
		for (auto cmp = pt; cmp != result.end(); cmp++) {
			auto dist = cv::norm(*pt - *cmp);
			if (dist < th) {
				cv::line(image, *pt, *cmp, cv::Scalar(0), 1);
			}
		}
	}

}

cv::Mat skeletonize(cv::Mat& image)
{
	auto img = image.clone();
	cv::Mat skel = cv::Mat::zeros(img.size(), CV_8UC1);
	cv::Mat temp;
	cv::Mat eroded;

	cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

	bool done;
	do
	{
		cv::erode(img, eroded, element);
		cv::dilate(eroded, temp, element); // temp = open(img)
		cv::subtract(img, temp, temp);
		cv::bitwise_or(skel, temp, skel);
		eroded.copyTo(img);

		done = (cv::countNonZero(img) == 0);
	} while (!done);
	return skel;
}

std::vector<double> getCurvature(std::vector<cv::Point> const& vecContourPoints, int step)
{
	std::vector< double > vecCurvature(vecContourPoints.size());

	if (vecContourPoints.size() < step)
		return vecCurvature;

	auto frontToBack = vecContourPoints.front() - vecContourPoints.back();
	bool isClosed = ((int)std::max(std::abs(frontToBack.x), std::abs(frontToBack.y))) <= 1;

	cv::Point2f pplus, pminus;
	cv::Point2f f1stDerivative, f2ndDerivative;
	for (int i = 0; i < vecContourPoints.size(); i++)
	{
		const cv::Point2f& pos = vecContourPoints[i];

		int maxStep = step;
		if (!isClosed)
		{
			maxStep = std::min(std::min(step, i), (int)vecContourPoints.size() - 1 - i);
			if (maxStep == 0)
			{
				vecCurvature[i] = std::numeric_limits<double>::infinity();
				continue;
			}
		}


		int iminus = (i - maxStep) % vecContourPoints.size();
		int iplus = (i + maxStep) % vecContourPoints.size();


		f1stDerivative.x = (pplus.x - pminus.x) / (iplus - iminus);
		f1stDerivative.y = (pplus.y - pminus.y) / (iplus - iminus);
		f2ndDerivative.x = (pplus.x - 2 * pos.x + pminus.x) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);
		f2ndDerivative.y = (pplus.y - 2 * pos.y + pminus.y) / ((iplus - iminus) / 2 * (iplus - iminus) / 2);

		double curvature2D;
		double divisor = f1stDerivative.x*f1stDerivative.x + f1stDerivative.y*f1stDerivative.y;
		if (std::abs(divisor) > 10e-8)
		{
			curvature2D = std::abs(f2ndDerivative.y*f1stDerivative.x - f2ndDerivative.x*f1stDerivative.y) /
				pow(divisor, 3.0 / 2.0);
		}
		else
		{
			curvature2D = std::numeric_limits<double>::infinity();
		}

		vecCurvature[i] = curvature2D;


	}
	return vecCurvature;
}

std::vector<double> SampleEntryManager::getFeat(cv::Mat& image)
{
	double hu[7];

	HuMoments(image, hu);

	for (int i = 0; i < 7; i++)
		hu[i] = HuMomentsLog(hu[i]);
	hu[6] = abs(hu[6]);

	return std::vector<double>(hu, hu + 6);
}

void SampleEntryManager::addSample(cv::Mat& image, std::string const mark, uint code)
{
	SampleEntry temp(mark, getFeat(image), code);
	entries.push_back(temp);
}

void SampleEntryManager::addSamplesFromFile(std::string const filename)
{
	std::ifstream file(filename);
	if (file.is_open()) {
		std::string name;
		std::string mark;
		cv::Mat image;
		uint code;
		while (file >> name) {
			file >> mark >> code;
			image = cv::imread(name, false);
			addSample(ThresholdNew(image, IterThreshold(image), true), mark, code);
		}
	}
}

SampleEntry* SampleEntryManager::classify(cv::Mat& image)
{
	return nearestClassifier(getFeat(image), entries, pNormDist<1, true>);
}
