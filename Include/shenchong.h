#pragma once

uchar rgb2gray(uint r, uint g, uint b);

cv::Mat convertMVImage(MVImage & m_image);

cv::Mat convertMVImageColor(MVImage & m_image);

//返回image中灰度为gray的像素个数
uint pixelCount(cv::Mat &image, uchar gray);

LPCWSTR stringToLPCWSTR(std::string orig);

struct ListEntry {
	std::string name;
	uchar count;
	ListEntry(std::string n, uchar c) :name(n), count(c) {}
};

/*
----------------------------------------------------------------------------------------------------------
阈值相关
*/

//返回图像直方图，是一个256维的array
std::array<uint, 256> getHist(cv::Mat & image);

//直接在图像上阈值分割，大于th的作为白色（255），其余为黑（0），若inv项为true则反相
void Threshold(cv::Mat& image, uint th, bool inv = false);

//返回新图的阈值分割，大于th的作为白色（255），其余为黑（0），若inv项为true则反相
cv::Mat ThresholdNew(cv::Mat& image, uint th, bool inv = false);

//返回图像的双峰阈值
uint TwinPeakThreshold(cv::Mat& image);

//返回迭代阈值
uint IterThreshold(cv::Mat& image);
/*
----------------------------------------------------------------------------------------------------------
图像矩相关
*/

//返回Hu矩
void HuMoments(cv::Mat& image, double output[7]);

//对Hu矩数值的对数化运算
double inline HuMomentsLog(double input) {
	return -copysign(1.0, input) * log10(abs(input));
}

/*
----------------------------------------------------------------------------------------------------------
分类相关
*/

//样本
struct SampleEntry {
	//类别名
	std::string name;
	//类别标号
	uint code;
	//特征向量
	std::vector<double> featVector;

	SampleEntry(std::string str, std::vector<double> vect, uint c);
};

/*
样本管理器
*/
class SampleEntryManager {
private:
	std::list<SampleEntry> entries;
public:
	//从灰度图像产生特征向量
	std::vector<double> getFeat(cv::Mat& image);
	//将一个灰度图像加入样本集并标记为mark
	void addSample(cv::Mat& image, std::string const mark, uint code);
	/*
	读取每一行格式为
	文件名 标记名
	的txt文件，并依照每一行添加样本
	*/
	void addSamplesFromFile(std::string const filename);
	//对一个灰度图像进行识别，返回最接近的样本的指针
	SampleEntry* classify(cv::Mat& image);
};

/*
采用距离分到就近训练样本的分类器
sample : 输入样本
data : 已知样本
distFunc : 距离函数指针
返回：最近的训练样本
*/
SampleEntry* nearestClassifier(std::vector<double> sample, std::list<SampleEntry>& data, double(*distFunc)(std::vector<double>, std::vector<double>));

//欧氏距离,noSqr为true则不开方(废弃)
//double euclideanDist(std::vector<double> v1, std::vector<double> v2, bool noSqr);

//向量的p范数
double pNorm(std::vector<double> v, uint p, bool noRoot);

/*
采用p范数的距离计算函数模板
用例：auto dist  = pNormDist<2, false>(v1,v2); //计算v1与v2的欧氏距离
noRoot 为true则不进行开方
*/
template <uint p, bool noRoot>
double pNormDist(std::vector<double> v1, std::vector<double> v2) {
	if (v1.size() != v2.size())
		return std::numeric_limits<double>::quiet_NaN();
	auto len = v1.size();
	std::vector<double> diff(v1);
	for (int i = 0; i < len; i++) {
		diff[i] -= v2[i];
	}
	return pNorm(diff, p, noRoot);
}

/*
----------------------------------------------------------------------------------------------------------
特征点相关
*/
struct HarrisPointEntry {
	cv::Point pos;
	double lp, ln;
	cv::Vec2d vecp, vecn;
};

/*Harris角点
目测阈值黑白8e5，灰度3e4比较好
*/
std::list<HarrisPointEntry> HarrisPoint(cv::Mat image, double k = 0.04, double rTh = 2e12);

/*
----------------------------------------------------------------------------------------------------------
分割相关
*/
//迭代骨架凹点法
void multiSkeletonConcaveDivision(cv::Mat & image);
//曲率法
void curvatureConcaveDivision(cv::Mat & image);
//骨架提取
cv::Mat skeletonize(cv::Mat & image);

std::vector<double> getCurvature(std::vector<cv::Point> const & vecContourPoints, int step);
