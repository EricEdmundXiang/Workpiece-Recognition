#pragma once

uchar rgb2gray(uint r, uint g, uint b);

cv::Mat convertMVImage(MVImage & m_image);

cv::Mat convertMVImageColor(MVImage & m_image);

//����image�лҶ�Ϊgray�����ظ���
uint pixelCount(cv::Mat &image, uchar gray);

LPCWSTR stringToLPCWSTR(std::string orig);

struct ListEntry {
	std::string name;
	uchar count;
	ListEntry(std::string n, uchar c) :name(n), count(c) {}
};

/*
----------------------------------------------------------------------------------------------------------
��ֵ���
*/

//����ͼ��ֱ��ͼ����һ��256ά��array
std::array<uint, 256> getHist(cv::Mat & image);

//ֱ����ͼ������ֵ�ָ����th����Ϊ��ɫ��255��������Ϊ�ڣ�0������inv��Ϊtrue����
void Threshold(cv::Mat& image, uint th, bool inv = false);

//������ͼ����ֵ�ָ����th����Ϊ��ɫ��255��������Ϊ�ڣ�0������inv��Ϊtrue����
cv::Mat ThresholdNew(cv::Mat& image, uint th, bool inv = false);

//����ͼ���˫����ֵ
uint TwinPeakThreshold(cv::Mat& image);

//���ص�����ֵ
uint IterThreshold(cv::Mat& image);
/*
----------------------------------------------------------------------------------------------------------
ͼ������
*/

//����Hu��
void HuMoments(cv::Mat& image, double output[7]);

//��Hu����ֵ�Ķ���������
double inline HuMomentsLog(double input) {
	return -copysign(1.0, input) * log10(abs(input));
}

/*
----------------------------------------------------------------------------------------------------------
�������
*/

//����
struct SampleEntry {
	//�����
	std::string name;
	//�����
	uint code;
	//��������
	std::vector<double> featVector;

	SampleEntry(std::string str, std::vector<double> vect, uint c);
};

/*
����������
*/
class SampleEntryManager {
private:
	std::list<SampleEntry> entries;
public:
	//�ӻҶ�ͼ�������������
	std::vector<double> getFeat(cv::Mat& image);
	//��һ���Ҷ�ͼ����������������Ϊmark
	void addSample(cv::Mat& image, std::string const mark, uint code);
	/*
	��ȡÿһ�и�ʽΪ
	�ļ��� �����
	��txt�ļ���������ÿһ���������
	*/
	void addSamplesFromFile(std::string const filename);
	//��һ���Ҷ�ͼ�����ʶ�𣬷�����ӽ���������ָ��
	SampleEntry* classify(cv::Mat& image);
};

/*
���þ���ֵ��ͽ�ѵ�������ķ�����
sample : ��������
data : ��֪����
distFunc : ���뺯��ָ��
���أ������ѵ������
*/
SampleEntry* nearestClassifier(std::vector<double> sample, std::list<SampleEntry>& data, double(*distFunc)(std::vector<double>, std::vector<double>));

//ŷ�Ͼ���,noSqrΪtrue�򲻿���(����)
//double euclideanDist(std::vector<double> v1, std::vector<double> v2, bool noSqr);

//������p����
double pNorm(std::vector<double> v, uint p, bool noRoot);

/*
����p�����ľ�����㺯��ģ��
������auto dist  = pNormDist<2, false>(v1,v2); //����v1��v2��ŷ�Ͼ���
noRoot Ϊtrue�򲻽��п���
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
���������
*/
struct HarrisPointEntry {
	cv::Point pos;
	double lp, ln;
	cv::Vec2d vecp, vecn;
};

/*Harris�ǵ�
Ŀ����ֵ�ڰ�8e5���Ҷ�3e4�ȽϺ�
*/
std::list<HarrisPointEntry> HarrisPoint(cv::Mat image, double k = 0.04, double rTh = 2e12);

/*
----------------------------------------------------------------------------------------------------------
�ָ����
*/
//�����Ǽܰ��㷨
void multiSkeletonConcaveDivision(cv::Mat & image);
//���ʷ�
void curvatureConcaveDivision(cv::Mat & image);
//�Ǽ���ȡ
cv::Mat skeletonize(cv::Mat & image);

std::vector<double> getCurvature(std::vector<cv::Point> const & vecContourPoints, int step);
