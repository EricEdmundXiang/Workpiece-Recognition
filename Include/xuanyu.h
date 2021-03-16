#pragma once
//存储璇鱼写的相关的函数

struct parted {
	cv::Mat pic;           //分割后的图像
	cv::Mat thinPic;       //细化后的图像
	cv::Point zuoshang;    //对应原图的左上角坐标
	int height;            //分割出来图像高
	int width;             //分割出来图像宽
	std::string name; 
	int mark;
	int pixNum;
	bool bian;
};


void thinImage(cv::Mat & srcImg);
//功能：骨架提取
//输入：Mat的引用
//返回：无

std::list<cv::Point> seed_grow(cv::Mat &srcImg, bool fromCenter = false);
//功能：区域生长，只是一个test函数，将输入srcImg的最左上角的白色区域“生长”为COLOR色。用于分割,返回生长出来的点集
//输入：Mat
//输出：Mat

int binary_num(const cv::Mat srcImg);
//计算双峰阈值二值化对应的阈值

void binary_img(cv::Mat & srcImg);
//利用双峰阈值进行二值化,使用了函数binary_num获取阈值

//std::list<cv::Mat> separation(cv::Mat srcImg);
//分割图片中的工件，保存到vector数组中

std::list<parted> separation(cv::Mat srcImg, bool & errorMark, cv::Point offset = cv::Point(0, 0), std::pair<bool, bool> bianOver = std::make_pair<bool,bool>(false,false));
//分割图片中的工件，保存到vector数组中
//返回结构列表

parted grow_part(cv::Mat & srcImg,cv::Point ptGrow,bool & errorMark);
//返回被分割出来的工件，若分割出来是噪音，返回的Mat是1*1

cv::Mat shuchuxiangsuge(const cv::Mat srcImg);
//用于test，扩大像素点用于观察

int recognition1(cv::Mat srcImg);
//识别工件类型，将类型打印在细化图像上。圆返回1，螺母2，L和螺钉返回0
//返回识别类型

//xuanyu函数的接口函数


cv::Mat robert(cv::Mat srcImg);//Rober算子边缘检测

bool screw_rec(cv::Mat srcImg);//平头螺钉和尖头螺钉识别，平头返回0，尖头返回1

cv::Mat PCA_rotate(std::list<cv::Point> listP,cv::Mat srcImg); //旋转图像，使最大PCA方向竖直,输入图像高度和宽度

void quzhan(cv::Mat & src);

void hough_test_new(cv::Mat & src);

void hough_test(cv::Mat & src);

void yu_chu_li(cv::Mat & src);