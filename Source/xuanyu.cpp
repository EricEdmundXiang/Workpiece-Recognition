#include "pch.h"
#include"xuanyu.h"
#include"shenchong.h"

void thinImage(cv::Mat & srcImg)
{
//	std::vector<cv::Point> deleteList;
	std::list<cv::Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount += neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(cv::Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(cv::Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		//for (size_t i = 0; i < deleteList.size(); i++) {
		//	cv::Point tem;
		//	tem = deleteList[i];
		//	uchar* data = srcImg.ptr<uchar>(tem.y);
		//	data[tem.x] = 0;
		//}
		for (std::list<cv::Point>::iterator pw = deleteList.begin(); pw != deleteList.end(); pw++) {
			cv::Point tem;
			tem = *pw;
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}

std::list<cv::Point> seed_grow(cv::Mat &srcImg, bool fromCenter)
{
	const int COLOR = 100;//生长区域的颜色

	std::queue<cv::Point2i> q1;
	std::list<cv::Point> vecP; //生长出来的点集
	cv::Point2i ptCenter = { 0,0 };
	cv::Point2i ptGrowing = { 0,0 };

	int h = srcImg.rows;
	int w = srcImg.cols;

	int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	if (fromCenter) {
		q1.push(cv::Point(h / 2, w / 2));
	}
	else {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				if (srcImg.at<uchar>(i, j) == 255) {
					srcImg.at<uchar>(i, j) = COLOR;
					ptCenter.x = i;
					ptCenter.y = j;
					q1.push(ptCenter);
					break;//当扫描到第一个255的像素时，颜色设为COLOR并入列，推出内层for循环
				}
			}
			if (!q1.empty()) break;//当扫描到第一个255像素时，队列元素为1，退出外层for循环
		}
	}
	

	while (!q1.empty()) {
		ptCenter = q1.front();
		q1.pop();//删除第一个元素

		for (int i = 0; i < 8; i++) {
			ptGrowing.x = ptCenter.x + DIR[i][0];
			ptGrowing.y = ptCenter.y + DIR[i][1];

			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.y >(srcImg.cols - 1) || (ptGrowing.x > srcImg.rows - 1))
				continue;

			if (srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) == 255) {
				srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) = COLOR;
				vecP.push_back(ptGrowing);
				q1.push(ptGrowing);
			}
		}
	}
	return vecP;
}

int binary_num(const cv::Mat srcImg)
{
	int h = srcImg.rows;
	int w = srcImg.cols;

	int color = 0;//当前像素灰度级
	int maxColor = 0;//数目最多的灰度级
	int secColor = 0;//数目第二多的灰度级
	int maxNum = 0;//数目最多的灰度级的数目
	int secNum = 0;
	int peakNum = 0;//局部极大值点

	std::vector<cv::Point> ptPeak;
	cv::Point ptTemp = { 0,0 };

	int grayLevel[256] = { 0 };//统计各灰度级的像素个数

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			color = srcImg.at<uchar>(i, j);
			grayLevel[color] ++;
		}
	}//统计各灰度级数目

	//for (int i = 0; i < 256; i++) {
	//	std::cout << grayLevel[i] << std::endl;
	//}

	for (int i = 0; i < 254; i++) {
		if (grayLevel[i] >= peakNum) {
			peakNum = grayLevel[i];
			if (grayLevel[i + 1] < peakNum && grayLevel[i + 2] < peakNum) {
				ptTemp.x = peakNum;   //x记录数目
				ptTemp.y = i;         //y记录该数目对应的灰度级
				ptPeak.push_back(ptTemp);
			}
		}
	}//统计各极大值灰度级以及其对应的数目

	//取极大值vector数组中最大的和第二大的灰度级
	if (ptPeak[0].x > ptPeak[1].x) {
		maxNum = ptPeak[0].x;
		secNum = ptPeak[1].x;
		maxColor = ptPeak[0].y;
		secColor = ptPeak[1].y;
	}
	else {
		maxNum = ptPeak[1].x;
		secNum = ptPeak[0].x;
		maxColor = ptPeak[1].y;
		secColor = ptPeak[0].y;
	}

	for (int i = 2; i < ptPeak.size(); i++) {
		if (ptPeak[i].x > maxNum) {
			if (ptPeak[i].x > secNum) {
				secNum = maxNum;
				secColor = maxColor;
			}
			maxNum = ptPeak[i].x;
			maxColor = ptPeak[i].y;
		}
		else if(ptPeak[i].x == maxNum)
		{
			continue;
		}
		else if (ptPeak[i].x > secNum) {
			secNum = ptPeak[i].x;
			secColor = ptPeak[i].y;
		}
	}

	return (maxColor + secColor) / 2;
}

void binary_img(cv::Mat & srcImg)
{
	int threshold = binary_num(srcImg);
	int h = srcImg.rows;
	int w = srcImg.cols;

	for (int i = 0; i < h; i++)
		for (int j = 0; j < w; j++)
		{
			if ((srcImg.ptr<uchar>(i)[j]) > threshold)
			{
				(srcImg.ptr<uchar>(i)[j]) = 0;
			}
			else {
				(srcImg.ptr<uchar>(i)[j]) = 255;
			}
		}
}

std::list<parted> separation(cv::Mat srcImg,bool & errorMark, cv::Point offset, std::pair<bool,bool> bianOver)
{
	const int h = srcImg.rows;
	const int w = srcImg.cols;

	std::list<parted> lisParted;            //被分割出来的工件列表
	parted partedOne;                       //被分割出来的某一个工件
	cv::Point ptCenter = { 0,0 };           //区域生长的种子点

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			if (srcImg.at<uchar>(i, j) == 255) {
				srcImg.at<uchar>(i, j) = 0;
				ptCenter.x = i;
				ptCenter.y = j;
				partedOne = grow_part(srcImg,ptCenter,errorMark);
				partedOne.zuoshang += offset;
				if (bianOver.first)
					partedOne.bian = bianOver.second;
				if(errorMark == 1) return lisParted;          //错误
				else if (partedOne.pic.rows == 1) continue;   //检测到的为噪声
				else lisParted.push_back(partedOne);          //结构入队列
			}
		}
	}

	return lisParted;
}

parted grow_part(cv::Mat & srcImg, cv::Point ptGrow,bool & errorMark)
{
	parted partedOne = { cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)),cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)),{0,0},0,0,"",0,0,false}; //初始化

	std::queue<cv::Point> q1;                   //种子生长队列
	cv::Point ptGrowing = { 0,0 };              //生长领域点
	cv::Point ptCenter = { 0,0 };               //生长中心点
	const int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };

	int maxX = ptGrow.x, maxY = ptGrow.y;
	int minX = ptGrow.x, minY = ptGrow.y;       //用来记录工件区域的左上角右下角，以创相应size的Mat
	std::list<cv::Point> ltPixel;               //记录被分离出来的工件的像素点集合

	ltPixel.push_back(ptGrow);
	q1.push(ptGrow);

	while (!q1.empty()) {
		ptCenter = q1.front();          //取中心点
		q1.pop();                       //删除第一个元素

		for (int i = 0; i < 8; i++) {
			ptGrowing.x = ptCenter.x + DIR[i][0];
			ptGrowing.y = ptCenter.y + DIR[i][1];

			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.y >(srcImg.cols - 1) || (ptGrowing.x > srcImg.rows - 1))
			{
//				errorMark = 1;     //检测到有“对象”有部分在图像边界
//				return partedOne;  //退出函数
				partedOne.bian = true;
				continue;
			}

			if (srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) == 255) {
				srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) = 0;
				q1.push(ptGrowing);
				if (ptGrowing.x > maxX) maxX = ptGrowing.x;
				if (ptGrowing.y > maxY) maxY = ptGrowing.y;
				if (ptGrowing.x < minX) minX = ptGrowing.x;
				if (ptGrowing.y < minY) minY = ptGrowing.y;        //存左上角和右下角
				ltPixel.push_back(ptGrowing);                      //属于该工件区域的像素进入链表
			}
		}
	}

	if (ltPixel.size() < 100) return partedOne;            //生长区域点数小于30，噪音
	else {
		int partedH = maxX - minX + 10;                   //新图像的高
		int partedW = maxY - minY + 10;                   //新图像的宽
		cv::Mat partedPic = cv::Mat(partedH, partedW, CV_8UC1, cv::Scalar(0));

		for (std::list<cv::Point>::iterator pw = ltPixel.begin(); pw != ltPixel.end(); pw++) {
			partedPic.at<uchar>((*pw).x - minX + 5, (*pw).y - minY + 5) = 255;    //在新图中画工件
		}
		partedOne.pic = partedPic;
		partedOne.height = partedH;
		partedOne.width = partedW;
		partedOne.zuoshang = { minX,minY };
		partedOne.pixNum = ltPixel.size();
		
		return partedOne;        //返回分割出来的一个工件结构
	}
}

cv::Mat shuchuxiangsuge(const cv::Mat srcImg)
{
	int h = srcImg.rows;
	int w = srcImg.cols;
	cv::Mat daXiangSuGe = cv::Mat(4 * h, 4 * w, CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			if (srcImg.at<uchar>(i, j) == 255) {
				cv::rectangle(daXiangSuGe, cv::Rect(4 * j, 4 * i, 4, 4), cv::Scalar(255, 0, 0), 1, 1, 0);//扩大像素点
			}
		}
	}
	return daXiangSuGe;
}

int recognition1(cv::Mat srcImg)
{
	cv::Point startP = { 0,0 };     //记录扫描到的第一个点
	cv::Point lastP = { 0,0 };      //记录迭代过程中相对于currentP的上一个点
	cv::Point currentP = { 0,0 };   //迭代过程中当前点
	cv::Point aroundP = { 0,0 };    //记录顺时针扫描领域点

	const int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };//领域扫描顺序，顺时针
	const int DIR1[16][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} ,{-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	//用来保证转圈扫描的完整性；

	int L_Cx = 0;     //lastP - currentP的x之差
	int L_Cy = 0;     //lastP - currentP的y之差，方便环形扫描找lastP的顺时针下一个DIR元素
	int ilast = 0;    //对应lastP的DIR的i的值
	int roundNum = 0; //记录扫描迭代中迭代的次数
	bool yuan = 1;    //圆判断标志

	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) == 255) {
				startP.x = i;
				startP.y = j;
				lastP.x = i;
				lastP.y = j;
				currentP.x = i;
				currentP.y = j;//初始化三个点为扫描到的第一个点
				break;
			}
		}
		if (startP.x != 0) break;//说明已经找到了第一个点
	}

	for (int i = 0; i < 8; i++) {
		aroundP.x = currentP.x + DIR[i][0];
		aroundP.y = currentP.y + DIR[i][1];//取领域正上角为扫描起点顺时针扫描
		if (aroundP.x > 0 && aroundP.y > 0 && aroundP.x < srcImg.rows  && aroundP.y < srcImg.cols)
		{

			if (srcImg.at<uchar>(aroundP.x, aroundP.y) == 255) {
				yuan = 0;
				if (i == 0) continue;//不取左上角的点为起点
				currentP.x = aroundP.x;
				currentP.y = aroundP.y;
				break;//在领域扫描到第一个255点就结束循环
			}
		}
	}
	if (yuan) {
		//扫描一圈没有检测到领域255点，说明这个工件是圆！但是还有其他情况也是圆
		return 1;
	}
	else {
		while(1){//不断循环，直到判断出一个结果
			roundNum++;  //迭代次数加一
			L_Cx = lastP.x - currentP.x;
			L_Cy = lastP.y - currentP.y;
			for (int i = 0; i < 8; i++) {
				if (L_Cx == DIR[i][0] && L_Cy == DIR[i][1]) {
					ilast = ++i;
					break;
				}
				else continue;
			}//找到lastP对应的DIR数组的i，并指定i对应到下一个顺时针相邻元素

			for (int i = ilast; i < 16; i++) {
				aroundP.x = currentP.x + DIR1[i][0];
				aroundP.y = currentP.y + DIR1[i][1];

				if (srcImg.at<uchar>(aroundP.x, aroundP.y) == 255) {//扫描到一个255点
					if (aroundP.x == startP.x && aroundP.y == startP.y) {
						//该255点是起始点，说明循环了一圈
						if(roundNum > 10) return 2;     //否则说明时螺母
						else return 1;                  //说明还是圆
					}
					else if (aroundP.x == lastP.x && aroundP.y == lastP.y) {
						//说明该currentP已经到了一条线的端点，这是L件或者螺钉的特征
						return 0;  //返回0
					}
					else {//新的255点
						lastP.x = currentP.x;
						lastP.y = currentP.y;
						currentP.x = aroundP.x;
						currentP.y = aroundP.y;
						break;
					}//else是一个全新的255点
				}//if扫描到第一个255点
			}//for
		}//for
	}
	
}


cv::Mat robert(cv::Mat srcImg)
{
	int nRows = srcImg.rows;
	int nCols = srcImg.cols;
	int t1 = 0;
	int t2 = 0;

	cv::Mat dstImage = cv::Mat(nRows, nCols, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < nRows - 1; i++)
	{
		for (int j = 0; j < nCols - 1; j++)
		{
			t1 = abs(srcImg.at<uchar>(i, j) - srcImg.at<uchar>(i + 1, j + 1));
			t2 = abs(srcImg.at<uchar>(i + 1, j) - srcImg.at<uchar>(i, j + 1));
			dstImage.at<uchar>(i, j) = (t1 + t2);
		}
	}

	return dstImage;
}

bool screw_rec(cv::Mat srcImg)
{
	std::list<cv::Point> listP;     //存取对象点集队
	bool firstMeet = 0;             //第一次遇到100点标志
	int mark = 0;                   //上一或下移一行标志
	int topNum = 0;                 //顶层扫描一行点数
	int botNum = 0;                 //底层
	int minNum = 0;                 //top和not的最小值

	listP = seed_grow(srcImg);           //获取对象点队列
	srcImg = PCA_rotate(listP, srcImg);  //返回PCA旋转后的图像

	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) == 100) {
				if (firstMeet == 0) {
					firstMeet = 1;
					break;
				}
				else topNum++;
			}
		}
		if (firstMeet == 1) {
			mark++;
			if (mark == 2) goto for1;
		}
	}   //找螺钉从上往下的第二行的100点个数

for1:
	firstMeet = 0;
	mark = 0;
	for (int i = srcImg.rows - 1; i > 0; i--) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) == 100) {
				if (firstMeet == 0) {
					firstMeet = 1;
					break;
				}
				else botNum++;
			}
		}
		if (firstMeet == 1) {
			mark++;
			if (mark == 2) goto for2;
		}
	}  //找螺钉从下往上的第二行的个数
for2:
	minNum = std::min(botNum, topNum);

	if (minNum <= 5) return 1;     //说明是尖头螺钉
	else return 0;                 //说明是平头螺钉
}

cv::Mat PCA_rotate(std::list<cv::Point> listP,cv::Mat srcImg)
{
	cv::Mat data = cv::Mat(listP.size(), 2, CV_32FC1);      //将队列转化为Mat
	int n = 0;

	for (std::list<cv::Point>::iterator pw = listP.begin(); pw != listP.end(); pw++) {
		data.at<float>(n, 0) = (*pw).x;
		data.at<float>(n, 1) = (*pw).y;
		n++;
	}

	cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1);   //主成分PCA
	//	cv::Mat mean = pca.mean.clone();                   //平均值
	//	cv::Mat eigenvalues = pca.eigenvalues.clone();     //降序特征值
	cv::Mat eigenvectors = pca.eigenvectors.clone();       //对应特征向量

	float angle = -atan(eigenvectors.at<float>(0, 1) / eigenvectors.at<float>(0, 0)) * 180.0 / CV_PI; //主成分特征矢量角度

	//填充图像，因为选装后x轴会变长
	int maxBorder = (int)(std::max(srcImg.cols, srcImg.rows)* 1.414); //即为sqrt(2)*max
	int dy = (maxBorder - srcImg.rows) / 2;
	cv::copyMakeBorder(srcImg, srcImg, dy, dy, 0, 0, cv::BORDER_CONSTANT);

	//旋转
	cv::Point2f center((float)(srcImg.cols / 2), (float)(srcImg.rows / 2));
	cv::Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);  //求得旋转矩阵
	cv::warpAffine(srcImg, srcImg, affine_matrix, srcImg.size());

	return srcImg;
}

void quzhan(cv::Mat & src)
{
	using namespace cv;
	using namespace std;

	const cv::Point directions[8] = { { 0, 1 }, {1,1}, { 1, 0 }, { 1, -1 }, { 0, -1 },  { -1, -1 }, { -1, 0 },{ -1, 1 } };
	//	const int DIR[13][2] = { { 0, 1 }, {1,1}, { 1, 0 }, { 1, -1 }, { 0, -1 },  { -1, -1 }, { -1, 0 },{ -1, 1 },{ 0 ,0 },{-2,0},{2,0},{0,-2},{0,2} };
	const int DIR[17][2] = { { 0, 1 }, {1,1}, { 1, 0 }, { 1, -1 }, { 0, -1 },  { -1, -1 }, { -1, 0 },{ -1, 1 },{ 0 ,0 },{-2,0},{2,0},{0,-2},{0,2},{-2,-1},{-2,1},{2,-1},{2,1} };

//	RNG rng(time(0));

//	Mat src = imread("stick1.bmp");
	cv::resize(src, src, cv::Size(), 0.8, 0.8);
	//	cvtColor(src, gray, CV_BGR2GRAY);

	Mat Edge;
	//binary_img(gray);
	// Canny边缘检测
	Canny(src, Edge, 50, 100);
	Mat EdgeTemp = Edge.clone();


//	imshow("1", Edge);
	vector<Point> edge_t;
	vector<vector<Point>> edges;

	// 边缘跟踪
	int i, j, counts = 0, curr_d = 0;
	for (i = 1; i < Edge.rows - 1; i++)
		for (j = 1; j < Edge.cols - 1; j++)
		{
			// 起始点及当前点
			//Point s_pt = Point(i, j);
			Point b_pt = Point(i, j);
			Point c_pt = Point(i, j);

			// 如果当前点为前景点
			if (255 == Edge.at<uchar>(c_pt.x, c_pt.y))
			{
				edge_t.clear();
				bool tra_flag = false;
				// 存入
				edge_t.push_back(c_pt);
				Edge.at<uchar>(c_pt.x, c_pt.y) = 0;    // 用过的点直接给设置为0

				// 进行跟踪
				while (!tra_flag)
				{
					// 循环八次
					for (counts = 0; counts < 8; counts++)
					{
						// 防止索引出界
						if (curr_d >= 8)
						{
							curr_d -= 8;
						}
						if (curr_d < 0)
						{
							curr_d += 8;
						}

						// 当前点坐标
						// 跟踪的过程，应该是个连续的过程，需要不停的更新搜索的root点
						c_pt = Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);

						// 边界判断
						if ((c_pt.x > 0) && (c_pt.x < Edge.cols - 1) &&
							(c_pt.y > 0) && (c_pt.y < Edge.rows - 1))
						{
							// 如果存在边缘
							if (255 == Edge.at<uchar>(c_pt.x, c_pt.y))
							{
								curr_d -= 2;   // 更新当前方向
								edge_t.push_back(c_pt);
								Edge.at<uchar>(c_pt.x, c_pt.y) = 0;

								// 更新b_pt:跟踪的root点
								b_pt.x = c_pt.x;
								b_pt.y = c_pt.y;

								//cout << c_pt.x << " " << c_pt.y << endl;

								break;   // 跳出for循环
							}
						}
						curr_d++;
					}   // end for
					// 跟踪的终止条件：如果8邻域都不存在边缘
					if (8 == counts)
					{
						// 清零
						curr_d = 0;
						tra_flag = true;
						edges.push_back(edge_t);

						break;
					}

				}  // end if
			}  // end while

		}

	// 显示一下
	//Mat trace_edge = Mat::zeros(Edge.rows, Edge.cols, CV_8UC1);
	//Mat trace_edge_color;
	//cvtColor(trace_edge, trace_edge_color, CV_GRAY2BGR);
	//for (i = 0; i < edges.size(); i++)
	//{
	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

	//	//cout << edges[i].size() << endl;
	//	// 过滤掉较小的边缘
	//	if (edges[i].size() > 5)
	//	{
	//		for (j = 0; j < edges[i].size(); j++)
	//		{
	//			trace_edge_color.at<Vec3b>(edges[i][j].x, edges[i][j].y)[0] = color[0];
	//			trace_edge_color.at<Vec3b>(edges[i][j].x, edges[i][j].y)[1] = color[1];
	//			trace_edge_color.at<Vec3b>(edges[i][j].x, edges[i][j].y)[2] = color[2];
	//		}
	//	}

	//}

	cv::Point aroundP;
	std::vector<cv::Point> delP;
	int mark = 0;
	for (int i = 0; i < edges.size(); i++) {
		for (int j = 0; j < edges[i].size(); j++) {
			mark = 0;
			for (int n = 0; n < 17; n++) {
				aroundP = { edges[i][j].x + DIR[n][0],edges[i][j].y + DIR[n][1] };
				if ((aroundP.x > 0) && (aroundP.x < EdgeTemp.cols - 1) &&
					(aroundP.y > 0) && (aroundP.y < EdgeTemp.rows - 1)) {



					if (EdgeTemp.at<uchar>(aroundP.x, aroundP.y) == 255) mark++;

					if (mark >= 8) {
						cv::circle(EdgeTemp, cv::Point(edges[i][j].y, edges[i][j].x), 5, cv::Scalar(100));
						delP.push_back(cv::Point(edges[i][j].y, edges[i][j].x));
					}
				}
			}
		}
	}
//	imshow("1", EdgeTemp);
//	waitKey();
	float dis = 0;
	for (int i = 0; i < delP.size(); i++) {
		for (int j = 0; j < delP.size(); j++) {
			dis = abs(delP[i].x - delP[j].x) + abs(delP[i].y - delP[j].y);
			if (dis <= 30 && dis > 1) line(src, delP[i], delP[j], cv::Scalar(255), 2);
		}
	}

}

void hough_test_new(cv::Mat &src)
{
	//cv::Mat src = cv::imread("stick1.bmp", 0);
	std::vector<cv::Vec4i> lines;
	const int HOUGHJIAO = 85;
	//	const int DIS = 100;
	int dis = 0;

	cv::resize(src, src, cv::Size(), 0.8, 0.8);


	binary_img(src);    //二值化
	src = robert(src);
	cv::HoughLinesP(src, lines, 1, CV_PI / 180, HOUGHJIAO, 90, 35);

	std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();
	while (it2 != lines.end())
	{
		cv::Point pt1((*it2)[0], (*it2)[1]);   //第一个端点
		cv::Point pt2((*it2)[2], (*it2)[3]);   //第二个端点
		dis = sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
		//		if (dis > DIS) {
		cv::line(src, pt1, pt2, cv::Scalar(100), 2);
		//		}
		++it2;
	}

	//cv::imshow("test", src);
	//cv::waitKey();
}

void hough_test(cv::Mat & src)
{
	//cv::Mat src = cv::imread("brStick.bmp", 0);
	cv::resize(src, src, cv::Size(), 0.8, 0.8);

	std::vector<cv::Vec4i> lines;
	std::vector<cv::Vec4i> drawLines;
	const int HOUGHJIAO = 70;
	const double k = 0.1; //斜率阈值参数
	const double b = 10.0; //截距阈值参数
	const double kv = 20.0; //垂直直线斜率参数
	const int lineNum = 20;

	double k1 = 0, k2 = 0;  //直线斜率
	double b1 = 0, b2 = 0;  //直线截距

	binary_img(src);    //二值化
//	Threshold(src, TwinPeakThreshold(src), true); //双峰二值化
	src = robert(src);  //robert边缘化
	cv::HoughLinesP(src, lines, 2, CV_PI / 180, HOUGHJIAO, 10.0, 15.0);

	std::vector<cv::Vec4i>::const_iterator it1 = lines.begin();
	std::vector<cv::Vec4i>::const_iterator it2;
	bool first = 1;
	//while (it2 != lines.end())
	//{
	//	cv::Point pt1((*it2)[0], (*it2)[1]);   //第一个端点
	//	cv::Point pt2((*it2)[2], (*it2)[3]);   //第二个端点
	//	cv::line(src, pt1, pt2, cv::Scalar(100), 2);
	//	k = double(pt2.y - pt1.y) / double(pt2.x - pt1.x);
	//	b = double(pt1.y) - k * double(pt1.x);
	//	std::cout << k << " " << b << std::endl;
	//	std::cout << pt1.x << "," << pt1.y << " " << pt2.x << "," << pt2.y << std::endl<<std::endl;
	//	imshow("result", src);
	//	cv::waitKey();
	//	++it2;
	//}

	for (it1; it1 != lines.end() - 1;) {
		if (lines.size() > lineNum || lines.size() <= 2) break;

		first = 1;
		cv::Point pt1((*it1)[0], (*it1)[1]);   //第一个端点
		cv::Point pt2((*it1)[2], (*it1)[3]);   //第二个端点
		k1 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //计算斜率，截距
		b1 = double(pt1.y) - k1 * double(pt1.x);

		if (fabs(k1) >= kv) k1 = kv;           //如果斜率绝对值大于10，那么设为10
		cv::Vec4i temp = { (*it1)[0] ,(*it1)[1],(*it1)[2], (*it1)[3] };
		drawLines.push_back(temp);             //先保存这个线段


		for (it2 = it1 + 1; it2 != lines.end();) {

			pt1.x = (*it2)[0], pt1.y = (*it2)[1];
			pt2.x = (*it2)[2], pt2.y = (*it2)[3];
			k2 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //计算斜率，截距
			b2 = double(pt1.y) - k2 * double(pt1.x);
			if (fabs(k2) >= kv) k2 = kv;

			if (fabs(k1 - k2) > k || fabs(b1 - b2) > b) {
				if (first == 1) it1 = it2, first = 0; //遇到了不同类的直线，first指向
				it2++;    //不同类直线，指向下一个
			}
			else { //是相同类的直线
				int minX = 2000, maxX = 0;
				int iy, ay;
				if (temp[0] < minX) minX = temp[0], iy = temp[1];
				if (temp[2] < minX) minX = temp[2], iy = temp[3];
				if (pt1.x < minX) minX = pt1.x, iy = pt1.y;
				if (pt2.x < minX) minX = pt2.x, iy = pt2.y;    //找最小的x

				if (temp[0] > maxX) maxX = temp[0], ay = temp[1];
				if (temp[2] > maxX) maxX = temp[2], ay = temp[3];
				if (pt1.x > maxX) maxX = pt1.x, ay = pt1.y;
				if (pt2.x > maxX) maxX = pt2.x, ay = pt2.y;    //找最大的x

				(drawLines.back())[0] = minX, (drawLines.back())[1] = iy;
				(drawLines.back())[2] = maxX, (drawLines.back())[3] = ay; //更改要画的线段的端点

				it2 = lines.erase(it2);   //删除it2指向的这个相似的直线，it2会自动指向下一个直线
			}
		}

	}

	it2 = drawLines.begin();
	while (it2 != drawLines.end())
	{
		if (drawLines.size() == 0) break;

		cv::Point pt1((*it2)[0], (*it2)[1]);   //第一个端点
		cv::Point pt2((*it2)[2], (*it2)[3]);   //第二个端点
		cv::line(src, pt1, pt2, cv::Scalar(100), 2);
		++it2;
	}
	//	imshow("result", src);
	cv::waitKey(30);
}

void yu_chu_li(cv::Mat & src)
{
	std::vector<cv::Vec4i> lines;
	std::vector<cv::Vec4i> drawLines;
	const int HOUGHJIAO = 80;
	const double k = 0.02; //斜率阈值参数
	const double b = 10.0; //截距阈值参数
	const double kv = 15.0; //垂直直线斜率参数
	const int lineNum = 20;
	const int distan = 200;

	double k1 = 0, k2 = 0;  //直线斜率
	double b1 = 0, b2 = 0;  //直线截距

//	binary_img(src);    //二值化
//	Threshold(src, TwinPeakThreshold(src), true); //双峰二值化
//	cv::Mat robert_image = robert(src);  //robert边缘化
	cv::Mat robert_image;
	cv::Canny(src,robert_image, 100, 200);
	cv::HoughLinesP(robert_image, lines, 1, CV_PI / 180, HOUGHJIAO, 10.0, 15.0);

	std::vector<cv::Vec4i>::const_iterator it1 = lines.begin();
	std::vector<cv::Vec4i>::const_iterator it2;
	bool first = 1;

	for (it1; it1 != lines.end() - 1;) {
		if (lines.size() > lineNum || lines.size() <= 2) break;

		first = 1;
		cv::Point pt1((*it1)[0], (*it1)[1]);   //第一个端点
		cv::Point pt2((*it1)[2], (*it1)[3]);   //第二个端点
		k1 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //计算斜率，截距
		b1 = double(pt1.y) - k1 * double(pt1.x);

		if (fabs(k1) >= kv) k1 = kv;           //如果斜率绝对值大于10，那么设为10
		cv::Vec4i temp = { (*it1)[0] ,(*it1)[1],(*it1)[2], (*it1)[3] };
		drawLines.push_back(temp);             //先保存这个线段


		for (it2 = it1 + 1; it2 != lines.end();) {

			pt1.x = (*it2)[0], pt1.y = (*it2)[1];
			pt2.x = (*it2)[2], pt2.y = (*it2)[3];
			k2 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //计算斜率，截距
			b2 = double(pt1.y) - k2 * double(pt1.x);
			if (fabs(k2) >= kv) k2 = kv;

			if (fabs(k1 - k2) > k || fabs(b1 - b2) > b) {
				if (first == 1) it1 = it2, first = 0; //遇到了不同类的直线，first指向
				it2++;    //不同类直线，指向下一个
			}
			else { //是相同类的直线
				int minX = 2000, maxX = 0;
				int iy, ay;
				if (temp[0] < minX) minX = temp[0], iy = temp[1];
				if (temp[2] < minX) minX = temp[2], iy = temp[3];
				if (pt1.x < minX) minX = pt1.x, iy = pt1.y;
				if (pt2.x < minX) minX = pt2.x, iy = pt2.y;    //找最小的x

				if (temp[0] > maxX) maxX = temp[0], ay = temp[1];
				if (temp[2] > maxX) maxX = temp[2], ay = temp[3];
				if (pt1.x > maxX) maxX = pt1.x, ay = pt1.y;
				if (pt2.x > maxX) maxX = pt2.x, ay = pt2.y;    //找最大的x

				(drawLines.back())[0] = minX, (drawLines.back())[1] = iy;
				(drawLines.back())[2] = maxX, (drawLines.back())[3] = ay; //更改要画的线段的端点

				it2 = lines.erase(it2);   //删除it2指向的这个相似的直线，it2会自动指向下一个直线
			}
		}

	}

	it2 = drawLines.begin();
	while (it2 != drawLines.end())
	{
		if (drawLines.size() == 0) break;

		
		cv::Point pt1((*it2)[0], (*it2)[1]);   //第一个端点
		cv::Point pt2((*it2)[2], (*it2)[3]);   //第二个端点
		int dis = (pt2.y - pt1.y)*(pt2.y - pt1.y) + (pt2.x - pt1.x)*(pt2.x - pt1.x);
//		std::cout << dis << std::endl;
		if (dis > distan) {
			cv::line(src, pt1, pt2, cv::Scalar(0), 2);
		}
		++it2;
	}

//	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
//	erode(src, src, element);//腐蚀
}
