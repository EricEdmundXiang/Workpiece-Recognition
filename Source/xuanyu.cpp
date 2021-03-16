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
	const int COLOR = 100;//�����������ɫ

	std::queue<cv::Point2i> q1;
	std::list<cv::Point> vecP; //���������ĵ㼯
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
					break;//��ɨ�赽��һ��255������ʱ����ɫ��ΪCOLOR�����У��Ƴ��ڲ�forѭ��
				}
			}
			if (!q1.empty()) break;//��ɨ�赽��һ��255����ʱ������Ԫ��Ϊ1���˳����forѭ��
		}
	}
	

	while (!q1.empty()) {
		ptCenter = q1.front();
		q1.pop();//ɾ����һ��Ԫ��

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

	int color = 0;//��ǰ���ػҶȼ�
	int maxColor = 0;//��Ŀ���ĻҶȼ�
	int secColor = 0;//��Ŀ�ڶ���ĻҶȼ�
	int maxNum = 0;//��Ŀ���ĻҶȼ�����Ŀ
	int secNum = 0;
	int peakNum = 0;//�ֲ�����ֵ��

	std::vector<cv::Point> ptPeak;
	cv::Point ptTemp = { 0,0 };

	int grayLevel[256] = { 0 };//ͳ�Ƹ��Ҷȼ������ظ���

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			color = srcImg.at<uchar>(i, j);
			grayLevel[color] ++;
		}
	}//ͳ�Ƹ��Ҷȼ���Ŀ

	//for (int i = 0; i < 256; i++) {
	//	std::cout << grayLevel[i] << std::endl;
	//}

	for (int i = 0; i < 254; i++) {
		if (grayLevel[i] >= peakNum) {
			peakNum = grayLevel[i];
			if (grayLevel[i + 1] < peakNum && grayLevel[i + 2] < peakNum) {
				ptTemp.x = peakNum;   //x��¼��Ŀ
				ptTemp.y = i;         //y��¼����Ŀ��Ӧ�ĻҶȼ�
				ptPeak.push_back(ptTemp);
			}
		}
	}//ͳ�Ƹ�����ֵ�Ҷȼ��Լ����Ӧ����Ŀ

	//ȡ����ֵvector���������ĺ͵ڶ���ĻҶȼ�
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

	std::list<parted> lisParted;            //���ָ�����Ĺ����б�
	parted partedOne;                       //���ָ������ĳһ������
	cv::Point ptCenter = { 0,0 };           //�������������ӵ�

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
				if(errorMark == 1) return lisParted;          //����
				else if (partedOne.pic.rows == 1) continue;   //��⵽��Ϊ����
				else lisParted.push_back(partedOne);          //�ṹ�����
			}
		}
	}

	return lisParted;
}

parted grow_part(cv::Mat & srcImg, cv::Point ptGrow,bool & errorMark)
{
	parted partedOne = { cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)),cv::Mat(1, 1, CV_8UC1, cv::Scalar(0)),{0,0},0,0,"",0,0,false}; //��ʼ��

	std::queue<cv::Point> q1;                   //������������
	cv::Point ptGrowing = { 0,0 };              //���������
	cv::Point ptCenter = { 0,0 };               //�������ĵ�
	const int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };

	int maxX = ptGrow.x, maxY = ptGrow.y;
	int minX = ptGrow.x, minY = ptGrow.y;       //������¼������������Ͻ����½ǣ��Դ���Ӧsize��Mat
	std::list<cv::Point> ltPixel;               //��¼����������Ĺ��������ص㼯��

	ltPixel.push_back(ptGrow);
	q1.push(ptGrow);

	while (!q1.empty()) {
		ptCenter = q1.front();          //ȡ���ĵ�
		q1.pop();                       //ɾ����һ��Ԫ��

		for (int i = 0; i < 8; i++) {
			ptGrowing.x = ptCenter.x + DIR[i][0];
			ptGrowing.y = ptCenter.y + DIR[i][1];

			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.y >(srcImg.cols - 1) || (ptGrowing.x > srcImg.rows - 1))
			{
//				errorMark = 1;     //��⵽�С������в�����ͼ��߽�
//				return partedOne;  //�˳�����
				partedOne.bian = true;
				continue;
			}

			if (srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) == 255) {
				srcImg.at<uchar>(ptGrowing.x, ptGrowing.y) = 0;
				q1.push(ptGrowing);
				if (ptGrowing.x > maxX) maxX = ptGrowing.x;
				if (ptGrowing.y > maxY) maxY = ptGrowing.y;
				if (ptGrowing.x < minX) minX = ptGrowing.x;
				if (ptGrowing.y < minY) minY = ptGrowing.y;        //�����ϽǺ����½�
				ltPixel.push_back(ptGrowing);                      //���ڸù�����������ؽ�������
			}
		}
	}

	if (ltPixel.size() < 100) return partedOne;            //�����������С��30������
	else {
		int partedH = maxX - minX + 10;                   //��ͼ��ĸ�
		int partedW = maxY - minY + 10;                   //��ͼ��Ŀ�
		cv::Mat partedPic = cv::Mat(partedH, partedW, CV_8UC1, cv::Scalar(0));

		for (std::list<cv::Point>::iterator pw = ltPixel.begin(); pw != ltPixel.end(); pw++) {
			partedPic.at<uchar>((*pw).x - minX + 5, (*pw).y - minY + 5) = 255;    //����ͼ�л�����
		}
		partedOne.pic = partedPic;
		partedOne.height = partedH;
		partedOne.width = partedW;
		partedOne.zuoshang = { minX,minY };
		partedOne.pixNum = ltPixel.size();
		
		return partedOne;        //���طָ������һ�������ṹ
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
				cv::rectangle(daXiangSuGe, cv::Rect(4 * j, 4 * i, 4, 4), cv::Scalar(255, 0, 0), 1, 1, 0);//�������ص�
			}
		}
	}
	return daXiangSuGe;
}

int recognition1(cv::Mat srcImg)
{
	cv::Point startP = { 0,0 };     //��¼ɨ�赽�ĵ�һ����
	cv::Point lastP = { 0,0 };      //��¼���������������currentP����һ����
	cv::Point currentP = { 0,0 };   //���������е�ǰ��
	cv::Point aroundP = { 0,0 };    //��¼˳ʱ��ɨ�������

	const int DIR[8][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };//����ɨ��˳��˳ʱ��
	const int DIR1[16][2] = { {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} ,{-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	//������֤תȦɨ��������ԣ�

	int L_Cx = 0;     //lastP - currentP��x֮��
	int L_Cy = 0;     //lastP - currentP��y֮����㻷��ɨ����lastP��˳ʱ����һ��DIRԪ��
	int ilast = 0;    //��ӦlastP��DIR��i��ֵ
	int roundNum = 0; //��¼ɨ������е����Ĵ���
	bool yuan = 1;    //Բ�жϱ�־

	for (int i = 0; i < srcImg.rows; i++) {
		for (int j = 0; j < srcImg.cols; j++) {
			if (srcImg.at<uchar>(i, j) == 255) {
				startP.x = i;
				startP.y = j;
				lastP.x = i;
				lastP.y = j;
				currentP.x = i;
				currentP.y = j;//��ʼ��������Ϊɨ�赽�ĵ�һ����
				break;
			}
		}
		if (startP.x != 0) break;//˵���Ѿ��ҵ��˵�һ����
	}

	for (int i = 0; i < 8; i++) {
		aroundP.x = currentP.x + DIR[i][0];
		aroundP.y = currentP.y + DIR[i][1];//ȡ�������Ͻ�Ϊɨ�����˳ʱ��ɨ��
		if (aroundP.x > 0 && aroundP.y > 0 && aroundP.x < srcImg.rows  && aroundP.y < srcImg.cols)
		{

			if (srcImg.at<uchar>(aroundP.x, aroundP.y) == 255) {
				yuan = 0;
				if (i == 0) continue;//��ȡ���Ͻǵĵ�Ϊ���
				currentP.x = aroundP.x;
				currentP.y = aroundP.y;
				break;//������ɨ�赽��һ��255��ͽ���ѭ��
			}
		}
	}
	if (yuan) {
		//ɨ��һȦû�м�⵽����255�㣬˵�����������Բ�����ǻ����������Ҳ��Բ
		return 1;
	}
	else {
		while(1){//����ѭ����ֱ���жϳ�һ�����
			roundNum++;  //����������һ
			L_Cx = lastP.x - currentP.x;
			L_Cy = lastP.y - currentP.y;
			for (int i = 0; i < 8; i++) {
				if (L_Cx == DIR[i][0] && L_Cy == DIR[i][1]) {
					ilast = ++i;
					break;
				}
				else continue;
			}//�ҵ�lastP��Ӧ��DIR�����i����ָ��i��Ӧ����һ��˳ʱ������Ԫ��

			for (int i = ilast; i < 16; i++) {
				aroundP.x = currentP.x + DIR1[i][0];
				aroundP.y = currentP.y + DIR1[i][1];

				if (srcImg.at<uchar>(aroundP.x, aroundP.y) == 255) {//ɨ�赽һ��255��
					if (aroundP.x == startP.x && aroundP.y == startP.y) {
						//��255������ʼ�㣬˵��ѭ����һȦ
						if(roundNum > 10) return 2;     //����˵��ʱ��ĸ
						else return 1;                  //˵������Բ
					}
					else if (aroundP.x == lastP.x && aroundP.y == lastP.y) {
						//˵����currentP�Ѿ�����һ���ߵĶ˵㣬����L�������ݶ�������
						return 0;  //����0
					}
					else {//�µ�255��
						lastP.x = currentP.x;
						lastP.y = currentP.y;
						currentP.x = aroundP.x;
						currentP.y = aroundP.y;
						break;
					}//else��һ��ȫ�µ�255��
				}//ifɨ�赽��һ��255��
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
	std::list<cv::Point> listP;     //��ȡ����㼯��
	bool firstMeet = 0;             //��һ������100���־
	int mark = 0;                   //��һ������һ�б�־
	int topNum = 0;                 //����ɨ��һ�е���
	int botNum = 0;                 //�ײ�
	int minNum = 0;                 //top��not����Сֵ

	listP = seed_grow(srcImg);           //��ȡ��������
	srcImg = PCA_rotate(listP, srcImg);  //����PCA��ת���ͼ��

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
	}   //���ݶ��������µĵڶ��е�100�����

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
	}  //���ݶ��������ϵĵڶ��еĸ���
for2:
	minNum = std::min(botNum, topNum);

	if (minNum <= 5) return 1;     //˵���Ǽ�ͷ�ݶ�
	else return 0;                 //˵����ƽͷ�ݶ�
}

cv::Mat PCA_rotate(std::list<cv::Point> listP,cv::Mat srcImg)
{
	cv::Mat data = cv::Mat(listP.size(), 2, CV_32FC1);      //������ת��ΪMat
	int n = 0;

	for (std::list<cv::Point>::iterator pw = listP.begin(); pw != listP.end(); pw++) {
		data.at<float>(n, 0) = (*pw).x;
		data.at<float>(n, 1) = (*pw).y;
		n++;
	}

	cv::PCA pca(data, cv::Mat(), CV_PCA_DATA_AS_ROW, 1);   //���ɷ�PCA
	//	cv::Mat mean = pca.mean.clone();                   //ƽ��ֵ
	//	cv::Mat eigenvalues = pca.eigenvalues.clone();     //��������ֵ
	cv::Mat eigenvectors = pca.eigenvectors.clone();       //��Ӧ��������

	float angle = -atan(eigenvectors.at<float>(0, 1) / eigenvectors.at<float>(0, 0)) * 180.0 / CV_PI; //���ɷ�����ʸ���Ƕ�

	//���ͼ����Ϊѡװ��x���䳤
	int maxBorder = (int)(std::max(srcImg.cols, srcImg.rows)* 1.414); //��Ϊsqrt(2)*max
	int dy = (maxBorder - srcImg.rows) / 2;
	cv::copyMakeBorder(srcImg, srcImg, dy, dy, 0, 0, cv::BORDER_CONSTANT);

	//��ת
	cv::Point2f center((float)(srcImg.cols / 2), (float)(srcImg.rows / 2));
	cv::Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);  //�����ת����
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
	// Canny��Ե���
	Canny(src, Edge, 50, 100);
	Mat EdgeTemp = Edge.clone();


//	imshow("1", Edge);
	vector<Point> edge_t;
	vector<vector<Point>> edges;

	// ��Ե����
	int i, j, counts = 0, curr_d = 0;
	for (i = 1; i < Edge.rows - 1; i++)
		for (j = 1; j < Edge.cols - 1; j++)
		{
			// ��ʼ�㼰��ǰ��
			//Point s_pt = Point(i, j);
			Point b_pt = Point(i, j);
			Point c_pt = Point(i, j);

			// �����ǰ��Ϊǰ����
			if (255 == Edge.at<uchar>(c_pt.x, c_pt.y))
			{
				edge_t.clear();
				bool tra_flag = false;
				// ����
				edge_t.push_back(c_pt);
				Edge.at<uchar>(c_pt.x, c_pt.y) = 0;    // �ù��ĵ�ֱ�Ӹ�����Ϊ0

				// ���и���
				while (!tra_flag)
				{
					// ѭ���˴�
					for (counts = 0; counts < 8; counts++)
					{
						// ��ֹ��������
						if (curr_d >= 8)
						{
							curr_d -= 8;
						}
						if (curr_d < 0)
						{
							curr_d += 8;
						}

						// ��ǰ������
						// ���ٵĹ��̣�Ӧ���Ǹ������Ĺ��̣���Ҫ��ͣ�ĸ���������root��
						c_pt = Point(b_pt.x + directions[curr_d].x, b_pt.y + directions[curr_d].y);

						// �߽��ж�
						if ((c_pt.x > 0) && (c_pt.x < Edge.cols - 1) &&
							(c_pt.y > 0) && (c_pt.y < Edge.rows - 1))
						{
							// ������ڱ�Ե
							if (255 == Edge.at<uchar>(c_pt.x, c_pt.y))
							{
								curr_d -= 2;   // ���µ�ǰ����
								edge_t.push_back(c_pt);
								Edge.at<uchar>(c_pt.x, c_pt.y) = 0;

								// ����b_pt:���ٵ�root��
								b_pt.x = c_pt.x;
								b_pt.y = c_pt.y;

								//cout << c_pt.x << " " << c_pt.y << endl;

								break;   // ����forѭ��
							}
						}
						curr_d++;
					}   // end for
					// ���ٵ���ֹ���������8���򶼲����ڱ�Ե
					if (8 == counts)
					{
						// ����
						curr_d = 0;
						tra_flag = true;
						edges.push_back(edge_t);

						break;
					}

				}  // end if
			}  // end while

		}

	// ��ʾһ��
	//Mat trace_edge = Mat::zeros(Edge.rows, Edge.cols, CV_8UC1);
	//Mat trace_edge_color;
	//cvtColor(trace_edge, trace_edge_color, CV_GRAY2BGR);
	//for (i = 0; i < edges.size(); i++)
	//{
	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

	//	//cout << edges[i].size() << endl;
	//	// ���˵���С�ı�Ե
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


	binary_img(src);    //��ֵ��
	src = robert(src);
	cv::HoughLinesP(src, lines, 1, CV_PI / 180, HOUGHJIAO, 90, 35);

	std::vector<cv::Vec4i>::const_iterator it2 = lines.begin();
	while (it2 != lines.end())
	{
		cv::Point pt1((*it2)[0], (*it2)[1]);   //��һ���˵�
		cv::Point pt2((*it2)[2], (*it2)[3]);   //�ڶ����˵�
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
	const double k = 0.1; //б����ֵ����
	const double b = 10.0; //�ؾ���ֵ����
	const double kv = 20.0; //��ֱֱ��б�ʲ���
	const int lineNum = 20;

	double k1 = 0, k2 = 0;  //ֱ��б��
	double b1 = 0, b2 = 0;  //ֱ�߽ؾ�

	binary_img(src);    //��ֵ��
//	Threshold(src, TwinPeakThreshold(src), true); //˫���ֵ��
	src = robert(src);  //robert��Ե��
	cv::HoughLinesP(src, lines, 2, CV_PI / 180, HOUGHJIAO, 10.0, 15.0);

	std::vector<cv::Vec4i>::const_iterator it1 = lines.begin();
	std::vector<cv::Vec4i>::const_iterator it2;
	bool first = 1;
	//while (it2 != lines.end())
	//{
	//	cv::Point pt1((*it2)[0], (*it2)[1]);   //��һ���˵�
	//	cv::Point pt2((*it2)[2], (*it2)[3]);   //�ڶ����˵�
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
		cv::Point pt1((*it1)[0], (*it1)[1]);   //��һ���˵�
		cv::Point pt2((*it1)[2], (*it1)[3]);   //�ڶ����˵�
		k1 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //����б�ʣ��ؾ�
		b1 = double(pt1.y) - k1 * double(pt1.x);

		if (fabs(k1) >= kv) k1 = kv;           //���б�ʾ���ֵ����10����ô��Ϊ10
		cv::Vec4i temp = { (*it1)[0] ,(*it1)[1],(*it1)[2], (*it1)[3] };
		drawLines.push_back(temp);             //�ȱ�������߶�


		for (it2 = it1 + 1; it2 != lines.end();) {

			pt1.x = (*it2)[0], pt1.y = (*it2)[1];
			pt2.x = (*it2)[2], pt2.y = (*it2)[3];
			k2 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //����б�ʣ��ؾ�
			b2 = double(pt1.y) - k2 * double(pt1.x);
			if (fabs(k2) >= kv) k2 = kv;

			if (fabs(k1 - k2) > k || fabs(b1 - b2) > b) {
				if (first == 1) it1 = it2, first = 0; //�����˲�ͬ���ֱ�ߣ�firstָ��
				it2++;    //��ͬ��ֱ�ߣ�ָ����һ��
			}
			else { //����ͬ���ֱ��
				int minX = 2000, maxX = 0;
				int iy, ay;
				if (temp[0] < minX) minX = temp[0], iy = temp[1];
				if (temp[2] < minX) minX = temp[2], iy = temp[3];
				if (pt1.x < minX) minX = pt1.x, iy = pt1.y;
				if (pt2.x < minX) minX = pt2.x, iy = pt2.y;    //����С��x

				if (temp[0] > maxX) maxX = temp[0], ay = temp[1];
				if (temp[2] > maxX) maxX = temp[2], ay = temp[3];
				if (pt1.x > maxX) maxX = pt1.x, ay = pt1.y;
				if (pt2.x > maxX) maxX = pt2.x, ay = pt2.y;    //������x

				(drawLines.back())[0] = minX, (drawLines.back())[1] = iy;
				(drawLines.back())[2] = maxX, (drawLines.back())[3] = ay; //����Ҫ�����߶εĶ˵�

				it2 = lines.erase(it2);   //ɾ��it2ָ���������Ƶ�ֱ�ߣ�it2���Զ�ָ����һ��ֱ��
			}
		}

	}

	it2 = drawLines.begin();
	while (it2 != drawLines.end())
	{
		if (drawLines.size() == 0) break;

		cv::Point pt1((*it2)[0], (*it2)[1]);   //��һ���˵�
		cv::Point pt2((*it2)[2], (*it2)[3]);   //�ڶ����˵�
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
	const double k = 0.02; //б����ֵ����
	const double b = 10.0; //�ؾ���ֵ����
	const double kv = 15.0; //��ֱֱ��б�ʲ���
	const int lineNum = 20;
	const int distan = 200;

	double k1 = 0, k2 = 0;  //ֱ��б��
	double b1 = 0, b2 = 0;  //ֱ�߽ؾ�

//	binary_img(src);    //��ֵ��
//	Threshold(src, TwinPeakThreshold(src), true); //˫���ֵ��
//	cv::Mat robert_image = robert(src);  //robert��Ե��
	cv::Mat robert_image;
	cv::Canny(src,robert_image, 100, 200);
	cv::HoughLinesP(robert_image, lines, 1, CV_PI / 180, HOUGHJIAO, 10.0, 15.0);

	std::vector<cv::Vec4i>::const_iterator it1 = lines.begin();
	std::vector<cv::Vec4i>::const_iterator it2;
	bool first = 1;

	for (it1; it1 != lines.end() - 1;) {
		if (lines.size() > lineNum || lines.size() <= 2) break;

		first = 1;
		cv::Point pt1((*it1)[0], (*it1)[1]);   //��һ���˵�
		cv::Point pt2((*it1)[2], (*it1)[3]);   //�ڶ����˵�
		k1 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //����б�ʣ��ؾ�
		b1 = double(pt1.y) - k1 * double(pt1.x);

		if (fabs(k1) >= kv) k1 = kv;           //���б�ʾ���ֵ����10����ô��Ϊ10
		cv::Vec4i temp = { (*it1)[0] ,(*it1)[1],(*it1)[2], (*it1)[3] };
		drawLines.push_back(temp);             //�ȱ�������߶�


		for (it2 = it1 + 1; it2 != lines.end();) {

			pt1.x = (*it2)[0], pt1.y = (*it2)[1];
			pt2.x = (*it2)[2], pt2.y = (*it2)[3];
			k2 = double(pt2.y - pt1.y) / double(pt2.x - pt1.x); //����б�ʣ��ؾ�
			b2 = double(pt1.y) - k2 * double(pt1.x);
			if (fabs(k2) >= kv) k2 = kv;

			if (fabs(k1 - k2) > k || fabs(b1 - b2) > b) {
				if (first == 1) it1 = it2, first = 0; //�����˲�ͬ���ֱ�ߣ�firstָ��
				it2++;    //��ͬ��ֱ�ߣ�ָ����һ��
			}
			else { //����ͬ���ֱ��
				int minX = 2000, maxX = 0;
				int iy, ay;
				if (temp[0] < minX) minX = temp[0], iy = temp[1];
				if (temp[2] < minX) minX = temp[2], iy = temp[3];
				if (pt1.x < minX) minX = pt1.x, iy = pt1.y;
				if (pt2.x < minX) minX = pt2.x, iy = pt2.y;    //����С��x

				if (temp[0] > maxX) maxX = temp[0], ay = temp[1];
				if (temp[2] > maxX) maxX = temp[2], ay = temp[3];
				if (pt1.x > maxX) maxX = pt1.x, ay = pt1.y;
				if (pt2.x > maxX) maxX = pt2.x, ay = pt2.y;    //������x

				(drawLines.back())[0] = minX, (drawLines.back())[1] = iy;
				(drawLines.back())[2] = maxX, (drawLines.back())[3] = ay; //����Ҫ�����߶εĶ˵�

				it2 = lines.erase(it2);   //ɾ��it2ָ���������Ƶ�ֱ�ߣ�it2���Զ�ָ����һ��ֱ��
			}
		}

	}

	it2 = drawLines.begin();
	while (it2 != drawLines.end())
	{
		if (drawLines.size() == 0) break;

		
		cv::Point pt1((*it2)[0], (*it2)[1]);   //��һ���˵�
		cv::Point pt2((*it2)[2], (*it2)[3]);   //�ڶ����˵�
		int dis = (pt2.y - pt1.y)*(pt2.y - pt1.y) + (pt2.x - pt1.x)*(pt2.x - pt1.x);
//		std::cout << dis << std::endl;
		if (dis > distan) {
			cv::line(src, pt1, pt2, cv::Scalar(0), 2);
		}
		++it2;
	}

//	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
//	erode(src, src, element);//��ʴ
}
