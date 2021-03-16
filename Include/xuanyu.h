#pragma once
//�洢���д����صĺ���

struct parted {
	cv::Mat pic;           //�ָ���ͼ��
	cv::Mat thinPic;       //ϸ�����ͼ��
	cv::Point zuoshang;    //��Ӧԭͼ�����Ͻ�����
	int height;            //�ָ����ͼ���
	int width;             //�ָ����ͼ���
	std::string name; 
	int mark;
	int pixNum;
	bool bian;
};


void thinImage(cv::Mat & srcImg);
//���ܣ��Ǽ���ȡ
//���룺Mat������
//���أ���

std::list<cv::Point> seed_grow(cv::Mat &srcImg, bool fromCenter = false);
//���ܣ�����������ֻ��һ��test������������srcImg�������Ͻǵİ�ɫ����������ΪCOLORɫ�����ڷָ�,�������������ĵ㼯
//���룺Mat
//�����Mat

int binary_num(const cv::Mat srcImg);
//����˫����ֵ��ֵ����Ӧ����ֵ

void binary_img(cv::Mat & srcImg);
//����˫����ֵ���ж�ֵ��,ʹ���˺���binary_num��ȡ��ֵ

//std::list<cv::Mat> separation(cv::Mat srcImg);
//�ָ�ͼƬ�еĹ��������浽vector������

std::list<parted> separation(cv::Mat srcImg, bool & errorMark, cv::Point offset = cv::Point(0, 0), std::pair<bool, bool> bianOver = std::make_pair<bool,bool>(false,false));
//�ָ�ͼƬ�еĹ��������浽vector������
//���ؽṹ�б�

parted grow_part(cv::Mat & srcImg,cv::Point ptGrow,bool & errorMark);
//���ر��ָ�����Ĺ��������ָ���������������ص�Mat��1*1

cv::Mat shuchuxiangsuge(const cv::Mat srcImg);
//����test���������ص����ڹ۲�

int recognition1(cv::Mat srcImg);
//ʶ�𹤼����ͣ������ʹ�ӡ��ϸ��ͼ���ϡ�Բ����1����ĸ2��L���ݶ�����0
//����ʶ������

//xuanyu�����Ľӿں���


cv::Mat robert(cv::Mat srcImg);//Rober���ӱ�Ե���

bool screw_rec(cv::Mat srcImg);//ƽͷ�ݶ��ͼ�ͷ�ݶ�ʶ��ƽͷ����0����ͷ����1

cv::Mat PCA_rotate(std::list<cv::Point> listP,cv::Mat srcImg); //��תͼ��ʹ���PCA������ֱ,����ͼ��߶ȺͿ��

void quzhan(cv::Mat & src);

void hough_test_new(cv::Mat & src);

void hough_test(cv::Mat & src);

void yu_chu_li(cv::Mat & src);