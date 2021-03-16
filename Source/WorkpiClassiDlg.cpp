
// WorkpiClassiDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "WorkpiClassi.h"
#include "WorkpiClassiDlg.h"
#include "afxdialogex.h"



#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CWorkpiClassiDlg 对话框



CWorkpiClassiDlg::CWorkpiClassiDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_WORKPICLASSI_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CWorkpiClassiDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CWorkpiClassiDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(Button_ActivateCamera, &CWorkpiClassiDlg::OnBnClickedActivatecamera)
	ON_BN_CLICKED(Button_Initialize, &CWorkpiClassiDlg::OnBnClickedInitialize)
	ON_BN_CLICKED(Button_Run, &CWorkpiClassiDlg::OnBnClickedRun)
	ON_BN_CLICKED(Button_Config, &CWorkpiClassiDlg::OnBnClickedConfig)
END_MESSAGE_MAP()


// CWorkpiClassiDlg 消息处理程序

BOOL CWorkpiClassiDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	list = (CListCtrl*)GetDlgItem(List_Result);
	list->SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES);
	list->InsertColumn(0, _T("工件名"), LVCFMT_LEFT, 120);
	list->InsertColumn(1, _T("个数"), LVCFMT_LEFT, 60);

	viewSlct = (CComboBox*)GetDlgItem(ViewSelector);
	viewSlct->AddString(_T("shibie"));
	viewSlct->AddString(_T("original"));
	viewSlct->AddString(_T("shibie1"));
	viewSlct->AddString(_T("division"));

	cvNamedWindow("WorkpiClassiView", CV_WINDOW_AUTOSIZE);
	HWND hwnd = static_cast<HWND>(cvGetWindowHandle("WorkpiClassiView"));
	HWND parent = ::GetParent(hwnd);
	::SetParent(hwnd, GetDlgItem(ImageDisplay)->GetSafeHwnd());
	::ShowWindow(parent, SW_HIDE);


	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CWorkpiClassiDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CWorkpiClassiDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}


//初始化阶段显示
int __stdcall StreamCB(MV_IMAGE_INFO *pInfo, ULONG_PTR nUserVal)
{
	CWorkpiClassiDlg *pDlg = (CWorkpiClassiDlg *)nUserVal;
	return (pDlg->OnStreamCB(pInfo));
}

int CWorkpiClassiDlg::OnStreamCB(MV_IMAGE_INFO* pInfo) {
	MVInfo2Image(cam, pInfo, &currentFrame);
	DrawImage();
	return 0;
}

void CWorkpiClassiDlg::DrawImage() {
	auto dc = GetDC();
	currentFrame.Draw(dc->GetSafeHdc(), dispx, dispy, camw * dispresize, camh * dispresize);
	ReleaseDC(dc);
}

//开启相机
void CWorkpiClassiDlg::OnBnClickedActivatecamera()
{
	int nCams = 0;
	MVGetNumOfCameras(&nCams);
	if (nCams == 0) {
		AfxMessageBox(_T("没有找到相机"), MB_OK | MB_ICONHAND, 0);
		return;
	}

	MVSTATUS_CODES r = MVOpenCamByIndex(0, &cam);
	if (cam == NULL) {
		if (r == MVST_ACCESS_DENIED) {
			AfxMessageBox(_T("无法访问相机"), MB_OK | MB_ICONHAND, 0);
		}
		else {
			AfxMessageBox(_T("打开相机失败"), MB_OK | MB_ICONHAND, 0);
		}
		return;
	}

	TriggerModeEnums enumMode;
	MVGetTriggerMode(cam, &enumMode);
	if (enumMode != TriggerMode_Off)
	{
		//设置为连续非触发模式
		MVSetTriggerMode(cam, TriggerMode_Off);
	}

	MVGetWidth(cam, &camw);
	MVGetHeight(cam, &camh);
	MV_PixelFormatEnums pixelFormat;
	MVGetPixelFormat(cam, &pixelFormat);
	currentFrame.CreateByPixelFormat(camw, camh, pixelFormat);

	GetDlgItem(Button_Initialize)->EnableWindow(true);
	GetDlgItem(Button_ActivateCamera)->EnableWindow(false);
	GetDlgItem(Button_Config)->EnableWindow(true);

	isCamRun = true;
	MVStartGrab(cam, StreamCB, (ULONG_PTR)this);

	if (camProp == NULL) {
		MVCamProptySheetInit(&camProp, cam, this, _T("相机设置"));
	}
	if (camProp != NULL)
	{
		MVCamProptySheetCameraRun(camProp, MVCameraRun_ON);
	}

	appState = 1;
}

void CWorkpiClassiDlg::OnDestroy() {
	CDialog::OnDestroy();
	if (cam != NULL) {
		if (isCamRun) {
			MVStopGrab(cam);
			isCamRun = false;
		}
		MVCloseCam(cam);
	}
	if (camProp != NULL)
	{
		//销毁属性页对话框
		MVCamProptySheetDestroy(camProp);
		camProp = NULL;
	}
	MVTerminateLib();
}


void CWorkpiClassiDlg::OnBnClickedInitialize()
{
	MVStopGrab(cam);
	if (camProp != NULL)
	{
		MVCamProptySheetCameraRun(camProp, MVCameraRun_OFF);
	}

	isCamRun = false;
	auto r = MVSingleGrab(cam, &currentFrame, 1000);
	if (r != MVST_SUCCESS) {
		AfxMessageBox(_T("单帧采集失败"), MB_OK | MB_ICONHAND, 0);
		return;
	}

	auto bwimage = InitArea();
	//DrawImage();
	cv::resize(bwimage, bwimage, cv::Size(), 0.5, 0.5);
	cv::imshow("WorkpiClassiView", bwimage);

	GetDlgItem(Button_Run)->EnableWindow(true);

	appState = 2;
}

void CWorkpiClassiDlg::RefreshList(std::list<ListEntry> &items) {
	int count = 0;
	for (auto i : items) {
		LPCWSTR conv = stringToLPCWSTR(i.name);
		list->InsertItem(count, conv);
		list->SetItemText(count, 1, std::to_wstring(i.count).c_str());
		count++;
	}
}

cv::Mat CWorkpiClassiDlg::InitArea()
{
	auto currentMat = convertMVImage(currentFrame);
	auto bwimage = ThresholdNew(currentMat, IterThreshold(currentMat), true);
	auto pts = seed_grow(bwimage);
	areaPP = (1.5 * 1.5) / pts.size();
	return bwimage;
}

std::list<ListEntry> CWorkpiClassiDlg::shibie(cv::Mat & source_image)
{
	static bool initFlag = true;
	static SampleEntryManager manager;
	if (initFlag) {
		manager.addSamplesFromFile("samples/sampleList.txt");
		initFlag = false;
	}
	const int HOUGHJIAO = 70;//可调参数，hough直线检测的交点个数阈值
	bool errorMark = 0;      //错误判断，当有工件部分在图像范围内或者当有外界物体伸入图像时为1

	cv::resize(source_image, source_image, cv::Size(), 0.5, 0.5);

	cv::Mat grayImage;
	cv::cvtColor(source_image, grayImage, cv::COLOR_BGR2GRAY, 1);

//	cv::Mat source_image = cv::imread("test31.bmp", 0);//设置0单通道灰度图像读入

	std::list<parted> partedTools;               //分割出来的工件结构
	std::vector<cv::Vec2f> lines;                //用于hough直线提取
	std::list<parted> partedTools_real;

	cv::Mat er_image = ThresholdNew(grayImage, IterThreshold(grayImage), true); //双峰二值化

	multiSkeletonConcaveDivision(er_image);

	partedTools = separation(er_image, errorMark);//分割出来的工件的结构队列
	/*
	for (auto &Tools : partedTools) {
		//cv::imshow("1", Tools.pic);
		//cv::waitKey();

		multiSkeletonConcaveDivision(Tools.pic);
		
		auto temp = separation(Tools.pic, errorMark, Tools.zuoshang, std::pair<bool, bool>(true, Tools.bian));

		partedTools_real.splice(partedTools_real.end(), temp);
	}
	*/
	partedTools_real = partedTools;
	std::list<ListEntry> li;


	for (auto &Tools : partedTools_real) {
		if (Tools.bian == true) continue;
		Tools.name = manager.classify(Tools.pic)->name;
		auto temp = std::find_if(
			li.begin(),
			li.end(),
			[&](ListEntry a) {
			return a.name == Tools.name;
		}
		);
		if (temp == li.end()) {
			li.push_back(ListEntry(Tools.name, 1));
		}
		else {
			temp->count++;
		}
		char str[20];
		sprintf_s(str, "%.2f cm2", Tools.pixNum * areaPP);

		putTextZH(source_image, Tools.name.c_str(), cv::Point(Tools.zuoshang.y-5, Tools.zuoshang.x - 5), textColor, 18, "华文细黑", false, true);
		putTextZH(source_image, str, cv::Point(Tools.zuoshang.y -5, Tools.zuoshang.x + 15), textColor, 15, "华文细黑");
		//cv::putText(source_image, str, cv::Point(Tools.zuoshang.y, Tools.zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255));
		cv::rectangle(source_image, cv::Rect(Tools.zuoshang.y - 10, Tools.zuoshang.x - 10, Tools.width, Tools.height), boxColor, 2);
		//cv::putText(source_image, Tools.name, cv::Point(Tools.zuoshang.y, Tools.zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255));
	}

	return li;
}

std::list<ListEntry> CWorkpiClassiDlg::shibie1(cv::Mat & source_image)
{
	const int HOUGHJIAO = 60;//可调参数，hough直线检测的交点个数阈值
	bool errorMark = 0;      //错误判断，当有工件部分在图像范围内或者当有外界物体伸入图像时为1

	cv::resize(source_image, source_image, cv::Size(), 0.5, 0.5);

	cv::Mat grayImage;
	cv::cvtColor(source_image, grayImage, cv::COLOR_BGR2GRAY, 1);

//	cv::Mat source_image = cv::imread("test31.bmp", 0);//设置0单通道灰度图像读入
	std::list<parted> partedTools;               //分割出来的工件结构
	std::vector<cv::Vec2f> lines;                //用于hough直线提取
	std::list<parted> partedTools_real;

	cv::Mat er_image = ThresholdNew(grayImage, IterThreshold(grayImage), true); //双峰二值化
//	binary_img(er_image);
	//cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5), cv::Point(-1, -1));
	//erode(er_image, er_image, element);//腐蚀

	multiSkeletonConcaveDivision(er_image);

	partedTools = separation(er_image, errorMark);//分割出来的工件的结构队列
	/*
	for (auto &Tools : partedTools) {
		multiSkeletonConcaveDivision(Tools.pic);
	//	cv::imshow("1", Tools.pic);
	//	cv::waitKey();
		auto temp = separation(Tools.pic, errorMark, Tools.zuoshang, std::pair<bool, bool>(true, Tools.bian));
		partedTools_real.splice(partedTools_real.end(), temp);
	}
	*/
	partedTools_real = partedTools;
	std::list<ListEntry> li;
	li.push_back(ListEntry("螺母", 0));
	li.push_back(ListEntry("圆", 0));
	li.push_back(ListEntry("六角扳手", 0));
	li.push_back(ListEntry("平头螺钉", 0));
	li.push_back(ListEntry("尖头螺钉", 0));

	//if (errorMark == 1) {                        //此时说明工件有部分在图像外，或者有外界伸入检测区域
	//	cv::putText(source_image, "Error", cv::Point(100, 160), CV_FONT_HERSHEY_SIMPLEX, 3.0, cv::Scalar(255), 3);
	//}
//	else {

	for (std::list<parted>::iterator pw = partedTools_real.begin(); pw != partedTools_real.end(); pw++) {
		cv::Mat thinPic = (*pw).pic.clone();
		thinImage(thinPic);                //图像细化
		(*pw).thinPic = thinPic;           //存取细化图像
		(*pw).mark = recognition1(thinPic);//工件初步识别，识别圆和螺母。螺钉和L件分为一类
	//	imshow("2", pw->pic);
	//	cv::waitKey();
	}

	for (std::list<parted>::iterator pw = partedTools_real.begin(); pw != partedTools_real.end(); pw++) {
		if (pw->bian == true) continue;
		cv::rectangle(source_image, cv::Rect((*pw).zuoshang.y - 10, (*pw).zuoshang.x - 10, (*pw).width, (*pw).height), boxColor, 2);
		if ((*pw).mark == 1) {//已识别为圆
			auto temp = std::find_if(
				li.begin(),
				li.end(),
				[](ListEntry a) {
				return a.name == "圆";
			}
			);
			temp->count++;
			char str[20];
			sprintf_s(str, "%.2f cm2", pw->pixNum * areaPP);
			cv::putText(source_image, str, cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
			
			cv::putText(source_image, "Circle", cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
		}
		else if ((*pw).mark == 2) {//识别为螺母
			auto temp = std::find_if(
				li.begin(),
				li.end(),
				[](ListEntry a) {
				return a.name == "螺母";
			}
			);
			temp->count++;
			char str[20];
			sprintf_s(str, "%.2f cm2", pw->pixNum * areaPP);
			cv::putText(source_image, str, cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
			cv::putText(source_image, "Nut", cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
		}
		else if ((*pw).mark == 0) {//L or D
			cv::HoughLines((*pw).thinPic, lines, 1, CV_PI / 180, HOUGHJIAO);//HOUGHJIAO阈值是当相交的点数大于HOUGHJIAO时才判为直线

			std::vector<cv::Vec2f>::const_iterator it = lines.begin();
			std::vector<cv::Vec2f>::const_iterator jt;
			int first = 1;
			if (lines.size() >= 2) {
				for (it; it != lines.end() - 1;) {
					first = 1;
					float theta1 = (*it)[1];//第一类直线
					for (jt = it + 1; jt != lines.end();) {
						float theta = (*jt)[1];
						if (std::fabs(theta - theta1) < CV_PI / 4) {
							jt = lines.erase(jt);     //注意这个erase易错
						}
						else {
							if (first == 1) it = jt, first = 0;
							jt++;
						};//找到了第二类直线
					}
				}
			}            //将相似角度直线合并

			if (lines.size() >= 2) {//L
				auto temp = std::find_if(
					li.begin(),
					li.end(),
					[](ListEntry a) {
					return a.name == "六角扳手";
				}
				);
				temp->count++;
				char str[20];
				sprintf_s(str, "%.2f cm2", pw->pixNum * areaPP);
				cv::putText(source_image, str, cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
				cv::putText(source_image, "Lpiece", cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
				
			}
			else if (lines.size() <= 1) {//螺钉
				
				if (screw_rec((*pw).pic) == 1) {//尖头螺钉
					auto temp = std::find_if(
						li.begin(),
						li.end(),
						[](ListEntry a) {
						return a.name == "尖头螺钉";
					}
					);
					temp->count++;
					char str[20];
					sprintf_s(str, "%.2f cm2", pw->pixNum * areaPP);
					cv::putText(source_image, str, cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
					cv::putText(source_image, "ScrewJ", cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
				}
				else {                           //平头螺钉
					auto temp = std::find_if(
						li.begin(),
						li.end(),
						[](ListEntry a) {
						return a.name == "平头螺钉";
					}
					);
					temp->count++;
					char str[20];
					sprintf_s(str, "%.2f cm2", pw->pixNum * areaPP);
					cv::putText(source_image, str, cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15 + 20), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
					cv::putText(source_image, "ScrewP", cv::Point((*pw).zuoshang.y, (*pw).zuoshang.x + 15), CV_FONT_HERSHEY_SIMPLEX, 0.6, textColor);
				}
			}
		}
	}
	//	} //工件摆放合适且没有外界伸入


	//	cv::namedWindow("result", 1);
	//	imshow("result", source_image);
	cv::waitKey(20);
	return li;
}

//工作阶段显示
int __stdcall StreamCB_Alt(MV_IMAGE_INFO *pInfo, ULONG_PTR nUserVal)
{
	CWorkpiClassiDlg *pDlg = (CWorkpiClassiDlg *)nUserVal;
	return (pDlg->OnStreamCB_Alt(pInfo));
}

int CWorkpiClassiDlg::OnStreamCB_Alt(MV_IMAGE_INFO* pInfo) {
	MVInfo2Image(cam, pInfo, &currentFrame);

	DrawImage_Alt();
	return 0;
}

void CWorkpiClassiDlg::DrawImage_Alt() {
	auto currentMat = convertMVImageColor(currentFrame);

	refreshTick++;
	if (refreshTick == 10) {
		list->DeleteAllItems();
	}
	std::list<ListEntry> li;

	switch (viewSlct->GetCurSel()) {
	default:
	case 0://shibie
		li = shibie(currentMat);
		if (refreshTick == 10) {
			RefreshList(li);
		}
		break;
	case 1://houghTest
		//hough_test_new(currentMat);
		cv::resize(currentMat, currentMat, cv::Size(), 0.5, 0.5);
		break;
	case 2://shibie1
		li = shibie1(currentMat);
		if (refreshTick == 10) {
			RefreshList(li);
		}

		break;
	case 3://division
		cv::resize(currentMat, currentMat, cv::Size(), 0.5, 0.5);
		cv::cvtColor(currentMat, currentMat, cv::COLOR_BGR2GRAY, 1);
		Threshold(currentMat, IterThreshold(currentMat), true);
		multiSkeletonConcaveDivision(currentMat);
		//curvatureConcaveDivision(currentMat);
		break;
	}

	if (refreshTick == 10)
		refreshTick = 0;

	cv::imshow("WorkpiClassiView", currentMat);
}

void CWorkpiClassiDlg::OnBnClickedRun()
{
	GetDlgItem(Button_Run)->EnableWindow(false);
	GetDlgItem(ImageDisplay)->EnableWindow(true);
	GetDlgItem(ImageDisplay)->SetWindowPos(NULL, dispx, dispy, camw * dispresize, camh * dispresize, SWP_NOMOVE);



	isCamRun = true;
	MVStartGrab(cam, StreamCB_Alt, (ULONG_PTR)this);

	if (camProp != NULL)
	{
		MVCamProptySheetCameraRun(camProp, MVCameraRun_ON);
	}

	appState = 3;
}

void CWorkpiClassiDlg::OnBnClickedConfig()
{
	// TODO: 在此添加控件通知处理程序代码
	if (camProp != NULL)
	{
		MVCamProptySheetShow(camProp, SW_SHOW);
	}
}
