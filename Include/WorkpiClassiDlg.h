
// WorkpiClassiDlg.h: 头文件
//

#pragma once

#include "shenchong.h"
#include "xuanyu.h"
#include "putText.h"

// CWorkpiClassiDlg 对话框
class CWorkpiClassiDlg : public CDialogEx
{
// 构造
public:
	CWorkpiClassiDlg(CWnd* pParent = nullptr);	// 标准构造函数

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_WORKPICLASSI_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg void OnDestroy();
	void DrawImage();
	void DrawImage_Alt();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

	//我们的东西
	unsigned short appState = 0;//程序的状态，0-刚开启，1-相机连接了，2-初始化了，3-工作中

	const float dispresize = 0.5;
	int camw, camh;
	HANDLE cam;
	bool isCamRun = false;

	HANDLE camProp;

	MVImage currentFrame;

	double areaPP;

	uint refreshTick = 0;
	
	CListCtrl* list;
	bool listUpdateFlag = false;
	CComboBox* viewSlct;

	const int dispx = 200, dispy = 50;
	const cv::Scalar boxColor = cv::Scalar(255, 0, 0);
	const cv::Scalar textColor = cv::Scalar(255, 150, 0);
public:
	afx_msg
		int OnStreamCB(MV_IMAGE_INFO * pInfo);
	int OnStreamCB_Alt(MV_IMAGE_INFO * pInfo);

	afx_msg void RefreshList(std::list<ListEntry> &items);
	afx_msg cv::Mat InitArea();//1.5*1.5cm
	
	std::list<ListEntry> shibie(cv::Mat & source_image);
	std::list<ListEntry> shibie1(cv::Mat & source_image);
	
	void OnBnClickedActivatecamera();



	afx_msg void OnBnClickedInitialize();
	afx_msg void OnBnClickedRun();
	afx_msg void OnBnClickedConfig();
};
