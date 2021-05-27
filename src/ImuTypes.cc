/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "ImuTypes.h"
#include<iostream>

namespace ORB_SLAM3
{

namespace IMU
{

const float eps = 1e-4;

cv::Mat NormalizeRotation(const cv::Mat &R)
{
    cv::Mat U,w,Vt;
    cv::SVDecomp(R,w,U,Vt,cv::SVD::FULL_UV);
    return U*Vt;
}

cv::Mat Skew(const cv::Mat &v)
{
    const float x = v.at<float>(0);
    const float y = v.at<float>(1);
    const float z = v.at<float>(2);
    return (cv::Mat_<float>(3,3) << 0, -z, y,
            z, 0, -x,
            -y,  x, 0);
}

cv::Mat ExpSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    if(d<eps)
        return (I + W + 0.5f*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2);
}

Eigen::Matrix<double,3,3> ExpSO3(const double &x, const double &y, const double &z)
{
    Eigen::Matrix<double,3,3> I = Eigen::MatrixXd::Identity(3,3);
    const double d2 = x*x+y*y+z*z;
    const double d = sqrt(d2);
    Eigen::Matrix<double,3,3> W;
    W(0,0) = 0;
    W(0,1) = -z;
    W(0,2) = y;
    W(1,0) = z;
    W(1,1) = 0;
    W(1,2) = -x;
    W(2,0) = -y;
    W(2,1) = x;
    W(2,2) = 0;

    if(d<eps)
        return (I + W + 0.5*W*W);
    else
        return (I + W*sin(d)/d + W*W*(1.0-cos(d))/d2);
}

cv::Mat ExpSO3(const cv::Mat &v)
{
    return ExpSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}

cv::Mat LogSO3(const cv::Mat &R)
{
    const float tr = R.at<float>(0,0)+R.at<float>(1,1)+R.at<float>(2,2);
    cv::Mat w = (cv::Mat_<float>(3,1) <<(R.at<float>(2,1)-R.at<float>(1,2))/2,
                                        (R.at<float>(0,2)-R.at<float>(2,0))/2,
                                        (R.at<float>(1,0)-R.at<float>(0,1))/2);
    const float costheta = (tr-1.0f)*0.5f;
    if(costheta>1 || costheta<-1)
        return w;
    const float theta = acos(costheta);
    const float s = sin(theta);
    if(fabs(s)<eps)
        return w;
    else
        return theta*w/s;
}

cv::Mat RightJacobianSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    if(d<eps)
    {
        return cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        return I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

cv::Mat RightJacobianSO3(const cv::Mat &v)
{
    return RightJacobianSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}

cv::Mat InverseRightJacobianSO3(const float &x, const float &y, const float &z)
{
    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    if(d<eps)
    {
        return cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        return I + W/2 + W*W*(1.0f/d2 - (1.0f+cos(d))/(2.0f*d*sin(d)));
    }
}

cv::Mat InverseRightJacobianSO3(const cv::Mat &v)
{
    return InverseRightJacobianSO3(v.at<float>(0),v.at<float>(1),v.at<float>(2));
}


IntegratedRotation::IntegratedRotation(const cv::Point3f &angVel, const Bias &imuBias, const float &time):
    deltaT(time)
{
    //角速度减去偏置，再乘上时间，就是该时间内的旋转，xyz构成旋转向量
    const float x = (angVel.x-imuBias.bwx)*time;
    const float y = (angVel.y-imuBias.bwy)*time;
    const float z = (angVel.z-imuBias.bwz)*time;

    cv::Mat I = cv::Mat::eye(3,3,CV_32F);
    //计算旋转向量的模值
    const float d2 = x*x+y*y+z*z;
    const float d = sqrt(d2);
    //旋转向量(so3)写成反对称形式
    cv::Mat W = (cv::Mat_<float>(3,3) << 0, -z, y,
                 z, 0, -x,
                 -y,  x, 0);
    // eps = 1e-4 是一个小量，在ImuTypes.cc文件开头有定义
    if(d<eps)//旋转比较小，旋转向量到旋转矩阵的指数映射采用一阶近似
    {
        deltaR = I + W;
        //小量时，右扰动 Jr = I
        rightJ = cv::Mat::eye(3,3,CV_32F);
    }
    else
    {
        deltaR = I + W*sin(d)/d + W*W*(1.0f-cos(d))/d2;
        rightJ = I - W*(1.0f-cos(d))/d2 + W*W*(d-sin(d))/(d2*d);
    }
}

Preintegrated::Preintegrated(const Bias &b_, const Calib &calib)//构造函数
{
    Nga = calib.Cov.clone();//从标定得到的imu数据协方差矩阵
    NgaWalk = calib.CovWalk.clone();//从标定得到的bias随机游走协方差矩阵
    Initialize(b_);//根据bias初始化预积分相关的变量，bias改变时预积分会改变
}

// Copy constructor
Preintegrated::Preintegrated(Preintegrated* pImuPre): dT(pImuPre->dT), C(pImuPre->C.clone()), Info(pImuPre->Info.clone()),
    Nga(pImuPre->Nga.clone()), NgaWalk(pImuPre->NgaWalk.clone()), b(pImuPre->b), dR(pImuPre->dR.clone()), dV(pImuPre->dV.clone()),
    dP(pImuPre->dP.clone()), JRg(pImuPre->JRg.clone()), JVg(pImuPre->JVg.clone()), JVa(pImuPre->JVa.clone()), JPg(pImuPre->JPg.clone()),
    JPa(pImuPre->JPa.clone()), avgA(pImuPre->avgA.clone()), avgW(pImuPre->avgW.clone()), bu(pImuPre->bu), db(pImuPre->db.clone()), mvMeasurements(pImuPre->mvMeasurements)
{

}

void Preintegrated::CopyFrom(Preintegrated* pImuPre)
{
    std::cout << "Preintegrated: start clone" << std::endl;
    dT = pImuPre->dT;
    C = pImuPre->C.clone();
    Info = pImuPre->Info.clone();
    Nga = pImuPre->Nga.clone();
    NgaWalk = pImuPre->NgaWalk.clone();
    std::cout << "Preintegrated: first clone" << std::endl;
    b.CopyFrom(pImuPre->b);
    dR = pImuPre->dR.clone();
    dV = pImuPre->dV.clone();
    dP = pImuPre->dP.clone();
    JRg = pImuPre->JRg.clone();
    JVg = pImuPre->JVg.clone();
    JVa = pImuPre->JVa.clone();
    JPg = pImuPre->JPg.clone();
    JPa = pImuPre->JPa.clone();
    avgA = pImuPre->avgA.clone();
    avgW = pImuPre->avgW.clone();
    std::cout << "Preintegrated: second clone" << std::endl;
    bu.CopyFrom(pImuPre->bu);
    db = pImuPre->db.clone();
    std::cout << "Preintegrated: third clone" << std::endl;
    mvMeasurements = pImuPre->mvMeasurements;
    std::cout << "Preintegrated: end clone" << std::endl;
}


void Preintegrated::Initialize(const Bias &b_)
{
    dR = cv::Mat::eye(3,3,CV_32F);//旋转预积分初始值为单位阵
    dV = cv::Mat::zeros(3,1,CV_32F);//速度预积分初始为0
    dP = cv::Mat::zeros(3,1,CV_32F);//位置预积分初始为0
    JRg = cv::Mat::zeros(3,3,CV_32F);//旋转预积分对delta(bg)的雅克比，在bias改变后预积分量一阶近似更新模型中使用
    JVg = cv::Mat::zeros(3,3,CV_32F);//速度预积分对delta(bg)的雅克比
    JVa = cv::Mat::zeros(3,3,CV_32F);//速度预积分对delta(ba)的雅克比
    JPg = cv::Mat::zeros(3,3,CV_32F);//位置预积分对delta(bg)的雅克比
    JPa = cv::Mat::zeros(3,3,CV_32F);//位置预积分对delta(ba)的雅克比
    C = cv::Mat::zeros(15,15,CV_32F);//协方差传递所需的A,B矩阵，A为左上角9*9，B为右下角6*6
    Info=cv::Mat();//信息矩阵(协方差矩阵的逆)
    db = cv::Mat::zeros(6,1,CV_32F);//bias的变化量
    b=b_;
    bu=b_;
    avgA = cv::Mat::zeros(3,1,CV_32F);//平均加速度，用于判断加速度是否变化
    avgW = cv::Mat::zeros(3,1,CV_32F);//平均角速度
    dT=0.0f;//时间间隔
    mvMeasurements.clear();
}

void Preintegrated::Reintegrate()
{
    std::unique_lock<std::mutex> lock(mMutex);
    const std::vector<integrable> aux = mvMeasurements;
    Initialize(bu);
    for(size_t i=0;i<aux.size();i++)
        IntegrateNewMeasurement(aux[i].a,aux[i].w,aux[i].t);
}

void Preintegrated::IntegrateNewMeasurement(const cv::Point3f &acceleration, const cv::Point3f &angVel, const float &dt)
{
    //将imu数据构造成一个integrable结构体，保存到mvMeasurements中
    mvMeasurements.push_back(integrable(acceleration,angVel,dt));

    // Position is updated firstly, as it depends on previously computed velocity and rotation.
    // Velocity is updated secondly, as it depends on previously computed rotation.
    // Rotation is the last to be updated.

    //Matrices to compute covariance
    //计算协方差传递所需的A、B矩阵，下面的计算详见forster论文附录A.7~A.9
    cv::Mat A = cv::Mat::eye(9,9,CV_32F);
    cv::Mat B = cv::Mat::zeros(9,6,CV_32F);
    // 减去偏置后的加速度、角速度
    cv::Mat acc = (cv::Mat_<float>(3,1) << acceleration.x-b.bax,acceleration.y-b.bay, acceleration.z-b.baz);
    cv::Mat accW = (cv::Mat_<float>(3,1) << angVel.x-b.bwx, angVel.y-b.bwy, angVel.z-b.bwz);
    // 计算平均加速度和角速度
    avgA = (dT*avgA + dR*acc*dt)/(dT+dt);
    avgW = (dT*avgW + accW*dt)/(dT+dt);

    // Update delta position dP and velocity dV (rely on no-updated delta rotation)
    //更新P、V的预积分量
    dP = dP + dV*dt + 0.5f*dR*acc*dt*dt;
    dV = dV + dR*acc*dt;

    // Compute velocity and position parts of matrices A and B (rely on non-updated delta rotation)
    cv::Mat Wacc = (cv::Mat_<float>(3,3) << 0, -acc.at<float>(2), acc.at<float>(1),
                                                   acc.at<float>(2), 0, -acc.at<float>(0),
                                                   -acc.at<float>(1), acc.at<float>(0), 0);
    A.rowRange(3,6).colRange(0,3) = -dR*dt*Wacc;
    A.rowRange(6,9).colRange(0,3) = -0.5f*dR*dt*dt*Wacc;
    A.rowRange(6,9).colRange(3,6) = cv::Mat::eye(3,3,CV_32F)*dt;
    B.rowRange(3,6).colRange(3,6) = dR*dt;
    B.rowRange(6,9).colRange(3,6) = 0.5f*dR*dt*dt;

    // Update position and velocity jacobians wrt bias correction
    // 预积分量对bias变化量的雅克比
    JPa = JPa + JVa*dt -0.5f*dR*dt*dt;
    JPg = JPg + JVg*dt -0.5f*dR*dt*dt*Wacc*JRg;
    JVa = JVa - dR*dt;
    JVg = JVg - dR*dt*Wacc*JRg;

    // Update delta rotation
    //对角速度进行积分，得到旋转变化量
    IntegratedRotation dRi(angVel,b,dt);
    // 旧的旋转预积分量乘上旋转变化量，归一化使其符合旋转矩阵的格式
    dR = NormalizeRotation(dR*dRi.deltaR);

    // Compute rotation parts of matrices A and B
    A.rowRange(0,3).colRange(0,3) = dRi.deltaR.t();
    B.rowRange(0,3).colRange(0,3) = dRi.rightJ*dt;

    // Update covariance
    C.rowRange(0,9).colRange(0,9) = A*C.rowRange(0,9).colRange(0,9)*A.t() + B*Nga*B.t();
    C.rowRange(9,15).colRange(9,15) = C.rowRange(9,15).colRange(9,15) + NgaWalk;

    // Update rotation jacobian wrt bias correction
    JRg = dRi.deltaR.t()*JRg - dRi.rightJ*dt;

    // Total integrated time
    dT += dt;
}

void Preintegrated::MergePrevious(Preintegrated* pPrev)
{
    if (pPrev==this)
        return;

    std::unique_lock<std::mutex> lock1(mMutex);
    std::unique_lock<std::mutex> lock2(pPrev->mMutex);
    Bias bav;
    bav.bwx = bu.bwx;
    bav.bwy = bu.bwy;
    bav.bwz = bu.bwz;
    bav.bax = bu.bax;
    bav.bay = bu.bay;
    bav.baz = bu.baz;

    const std::vector<integrable > aux1 = pPrev->mvMeasurements;
    const std::vector<integrable> aux2 = mvMeasurements;

    Initialize(bav);
    for(size_t i=0;i<aux1.size();i++)
        IntegrateNewMeasurement(aux1[i].a,aux1[i].w,aux1[i].t);
    for(size_t i=0;i<aux2.size();i++)
        IntegrateNewMeasurement(aux2[i].a,aux2[i].w,aux2[i].t);

}

void Preintegrated::SetNewBias(const Bias &bu_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    bu = bu_;

    db.at<float>(0) = bu_.bwx-b.bwx;
    db.at<float>(1) = bu_.bwy-b.bwy;
    db.at<float>(2) = bu_.bwz-b.bwz;
    db.at<float>(3) = bu_.bax-b.bax;
    db.at<float>(4) = bu_.bay-b.bay;
    db.at<float>(5) = bu_.baz-b.baz;
}

IMU::Bias Preintegrated::GetDeltaBias(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    return IMU::Bias(b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz,b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
}

//bias更新之后新的旋转预积分(一阶线性模型)
cv::Mat Preintegrated::GetDeltaRotation(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    // 计算偏置的变化量
    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    return NormalizeRotation(dR*ExpSO3(JRg*dbg));//一阶近似更新
}

//bias更新之后新的速度预积分(一阶线性模型)
cv::Mat Preintegrated::GetDeltaVelocity(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    cv::Mat dba = (cv::Mat_<float>(3,1) << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz);
    return dV + JVg*dbg + JVa*dba;//一阶近似更新
}

//bias更新之后新的平移预积分(一阶线性模型)
cv::Mat Preintegrated::GetDeltaPosition(const Bias &b_)
{
    std::unique_lock<std::mutex> lock(mMutex);
    cv::Mat dbg = (cv::Mat_<float>(3,1) << b_.bwx-b.bwx,b_.bwy-b.bwy,b_.bwz-b.bwz);
    cv::Mat dba = (cv::Mat_<float>(3,1) << b_.bax-b.bax,b_.bay-b.bay,b_.baz-b.baz);
    return dP + JPg*dbg + JPa*dba;//一阶近似更新
}

cv::Mat Preintegrated::GetUpdatedDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return NormalizeRotation(dR*ExpSO3(JRg*db.rowRange(0,3)));
}

cv::Mat Preintegrated::GetUpdatedDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV + JVg*db.rowRange(0,3) + JVa*db.rowRange(3,6);
}

cv::Mat Preintegrated::GetUpdatedDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP + JPg*db.rowRange(0,3) + JPa*db.rowRange(3,6);
}

cv::Mat Preintegrated::GetOriginalDeltaRotation()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dR.clone();
}

cv::Mat Preintegrated::GetOriginalDeltaVelocity()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dV.clone();
}

cv::Mat Preintegrated::GetOriginalDeltaPosition()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return dP.clone();
}

Bias Preintegrated::GetOriginalBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return b;
}

Bias Preintegrated::GetUpdatedBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return bu;
}

cv::Mat Preintegrated::GetDeltaBias()
{
    std::unique_lock<std::mutex> lock(mMutex);
    return db.clone();
}

Eigen::Matrix<double,15,15> Preintegrated::GetInformationMatrix()
{
    std::unique_lock<std::mutex> lock(mMutex);
    if(Info.empty())
    {
        Info = cv::Mat::zeros(15,15,CV_32F);
        Info.rowRange(0,9).colRange(0,9)=C.rowRange(0,9).colRange(0,9).inv(cv::DECOMP_SVD);
        for(int i=9;i<15;i++)
            Info.at<float>(i,i)=1.0f/C.at<float>(i,i);
    }

    Eigen::Matrix<double,15,15> EI;
    for(int i=0;i<15;i++)
        for(int j=0;j<15;j++)
            EI(i,j)=Info.at<float>(i,j);
    return EI;
}

void Bias::CopyFrom(Bias &b)
{
    bax = b.bax;
    bay = b.bay;
    baz = b.baz;
    bwx = b.bwx;
    bwy = b.bwy;
    bwz = b.bwz;
}

std::ostream& operator<< (std::ostream &out, const Bias &b)
{
    if(b.bwx>0)
        out << " ";
    out << b.bwx << ",";
    if(b.bwy>0)
        out << " ";
    out << b.bwy << ",";
    if(b.bwz>0)
        out << " ";
    out << b.bwz << ",";
    if(b.bax>0)
        out << " ";
    out << b.bax << ",";
    if(b.bay>0)
        out << " ";
    out << b.bay << ",";
    if(b.baz>0)
        out << " ";
    out << b.baz;

    return out;
}

void Calib::Set(const cv::Mat &Tbc_, const float &ng, const float &na, const float &ngw, const float &naw)
{
    //相机->imu的变换矩阵
    Tbc = Tbc_.clone();
    //imu->相机的变换矩阵
    Tcb = cv::Mat::eye(4,4,CV_32F);
    Tcb.rowRange(0,3).colRange(0,3) = Tbc.rowRange(0,3).colRange(0,3).t();
    Tcb.rowRange(0,3).col(3) = -Tbc.rowRange(0,3).colRange(0,3).t()*Tbc.rowRange(0,3).col(3);
    Cov = cv::Mat::eye(6,6,CV_32F);//na、ng、naw、ngw都是标定的imu参数
    const float ng2 = ng*ng; //用它们来构造协方差矩阵
    const float na2 = na*na; //后面预积分时会用到
    Cov.at<float>(0,0) = ng2;
    Cov.at<float>(1,1) = ng2;
    Cov.at<float>(2,2) = ng2;
    Cov.at<float>(3,3) = na2;
    Cov.at<float>(4,4) = na2;
    Cov.at<float>(5,5) = na2;
    CovWalk = cv::Mat::eye(6,6,CV_32F);
    const float ngw2 = ngw*ngw;
    const float naw2 = naw*naw;
    CovWalk.at<float>(0,0) = ngw2;
    CovWalk.at<float>(1,1) = ngw2;
    CovWalk.at<float>(2,2) = ngw2;
    CovWalk.at<float>(3,3) = naw2;
    CovWalk.at<float>(4,4) = naw2;
    CovWalk.at<float>(5,5) = naw2;
}

Calib::Calib(const Calib &calib)
{
    Tbc = calib.Tbc.clone();
    Tcb = calib.Tcb.clone();
    Cov = calib.Cov.clone();
    CovWalk = calib.CovWalk.clone();
}

} //namespace IMU

} //namespace ORB_SLAM2
