#include "kalman_filter.h"
#include <iostream>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
	x_ = F_ * x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
	VectorXd diff = z - H_ * x_;
	MatrixXd S = H_ * P_ * H_.transpose() + R_;
	//cout << "S inverse: " << S.inverse() << endl;
	MatrixXd K = P_ * H_.transpose() * S.inverse();
	//cout << "KalmanGain: " << K << endl;
	x_ = x_ + K * diff;
	//cout << "after updating x: " << x_ << endl;
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
	float px = x_[0];
	float py = x_[1];
	float vx = x_[2];
	float vy = x_[3];
	// to polar 
	float cur_r = sqrt(px*px + py*py);
	float cur_angle = atan2(py, px);
	float cur_ang_vel = (px * vx + py * vy) / sqrt(px*px + py*py);
	// calculate angle diff 
	float pi = 3.1415926;
	float angle_diff;
	if (z[1] > 0 && cur_angle < 0) angle_diff = z[1] - (2 * pi + cur_angle);
	else if (z[1] < 0 && cur_angle > 0) angle_diff = (2 * pi + z[1]) - cur_angle;
	else angle_diff = z[1] - cur_angle;
	VectorXd diff = VectorXd(3);
	diff << z[0] - cur_r, angle_diff, z[2] - cur_ang_vel;

	//cout << "Raw angle: " << z[1] 
	//	<< "   " << "Calculated Angle: " << atan2(py, px) 
	//	<< "   " << "Angle Difference: " << angle_diff << endl;
	MatrixXd S = H_ * P_ * H_.transpose() + R_;
	//cout << "S inverse: " << S.inverse() << endl;
	MatrixXd K = P_ * H_.transpose() * S.inverse();
	//cout << "KalmanGain: " << K << endl;
	x_ = x_ + K * diff;
	//cout << "after updating x: " << x_ << endl;
	MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
	P_ = (I - K * H_) * P_;
}
