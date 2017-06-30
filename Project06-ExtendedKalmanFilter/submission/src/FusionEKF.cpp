#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
	  0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
		float rho = measurement_pack.raw_measurements_[0];
		float phi = measurement_pack.raw_measurements_[1];
		float rho_dot = measurement_pack.raw_measurements_[2];
		float x = rho * cos(phi);
		float y = rho * sin(phi);
		float vx = rho_dot * cos(phi);
		float vy = rho_dot * sin(phi);
		ekf_.x_ << x, y, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;
	previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  
  float delta_time = measurement_pack.timestamp_ - previous_timestamp_;
  delta_time /= 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, delta_time, 0,
	  0, 1, 0, delta_time,
	  0, 0, 1, 0,
	  0, 0, 0, 1;

  float noise_ax = 9.0;
  float noise_ay = 9.0;
  float power_2 = pow(delta_time, 2);
  float power_3 = pow(delta_time, 3)/2.0;
  float power_4 = pow(delta_time, 4)/4.0;
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << power_4 *noise_ax, 0,  power_3 *noise_ax, 0,
	  0, power_4 *noise_ay, 0, power_3 *noise_ay,
	  power_3 *noise_ax, 0, power_2 *noise_ax, 0,
	  0, power_3 *noise_ay, 0, power_2 *noise_ay;

  ekf_.Predict();
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
	  ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	  ekf_.R_ = R_radar_;
	  ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	  ekf_.H_ = H_laser_;
	  ekf_.R_ = R_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  //cout << "-----------------" << endl;
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "F_ = " << ekf_.F_ << endl;
  //cout << "Q_ = " << ekf_.Q_ << endl;
  //cout << "Measurement =" << measurement_pack.raw_measurements_ << endl;
  //cout << "H_ = " << ekf_.H_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
  //cout << "-----------------" << endl;
}
